"""Stage sequencing.

Ordering and fallback policy live here; the stages themselves stay ignorant of
each other. A stage that fails with a declared fallback degrades that stage
rather than failing the whole scene — the OBB geometry path is the main case.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.certify import certify, repair
from app.export import gltf, mjcf
from app.pipeline import (
    depth,
    inertia,
    labeling,
    occlusion,
    penetration,
    reconcile,
    rigid,
    segment,
    solve,
    support,
    verify,
)
from app.pipeline.base import (
    PipelineContext,
    cache_key,
    load_cached,
    source_fingerprint,
    store_cached,
)
from app.schemas import (
    ExportResult,
    ScaleAnchor,
    SceneSpec,
    SolveWeights,
    StageName,
)

StageReporter = Callable[[StageName, str, float | None], None]


def _run_stage[T: BaseModel](
    ctx: PipelineContext,
    stage: StageName,
    model: type[T],
    fn: Callable[[], T],
    inputs: tuple[object, ...],
    report: StageReporter,
) -> T:
    key = cache_key(ctx, stage, *inputs)
    if (cached := load_cached(ctx, key, model)) is not None:
        report(stage, "cached", 0.0)
        return cached

    report(stage, "running", None)
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started

    store_cached(ctx, key, result)
    report(stage, "done", elapsed)
    return result


def run_pipeline(
    job_id: str,
    image_path: Path,
    report: StageReporter = lambda *_: None,
    weights: SolveWeights | None = None,
    anchors: list[ScaleAnchor] | None = None,
) -> SceneSpec:
    ctx = PipelineContext.create(job_id, image_path)
    weights = weights or SolveWeights()
    anchors = anchors or []

    # Stage 3 straddles stage 1: SAM 3 is concept-prompted and needs a noun list,
    # but a label needs an object id and ids only exist once masks do. So the
    # inventory pass runs first and the labelling pass runs after segmentation.
    report(StageName.LABEL, "running", None)
    concepts = labeling.inventory(ctx)

    seg = _run_stage(
        ctx,
        StageName.SEGMENT,
        segment.SegmentResult,
        lambda: segment.run(ctx, concepts),
        (concepts, segment.dedup_fingerprint()),
        report,
    )
    dep = _run_stage(ctx, StageName.DEPTH, depth.DepthResult, lambda: depth.run(ctx), (), report)
    lab = _run_stage(
        ctx, StageName.LABEL, labeling.LabelResult, lambda: labeling.run(ctx, seg), (seg,), report
    )

    # Stage 3.5. Local and cheap, so it is not a cached stage of its own — but it
    # rewrites the mask set, so everything downstream takes `seg`/`lab` from here and
    # the reconstruct key below picks the change up for free.
    resolved = occlusion.run(ctx, seg, lab, dep)
    seg, lab = resolved.segments, resolved.labels

    # Keyed on the masks, not the labels. SAM 3D is given the photo and the masks and
    # nothing else — the labels never reach it — so including them bought nothing and
    # cost real money: `label` is a VLM call and not deterministic, so a rerun that
    # relabels the same masks differently invalidated this entry and paid fal again to
    # regenerate byte-identical meshes.
    #
    # `max_mesh_faces` *is* in the key, because it changes what this stage stores:
    # a mesh over the cap is decimated and one under it is not. Left out, raising the
    # cap to recover an object's collision surface would appear to do nothing, because
    # the decimated mesh would be served straight back from cache.
    rig = _run_stage(
        ctx,
        StageName.RECONSTRUCT,
        rigid.ReconstructionResult,
        lambda: rigid.run(ctx, seg),
        # Keyed on what SAM 3D is actually handed — the photo and the mask images —
        # rather than on the whole SegmentResult. Over-keying here has cost money
        # twice: once on labels the stage never read, and again when `ObjectMask`
        # grew a `concept` field that only the labelling pass looks at. Anything
        # added to the mask that SAM 3D *can* see has to be added to this tuple.
        (
            [(m.object_id, m.mask_path, m.area_px, m.bbox_px) for m in seg.masks],
            seg.image_size_px,
            ctx.settings.max_mesh_faces,
        ),
        report,
    )

    graph = _run_stage(
        ctx,
        StageName.RECONCILE,
        reconcile.SceneGraph,
        lambda: reconcile.run(ctx, rig, seg, dep, lab),
        (
            rig,
            seg,
            dep,
            lab,
            ctx.settings.support_settings(),
            source_fingerprint(reconcile, support),
        ),
        report,
    )
    verified = _run_stage(
        ctx,
        StageName.VERIFY,
        verify.VerifyResult,
        lambda: verify.run(ctx, graph),
        (graph, verify.prompts_fingerprint()),
        report,
    )
    graph = verified.graph

    graph = _run_stage(
        ctx,
        StageName.INERTIA,
        inertia.SceneGraph,
        lambda: inertia.run(ctx, graph),
        (graph, source_fingerprint(inertia)),
        report,
    )

    # Resolve supports a second time, now that collision geometry exists.
    #
    # `reconcile.run` already did this, and it had to guess: stage 9 is what produces
    # `collision_mesh_paths`, so at stage 5 every part has none and `support.build`
    # falls back to the visual mesh or the OBB box. The support *relation* was
    # therefore decided from geometry that no later stage uses — `E_supp`,
    # `certify.scale`, `certify.stability` and repair all measure the hulls. Same
    # "two sources of truth for the same shape" failure `app.pipeline.support` exists
    # to fix, one stage earlier and applied to the relation rather than the height.
    #
    # Measured on `room2.png`: 0 collision meshes at stage 5 against 263 after stage 9,
    # and re-running this unchanged on the post-stage-9 graph moves an armchair off the
    # floor and onto the rug it measurably rests on — 9.15 mm into it, which is what
    # every downstream stage was already reading while `E_supp` pulled it a further
    # 11.5 mm down toward a floor it was nowhere near. It also puts a book back on the
    # book it rests on rather than that book's own table.
    #
    # Inline rather than a cached stage of its own, following `occlusion` above: local,
    # pure and cheap next to one solve round. It needs no cache entry because it
    # rewrites the graph, and the graph is already an input to the solve key.
    graph = reconcile.resolve_supports(graph, ctx.settings)

    solved = _run_stage(
        ctx,
        StageName.SOLVE,
        solve.SolveResult,
        lambda: solve.run(ctx, graph, dep, seg, weights, anchors),
        (
            graph,
            dep,
            seg,
            weights,
            anchors,
            ctx.settings.support_settings(),
            source_fingerprint(solve, support, penetration),
        ),
        report,
    )
    graph = solved.graph

    cert = _run_stage(
        ctx,
        StageName.CERTIFY,
        certify.Certificate,
        lambda: certify.run(ctx, graph),
        # Thresholds are an input: retightening max_penetration_m must re-certify
        # rather than hand back the artifact from the looser run.
        (
            graph,
            ctx.settings.certification_thresholds(),
            source_fingerprint(certify, support, penetration),
        ),
        report,
    )

    # Repair only runs when certification actually found something, and
    # re-certifies its own work — a scene that comes back certified after
    # automatic repair is the result worth reporting.
    repairs = []
    if not cert.passed:
        report(StageName.REPAIR, "running", None)
        repaired = repair.run(ctx, graph, cert)
        graph, cert, repairs = repaired.graph, repaired.certificate, repaired.actions
        report(StageName.REPAIR, "done", None)

    report(StageName.EXPORT, "running", None)
    out = ctx.workdir()
    # Relative to storage_dir, because these are what the browser fetches through
    # the /storage mount and an absolute filesystem path is unusable to a client.
    exports = ExportResult(
        mjcf_path=ctx.settings.storage_relative(mjcf.write_mjcf(graph, out)),
        gltf_path=ctx.settings.storage_relative(gltf.write_gltf(graph, out)),
    )
    report(StageName.EXPORT, "done", None)

    return SceneSpec(
        scene_id=job_id,
        image_path=ctx.settings.storage_relative(image_path),
        intrinsics=dep.intrinsics,
        graph=graph,
        certificate=cert,
        anchors=anchors,
        weights=weights,
        diagnostics=solved.diagnostics,
        repairs_applied=repairs,
        removed_objects=verified.removed,
        exports=exports,
    )
