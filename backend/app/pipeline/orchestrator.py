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
    reconcile,
    rigid,
    segment,
    solve,
    verify,
)
from app.pipeline.base import PipelineContext, cache_key, load_cached, store_cached
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
        (concepts,),
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
        (seg, ctx.settings.max_mesh_faces),
        report,
    )

    graph = _run_stage(
        ctx,
        StageName.RECONCILE,
        reconcile.SceneGraph,
        lambda: reconcile.run(ctx, rig, seg, dep, lab),
        (rig, seg, dep, lab),
        report,
    )
    verified = _run_stage(
        ctx,
        StageName.VERIFY,
        verify.VerifyResult,
        lambda: verify.run(ctx, graph),
        (graph,),
        report,
    )
    graph = verified.graph

    graph = _run_stage(
        ctx,
        StageName.INERTIA,
        inertia.SceneGraph,
        lambda: inertia.run(ctx, graph),
        (graph,),
        report,
    )

    solved = _run_stage(
        ctx,
        StageName.SOLVE,
        solve.SolveResult,
        lambda: solve.run(ctx, graph, dep, seg, weights, anchors),
        (graph, dep, seg, weights, anchors),
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
        (graph, ctx.settings.certification_thresholds()),
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
