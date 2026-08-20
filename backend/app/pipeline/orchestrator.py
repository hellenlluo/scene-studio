"""Stage sequencing.

Ordering and fallback policy live here; the stages themselves stay ignorant of
each other. A stage that fails with a declared fallback degrades that stage
rather than failing the whole scene — the OBB geometry path and the
articulated-to-rigid fallback are the two that matter.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.certify import certify, repair
from app.export import gltf, mjcf, urdf
from app.pipeline import articulated, depth, inertia, labeling, reconcile, rigid, segment, solve
from app.pipeline.base import PipelineContext, cache_key, load_cached, store_cached
from app.schemas import (
    ExportResult,
    ReconstructionResult,
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

    seg = _run_stage(
        ctx, StageName.SEGMENT, segment.SegmentResult, lambda: segment.run(ctx), (), report
    )
    dep = _run_stage(ctx, StageName.DEPTH, depth.DepthResult, lambda: depth.run(ctx), (), report)
    lab = _run_stage(
        ctx, StageName.LABEL, labeling.LabelResult, lambda: labeling.run(ctx, seg), (seg,), report
    )

    # Part masks exist only to feed the articulated branch, so both are gated
    # together. With articulation off, every object is rigid and jointless — which
    # is a scope decision, not a degradation, and the certificate reports it as
    # such rather than as a wall of kinematic failures.
    if ctx.settings.enable_articulation:
        # Part masks need the routing decision, so this second segmentation pass
        # sits after labelling rather than alongside the first. Cached under the
        # same stage name — its inputs differ, so its key does too.
        seg = _run_stage(
            ctx,
            StageName.SEGMENT,
            segment.SegmentResult,
            lambda: segment.run_parts(ctx, seg, lab),
            (seg, lab),
            report,
        )

    rig = _run_stage(
        ctx,
        StageName.RIGID,
        rigid.ReconstructionResult,
        lambda: rigid.run(ctx, seg, lab),
        (seg, lab),
        report,
    )

    # Both branches run over the same inputs and each ignores the objects the
    # router did not send it. The articulated branch is a superset of the rigid
    # one, so an object it fails on falls back to 4a's output for the same id and
    # loses actuability rather than the scene.
    art = ReconstructionResult(objects=[])
    if ctx.settings.enable_articulation:
        art = _run_stage(
            ctx,
            StageName.ARTICULATED,
            articulated.ReconstructionResult,
            lambda: articulated.run(ctx, seg, lab),
            (seg, lab),
            report,
        )

    graph = _run_stage(
        ctx,
        StageName.RECONCILE,
        reconcile.SceneGraph,
        lambda: reconcile.run(ctx, rig, art, dep, lab),
        (rig, art, dep, lab),
        report,
    )
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
    exports = ExportResult(
        mjcf_path=str(mjcf.write_mjcf(graph, out)),
        urdf_path=str(urdf.write_urdf(graph, out)),
        gltf_path=str(gltf.write_gltf(graph, out)),
    )
    report(StageName.EXPORT, "done", None)

    return SceneSpec(
        scene_id=job_id,
        image_path=str(image_path),
        intrinsics=dep.intrinsics,
        graph=graph,
        certificate=cert,
        anchors=anchors,
        weights=weights,
        diagnostics=solved.diagnostics,
        repairs_applied=repairs,
        exports=exports,
    )
