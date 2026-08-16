"""Stage sequencing.

Ordering and fallback policy live here; the stages themselves stay ignorant of
each other. A stage that fails with a declared fallback degrades that stage
rather than failing the whole scene — the OBB geometry path is the main case.
"""

import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from app.pipeline import (
    assemble,
    collision,
    depth,
    geometry,
    labeling,
    repair,
    scale,
    segment,
    validate,
)
from app.pipeline.base import PipelineContext, cache_key, load_cached, store_cached
from app.schemas import SceneSpec, StageName

StageReporter = Callable[[StageName, str, float | None], None]


def _run_stage[T: BaseModel](
    ctx: PipelineContext,
    stage: StageName,
    model: type[T],
    fn: Callable[[], T],
    upstream: BaseModel | None,
    report: StageReporter,
) -> T:
    key = cache_key(ctx, stage, upstream)
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
) -> SceneSpec:
    ctx = PipelineContext.create(job_id, image_path)

    seg = _run_stage(
        ctx, StageName.SEGMENT, segment.SegmentResult, lambda: segment.run(ctx), None, report
    )
    dep = _run_stage(ctx, StageName.DEPTH, depth.DepthResult, lambda: depth.run(ctx), None, report)
    lab = _run_stage(
        ctx, StageName.LABEL, labeling.LabelResult, lambda: labeling.run(ctx, seg), seg, report
    )
    scl = _run_stage(
        ctx, StageName.SCALE, scale.ScaleResult, lambda: scale.run(ctx, seg, dep, lab), lab, report
    )
    geo = _run_stage(
        ctx,
        StageName.GEOMETRY,
        geometry.GeometryResult,
        lambda: geometry.run(ctx, seg, scl),
        scl,
        report,
    )
    col = _run_stage(
        ctx,
        StageName.COLLISION,
        collision.GeometryResult,
        lambda: collision.run(ctx, geo),
        geo,
        report,
    )
    lay = _run_stage(
        ctx,
        StageName.ASSEMBLE,
        assemble.SceneLayout,
        lambda: assemble.run(ctx, col, dep, scl),
        col,
        report,
    )
    val = _run_stage(
        ctx,
        StageName.VALIDATE,
        validate.ValidationResult,
        lambda: validate.run(ctx, col, lay),
        lay,
        report,
    )

    # Repair only runs when validation actually found something, and revalidates
    # its own work — a scene that comes back valid after automatic repair is the
    # result worth reporting.
    repairs = []
    if val.pass_rate < 1.0:
        report(StageName.REPAIR, "running", None)
        lay, val, repairs = repair.run(ctx, col, lay, val)
        report(StageName.REPAIR, "done", None)

    return SceneSpec(
        scene_id=job_id,
        image_path=str(image_path),
        intrinsics=dep.intrinsics,
        scene_scale=scl.scene_scale,
        labels=lab.labels,
        geometries=col.geometries,
        layout=lay,
        validation=val,
        repairs_applied=repairs,
    )
