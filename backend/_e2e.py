import glob, json, sys
from pathlib import Path
from app.certify import certify
from app.pipeline import reconcile, solve
from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, SceneGraph, SegmentResult, SolveWeights

ctx = PipelineContext.create("room2", Path("storage/uploads/room2.png").resolve())
base = SceneGraph.model_validate_json(Path("storage/artifacts/4011e138deaa828f.json").read_text())
load = lambda p: next(d for d in (json.load(open(f)) for f in glob.glob("storage/artifacts/*.json")) if p(d))
dep = DepthResult.model_validate(load(lambda d: isinstance(d, dict) and "is_metric" in d and "/room2/" in d["depth_path"]))
seg = SegmentResult.model_validate(load(lambda d: isinstance(d, dict) and "masks" in d and len(d["masks"]) == 19))

for label, g in (("supports as reconcile left them", base.model_copy(deep=True)),
                 ("supports re-resolved after stage 9", reconcile.resolve_supports(base.model_copy(deep=True), ctx.settings))):
    r = solve.run(ctx, g, dep, seg, SolveWeights())
    c = certify.run(ctx, r.graph)
    print(f"\n=== {label}")
    print(f"    settled={r.diagnostics.settled} converged={r.diagnostics.converged} "
          f"max settle drift {1000*r.diagnostics.max_settle_drift_m:.0f} mm  depth cost {r.diagnostics.residual_by_term.get('depth', 0):.3f}")
    print(f"    scale  {sum(1 for x in c.scale if x.passed)}/{len(c.scale)} pass   "
          f"stability {sum(1 for x in c.stability if x.passed)}/{len(c.stability)} pass")
    print(f"    worst |gap| {1000*max(abs(x.support_gap_m) for x in c.scale):.1f} mm   "
          f"worst penetration {1000*max(x.initial_penetration_m for x in c.stability):.2f} mm   "
          f"not touching declared parent: {sum(1 for x in c.scale if not x.touching_parent)}")
