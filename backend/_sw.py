import json, glob
from pathlib import Path
from app.certify import certify
from app.pipeline import solve
from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, SceneGraph, SegmentResult, SolveWeights
g0 = None
for f in sorted(glob.glob("storage/artifacts/*.json"), key=lambda p: Path(p).stat().st_mtime, reverse=True):
    d = json.load(open(f))
    if isinstance(d, dict) and "gravity_rotation" in d and len(d.get("objects", [])) == 16:
        if any((o["parts"][0].get("inertial") or {}).get("mass_kg") for o in d["objects"]):
            g0 = SceneGraph.model_validate(d); break
load = lambda p: next(d for d in (json.load(open(f)) for f in glob.glob("storage/artifacts/*.json")) if p(d))
dep = DepthResult.model_validate(load(lambda d: isinstance(d,dict) and "is_metric" in d and "/room2/" in d["depth_path"]))
seg = SegmentResult.model_validate(load(lambda d: isinstance(d,dict) and "masks" in d and len(d["masks"])==18))
ctx = PipelineContext.create("room2", Path("storage/uploads/room2.png").resolve())
print(f"{'weight':>7s} {'not-over':>9s} {'worst pen':>10s} {'drift':>9s} {'scale fail':>11s}")
for w in (0.0, 5.0, 15.0):
    r = solve.run(ctx, g0.model_copy(deep=True), dep, seg, SolveWeights(containment=w))
    c = certify.run(ctx, r.graph)
    print(f"{w:7.0f} {sum(1 for x in c.scale if not x.base_inside_parent):6d}/{len(c.scale)} "
          f"{1000*max(x.initial_penetration_m for x in c.stability):8.1f} mm "
          f"{1000*r.diagnostics.max_settle_drift_m:6.0f} mm {sum(1 for x in c.scale if not x.passed):8d}/{len(c.scale)}")
