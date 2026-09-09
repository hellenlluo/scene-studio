import glob, json
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

for tag, g in (("before", base.model_copy(deep=True)),
               ("after", reconcile.resolve_supports(base.model_copy(deep=True), ctx.settings))):
    Path(f"_solved_{tag}_pre.json").write_text(g.model_dump_json())
    r = solve.run(ctx, g, dep, seg, SolveWeights())
    Path(f"_solved_{tag}.json").write_text(r.graph.model_dump_json())
    Path(f"_diag_{tag}.json").write_text(r.diagnostics.model_dump_json())
    print(tag, "done")
