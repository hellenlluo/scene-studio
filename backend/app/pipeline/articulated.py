from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, ReconstructionResult, SegmentResult

__all__ = ["ReconstructionResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> ReconstructionResult:
    """Branch 4b: part meshes plus joint hypotheses, via URDF reconstruction.

    SPARK, Articulate-Anything or URDF-Anything; parse whatever URDF comes back
    with yourdfpy into parts and JointHypothesis candidates. Keep the full ranked
    candidate set, not just the argmax — the viewer's joint-drag refits within it,
    and the top-1/top-2 margin is v2's joint confidence signal.

    Failure is expected and cheap: set ObjectAssets.failed and let the
    orchestrator fall the object back to 4a. An object routed here that comes
    back with zero joints degrades to exactly the rigid result, which is the
    whole reason the router is biased toward this branch.
    """
    raise NotImplementedError("articulated branch")
