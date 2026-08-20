from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, ReconstructionResult, SegmentResult

__all__ = ["ReconstructionResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> ReconstructionResult:
    """Branch 4a: single-part mesh reconstruction, OBB proxy as the floor.

    Build the OBB path first and completely. It needs only the backprojected
    point cloud and the mask extent, it runs without a GPU, and it makes every
    downstream stage — reconcile, solve, certify, repair, export — exercisable
    end to end before any hosted model is wired up.

    Mesh reconstruction (TRELLIS / Hunyuan3D-2 / InstantMesh over the inpainted
    crop) then upgrades proxy_tier per object, and must stay optional: when it
    fails the object keeps its OBB and the scene stays valid.

    Emits one part per object, so the result is the degenerate case of the
    articulated branch rather than a different shape.
    """
    raise NotImplementedError("rigid branch")
