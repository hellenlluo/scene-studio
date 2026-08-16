from app.pipeline.base import PipelineContext
from app.schemas import GeometryResult, ScaleResult, SegmentResult

__all__ = ["GeometryResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult, scale: ScaleResult) -> GeometryResult:
    """Per-object geometry, with a guaranteed oriented-bounding-box fallback.

    Build the OBB path first and completely: it needs only the back-projected
    point cloud and the fitted dimensions, it runs without a GPU, and it makes
    every downstream stage exercisable end to end.

    Mesh reconstruction (TRELLIS / Hunyuan3D-2 / InstantMesh over the inpainted
    crop) then upgrades proxy_tier per object. It must stay optional — when it
    fails or is unavailable the object keeps its OBB and the scene stays valid.
    """
    raise NotImplementedError("geometry")
