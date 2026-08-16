from app.pipeline.base import PipelineContext
from app.schemas import DepthResult

__all__ = ["DepthResult", "run"]


def run(ctx: PipelineContext) -> DepthResult:
    """Metric monocular depth plus camera intrinsics.

    TODO: Depth Anything V2 (metric variant) or UniDepth. UniDepth predicts
    intrinsics too, which avoids guessing focal length from EXIF; if using
    Depth Anything, fall back to an assumed ~60 degree horizontal FOV and set
    is_metric honestly so the scale stage knows what it is working with.
    """
    raise NotImplementedError("depth")
