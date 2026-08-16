from app.pipeline.base import PipelineContext
from app.schemas import SegmentResult

__all__ = ["SegmentResult", "run"]


def run(ctx: PipelineContext) -> SegmentResult:
    """Instance masks over the input photo.

    TODO: SAM 2 for class-agnostic masks, or Grounded-SAM if text-prompted
    proposals turn out to segment furniture more reliably. Write each mask to
    ctx.workdir() as a single-channel PNG and reference it by path — masks are
    too large to carry through the stage contracts inline.
    """
    raise NotImplementedError("segment")
