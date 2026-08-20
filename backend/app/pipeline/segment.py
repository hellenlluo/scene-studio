from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, SegmentResult

__all__ = ["SegmentResult", "run", "run_parts"]


def run(ctx: PipelineContext) -> SegmentResult:
    """Instance masks over the input photo.

    TODO: SAM 2 for class-agnostic masks, or Grounded-SAM if text-prompted
    proposals turn out to segment furniture more reliably. Write each mask to
    ctx.workdir() as a single-channel PNG and reference it by path — masks are
    too large to carry through the stage contracts inline.
    """
    raise NotImplementedError("segment")


def run_parts(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> SegmentResult:
    """Second pass: part masks inside every object stage 3 routed to articulated.

    Runs after labelling rather than with it, because which objects need part
    decomposition is exactly stage 3's routing decision. Returns the same
    SegmentResult with ObjectMask.parts populated.
    """
    raise NotImplementedError("segment parts")
