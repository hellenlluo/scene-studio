from app.pipeline.base import PipelineContext
from app.schemas import SegmentResult

__all__ = ["SegmentResult", "run"]


def run(ctx: PipelineContext) -> SegmentResult:
    """Instance masks over the input photo, via SAM 3 on fal.

    Concept-prompted rather than class-agnostic: the endpoint's default prompt is
    literally "car", so it needs a noun list. That argues for a cheap VLM
    inventory pass before this stage, with the full labelling pass after.

    Write each mask to ctx.workdir() as a single-channel PNG and reference it by
    path — masks are too large to carry inline through the stage contracts. fal
    needs publicly fetchable URLs, so the photo and every mask get uploaded.
    """
    raise NotImplementedError("segment")
