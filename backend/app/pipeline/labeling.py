from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, SegmentResult

__all__ = ["LabelResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult) -> LabelResult:
    """One VLM pass per object: category, real-world size prior, articulation flag.

    Batch the crops into a single call rather than one call per object. The size
    prior is what makes the scale stage work, so ask for dimensions in metres
    with an explicit confidence, and treat a low-confidence prior as absent
    rather than letting it drag the global fit.
    """
    raise NotImplementedError("label")
