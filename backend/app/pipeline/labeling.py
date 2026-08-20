from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, SegmentResult

__all__ = ["LabelResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult) -> LabelResult:
    """One VLM pass: category, size prior with per-axis sigma, support, routing.

    Batch every crop into a single call rather than one call per object, and use
    structured output against LabelResult so validation happens at the API layer
    and the model retries on mismatch instead of us parsing JSON out of prose.

    Two things the prompt has to get right:

    Ask for per-axis sigma, not a scalar confidence. E_prior weights by
    1/sigma^2, so "dishwasher: 60 cm +/- 2 cm" has to be able to outweigh
    "chair: 50 cm +/- 20 cm". Without it every prior gets equal weight and the
    fusion against depth is meaningless.

    State the routing asymmetry explicitly and tell the model to choose
    articulated when unsure. A misrouted cabinet loses its articulation
    permanently; a misrouted armchair costs one wasted model call.
    """
    raise NotImplementedError("label")
