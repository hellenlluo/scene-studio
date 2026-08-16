from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, LabelResult, ScaleResult, SegmentResult

__all__ = ["ScaleResult", "run"]


def run(
    ctx: PipelineContext,
    segments: SegmentResult,
    depth: DepthResult,
    labels: LabelResult,
) -> ScaleResult:
    """Reconcile depth-derived extents against VLM class priors into one scale.

    The core of the project. Back-project each mask through the depth map and
    intrinsics to get an up-to-scale extent per object, then solve for the single
    scene scale minimising confidence-weighted disagreement with the priors —
    robustly, since one badly-segmented object should not move the global fit.

    Keep per-object residuals: they are the honest signal for which objects the
    reconstruction is least sure about, and they drive what the editor surfaces
    for correction.
    """
    raise NotImplementedError("scale")
