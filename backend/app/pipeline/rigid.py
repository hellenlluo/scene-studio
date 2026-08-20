from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, ReconstructionResult, SegmentResult

__all__ = ["ReconstructionResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> ReconstructionResult:
    """Stage 4: per-object mesh reconstruction, via SAM 3D Objects on fal.

    Takes stage 1's masks directly as `mask_urls`, and stage 2's metric point map
    as `pointmap_url` — feeding the same depth the solver scores against means the
    mesh comes back consistent with it rather than disagreeing.

    Two things to handle:

    * **The returned scale is anisotropic and not metric.** Metadata carries a
      rotation, a translation and per-axis scale factors, camera-relative.
      SceneObject.scale is one float by design, so bake the anisotropy into the
      geometry and keep only the isotropic residual in
      AssetFrame.normalization_scale. Never use the number as-is however
      convenient: scale that is inherited from a generative model and never
      reconciled is the exact failure this project exists to fix.
    * **Fire the per-object calls concurrently** through the queue API. Serial, a
      large scene takes minutes; concurrent it takes seconds, and end-to-end
      latency is a reported result.

    An OBB proxy from the backprojected cloud is the fallback when reconstruction
    fails, and it is worth building first: it needs no GPU and it makes every
    downstream stage exercisable end to end.
    """
    raise NotImplementedError("reconstruct")
