from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, ReconstructionResult, SegmentResult

__all__ = ["ReconstructionResult", "run"]


def run(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> ReconstructionResult:
    """Branch 4b: part meshes plus joint hypotheses, via URDF-Anything+.

    Single RGB crop in, URDF plus OBJ part meshes out, generated autoregressively.
    Parse the result with yourdfpy into parts and JointHypothesis candidates, and
    keep the full ranked candidate set rather than only the argmax — the viewer's
    joint-drag refits within it, and the top-1/top-2 margin is v2's joint
    confidence signal.

    Chosen over per-part SAM 3D for a reason beyond joint prediction: this branch
    *generates* part geometry from a learned prior instead of reconstructing only
    what the photo shows. A model that has seen thousands of cabinets can produce a
    hollow carcass; SAM 3D, shown one image of a closed cabinet, cannot — the
    interior is unobservable, not merely unobserved. If that holds in practice it
    removes the need for a synthetic container prior, so it is worth checking on
    the first real cabinet rather than assuming either way.

    Three things this branch has to handle that 4a does not:

    * **It is self-hosted.** torch-cluster and CUDA mean it does not run on an
      M-series Mac, so this is a client to a GPU endpoint (Modal, RunPod, or a fal
      custom deploy) rather than a priced per-call API. Slow and expensive enough
      that the stage cache is doing real work here.
    * **It expects canonically oriented input** — objects facing +z, with an
      interactive rotation step in the reference implementation. A pipeline cannot
      stop for that, so either stage 3 predicts a front-face and the crop is
      rotated before upload, or v1 accepts a manual step and says so.
    * **Output is normalised to a unit box**, so scale, joint origins *and*
      prismatic limits all arrive in normalised units. Record the factor in
      AssetFrame.normalization_scale and let stage 6 recover the real one; a
      prismatic limit that scales with its parent is the coupling this project
      exists to exploit, not an inconvenience.

    Failure is expected and cheap: set ObjectAssets.failed and let the orchestrator
    fall the object back to 4a, where it becomes one rigid mesh with no joints. An
    object routed here that returns zero joints degrades to exactly the rigid
    result, which is why the router is biased toward this branch in the first
    place.
    """
    raise NotImplementedError("articulated branch")
