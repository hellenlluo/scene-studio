from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, LabelResult, ReconstructionResult, SceneGraph

__all__ = ["SceneGraph", "run"]


def run(
    ctx: PipelineContext,
    reconstruction: ReconstructionResult,
    depth: DepthResult,
    labels: LabelResult,
) -> SceneGraph:
    """Merge reconstruction output into one scene graph. Unglamorous; nothing works
    without it.

    Reconstruction models emit assets in their own canonical orientation and their
    own scale convention — most normalise to a unit box. Per asset: detect up-axis
    and front-face, record the transform in AssetFrame, and divide out the
    internal normalisation so stage 6 optimises one scale variable per object
    rather than two incompatible ones.

    Then fit the floor plane from the backprojected cloud (RANSAC) and rotate the
    scene so that plane is world-down — everything downstream assumes gravity
    along -Z.

    Single-view depth only ever gives the front shell, so an object's extent away
    from the camera is inferred rather than observed. Resolve the resulting
    interpenetrations here rather than leaving them for the physics engine to
    discover explosively at t=0.
    """
    raise NotImplementedError("reconcile")
