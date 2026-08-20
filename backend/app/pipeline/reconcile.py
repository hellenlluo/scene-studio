from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, LabelResult, ReconstructionResult, SceneGraph

__all__ = ["SceneGraph", "run"]


def run(
    ctx: PipelineContext,
    rigid: ReconstructionResult,
    articulated: ReconstructionResult,
    depth: DepthResult,
    labels: LabelResult,
) -> SceneGraph:
    """Merge both branches into one scene graph. Unglamorous; nothing works without it.

    The branches disagree about frames and units. Mesh models and URDF models
    emit different canonical orientations, different scale conventions (most URDF
    models normalise to a unit box) and different part decompositions. For each
    asset: detect up-axis and front-face, record the transform in AssetFrame, and
    divide out the branch's internal normalisation so stage 6 optimises one scale
    variable per object instead of two incompatible ones.

    Then fit the floor plane from the backprojected cloud (RANSAC) and rotate the
    scene so that plane is world-down — everything downstream assumes gravity
    along -Z.

    Also the fallback site for joints: where 4b produced no hypotheses, derive
    geometric candidates from part extents, gaps and face normals, plus
    manufacturing-convention priors for the category. Hinges are occluded by the
    very door they attach, so this is missing information rather than noise; the
    honest response is a ranked candidate set the user can correct, not a
    confident guess.
    """
    raise NotImplementedError("reconcile")
