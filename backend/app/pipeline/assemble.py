from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, GeometryResult, ScaleResult, SceneLayout

__all__ = ["SceneLayout", "run"]


def run(
    ctx: PipelineContext,
    geometry: GeometryResult,
    depth: DepthResult,
    scale: ScaleResult,
) -> SceneLayout:
    """Place objects: floor fit, gravity alignment, support inference, de-overlap.

    Fit the floor plane from the back-projected cloud (RANSAC) and rotate the
    scene so that plane is world-down — everything after this assumes gravity
    points along -Z.

    Single-view depth only ever gives the front shell, so an object's extent away
    from the camera is inferred, not observed. Resolve the resulting
    interpenetrations here rather than leaving them for the physics engine to
    discover explosively at t=0.
    """
    raise NotImplementedError("assemble")
