from app.pipeline.base import PipelineContext
from app.schemas import GeometryResult

__all__ = ["GeometryResult", "run"]


def run(ctx: PipelineContext, geometry: GeometryResult) -> GeometryResult:
    """Collision proxies and inertial properties.

    Convex decomposition via CoACD (or V-HACD) over each visual mesh, preceded by
    watertightness repair — trimesh will happily hand you a non-watertight mesh
    whose volume, and therefore whose mass, is meaningless.

    Assign density by semantic category rather than a global constant, then
    compute mass and the inertia diagonal from the repaired volume. MuJoCo's
    behaviour depends on these more than on mesh fidelity.
    """
    raise NotImplementedError("collision")
