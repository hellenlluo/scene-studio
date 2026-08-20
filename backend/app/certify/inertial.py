from app.schemas import InertialCheck, SceneGraph

__all__ = ["run"]


def run(graph: SceneGraph) -> list[InertialCheck]:
    """Are the mass properties physically possible?

    Three checks, all cheap and all catching real failures: mass equals density
    times volume for a watertight mesh; the inertia diagonal is positive
    definite; the principal moments satisfy the triangle inequality. The helpers
    on InertialProperties cover the last two.

    Cheap to check and easy to get wrong — a non-watertight mesh yields a
    nonsense volume, and mass computed from it propagates silently into every
    actuation result downstream.
    """
    raise NotImplementedError("certify inertial")
