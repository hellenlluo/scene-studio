from app.pipeline.base import PipelineContext
from app.schemas import SceneGraph

__all__ = ["SceneGraph", "recompute", "run"]


def run(ctx: PipelineContext, graph: SceneGraph) -> SceneGraph:
    """Assign collision proxies and the inertial and dynamic priors.

    Convex decomposition via CoACD over each visual mesh, preceded by
    watertightness repair — trimesh will hand you a non-watertight mesh and a
    meaningless volume without complaining, and mass computed from that volume is
    meaningless too. Record InertialProperties.watertight either way.

    Density comes from the semantic category rather than a global constant, and
    joint damping and friction likewise. MuJoCo's behaviour under actuation
    depends on these more than on mesh fidelity.

    This runs before solve, not after certify: the solver's physics block steps
    MuJoCo, and MuJoCo cannot settle a body with no mass.
    """
    raise NotImplementedError("inertia")


def recompute(graph: SceneGraph) -> SceneGraph:
    """Refresh mass and inertia from density and the current scale.

    Volume goes as scale^3, so every mass in the scene is stale the moment the
    solver moves a scale. Pure function, called inside the solve loop; the
    density prior assigned in run() is what stays fixed.
    """
    raise NotImplementedError("inertia recompute")
