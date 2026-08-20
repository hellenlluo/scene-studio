from app.schemas import CostCheck, SceneGraph

__all__ = ["run"]


def run(graph: SceneGraph, budget_ms: float) -> CostCheck:
    """Is the scene fast enough to actually simulate?

    Time MuJoCo steps at the scene's collision-proxy tier. A convex
    decomposition that certifies kinematically but steps too slowly to train in
    is not simulation-ready, which is why this is an axis and not a footnote —
    and the tier-versus-step-time trade is a reported result.
    """
    raise NotImplementedError("certify cost")
