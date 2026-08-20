from app.pipeline.base import PipelineContext
from app.schemas import Certificate, RepairResult, SceneGraph

__all__ = ["RepairResult", "run"]


def run(ctx: PipelineContext, graph: SceneGraph, certificate: Certificate) -> RepairResult:
    """Compute the minimal correction that certifies a failing scene.

    Repair is underdetermined, and that is the difficulty rather than a detail.
    Given a joint that interferes, the fix could be moving the origin, rotating
    the axis, trimming the limit or rescaling the part. Cheapest and least
    destructive first: snap to the inferred support plane, push apart residual
    penetrations, trim joint limits to the feasible sub-range the sweep already
    found, and rescale only where the object's scale residual was already large.
    Weld the joint only as a last resort — it is the one repair that cannot be
    undone by a later solve.

    Prefer trimming to rescaling when both would work, then report how much range
    the trim cost. Choosing badly silently discards range the object really has,
    and silence is the failure mode worth engineering against.

    Re-certify after applying and keep an action only if that object's result
    actually improved; set RepairAction.improved and discard the rest rather than
    reporting a repair that made things worse.
    """
    raise NotImplementedError("repair")
