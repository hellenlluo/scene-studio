from app.pipeline.base import PipelineContext
from app.schemas import Certificate, RepairResult, SceneGraph

__all__ = ["RepairResult", "run"]


def run(ctx: PipelineContext, graph: SceneGraph, certificate: Certificate) -> RepairResult:
    """Compute the minimal correction that certifies a failing scene.

    Repair is underdetermined, and that is the difficulty rather than a detail.
    Given an object that penetrates its neighbour, the fix could be moving either
    body, rescaling either, or upgrading a collision proxy that was too coarse to
    sit flush. Cheapest and least destructive first: snap to the inferred support
    plane, push apart residual penetrations, upgrade the proxy tier, and rescale
    only where the object's scale residual was already large.

    Rescaling is the one to be reluctant about: it is the variable the whole
    system exists to estimate, so moving it to satisfy a contact quietly discards
    the measurement. Report the magnitude either way.

    Re-certify after applying and keep an action only if that object's result
    actually improved; set RepairAction.improved and discard the rest rather than
    reporting a repair that made things worse.
    """
    raise NotImplementedError("repair")
