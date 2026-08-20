from app.pipeline.base import PipelineContext
from app.schemas import AxisStatus, Certificate, SceneGraph

from . import cost, inertial, kinematic, scale, stability

# Re-exported so callers reach one axis directly — the PartNet-Mobility audit
# wants certify.kinematic and nothing else.
__all__ = ["Certificate", "cost", "inertial", "kinematic", "run", "scale", "stability"]


def run(ctx: PipelineContext, graph: SceneGraph) -> Certificate:
    """Run all five axes and assemble the certificate.

    An axis with nothing applicable to check reports NOT_APPLICABLE rather than
    PASS. The distinction is the whole point: a scene of jointless objects has
    not passed the kinematic axis, it has not been tested on it.
    """
    raise NotImplementedError("certify")


def _status(checks: list, applicable: bool) -> AxisStatus:
    if not applicable:
        return AxisStatus.NOT_APPLICABLE
    return AxisStatus.PASS if all(c.passed for c in checks) else AxisStatus.FAIL
