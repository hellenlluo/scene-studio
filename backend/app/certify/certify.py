"""Assemble the four-axis certificate.

Each axis reports one of four things, and the difference between the last two is
the point of the whole design:

* **PASS** — checked, and it holds.
* **FAIL** — checked, and it does not.
* **NOT_APPLICABLE** — there was nothing of that kind to check.
* **NOT_RUN** — no validator reached this axis. `Certificate.passed` treats that as
  disqualifying, so a scene cannot certify on the strength of whichever checks
  happen to be implemented.

All four axes have validators, so a sound scene now certifies. Anything that
reports NOT_RUN from here means a validator raised or was skipped, not that the
check does not exist.
"""

from app.pipeline.base import PipelineContext
from app.schemas import AxisStatus, Certificate, SceneGraph

from . import cost, inertial, scale, stability

# Re-exported so callers can reach a single axis directly.
__all__ = ["Certificate", "cost", "inertial", "run", "scale", "stability"]


def _status(checks: list) -> AxisStatus:
    """PASS, FAIL, or NOT_APPLICABLE when there was nothing to check.

    Note what this cannot return: NOT_RUN. Reaching here means the validator ran,
    so an empty list means the scene had nothing of that kind — not that the
    question went unasked.
    """
    if not checks:
        return AxisStatus.NOT_APPLICABLE
    return AxisStatus.PASS if all(c.passed for c in checks) else AxisStatus.FAIL


def run(ctx: PipelineContext, graph: SceneGraph) -> Certificate:
    settings = ctx.settings

    scale_checks = scale.run(graph, settings.max_prior_deviation_sigma, settings.max_support_gap_m)
    stability_checks = stability.run(graph, settings)
    inertial_checks = inertial.run(graph, settings.max_inertia_rel_error)
    cost_check = cost.run(graph, settings.step_time_budget_ms)

    return Certificate(
        scale_status=_status(scale_checks),
        stability_status=_status(stability_checks),
        inertial_status=_status(inertial_checks),
        cost_status=AxisStatus.PASS if cost_check.passed else AxisStatus.FAIL,
        scale=scale_checks,
        stability=stability_checks,
        inertial=inertial_checks,
        cost=cost_check,
        # Recorded on the certificate, not just read from settings, so a stored
        # result stays interpretable after the threshold moves.
        penetration_tolerance_m=settings.max_penetration_m,
    )
