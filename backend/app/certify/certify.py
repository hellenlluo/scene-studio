"""Assemble the multi-axis certificate.

Each axis reports one of four things, and the difference between the last two is
the point of the whole design:

* **PASS** — checked, and it holds.
* **FAIL** — checked, and it does not.
* **NOT_APPLICABLE** — there was nothing of that kind to check.
* **NOT_RUN** — no validator reached this axis. `Certificate.passed` treats that as
  disqualifying, so a scene cannot certify on the strength of whichever checks
  happen to be implemented.

All axes have validators, so a sound scene now certifies. Anything that
reports NOT_RUN from here means a validator raised or was skipped, not that the
check does not exist.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from app.pipeline import support
from app.pipeline.base import PipelineContext
from app.schemas import AxisStatus, Certificate, SceneGraph

from . import balance, cost, inertial, scale, stability

# Re-exported so callers can reach a single axis directly.
__all__ = ["Certificate", "balance", "cost", "inertial", "run", "scale", "stability"]

log = logging.getLogger(__name__)
CERTIFICATE_VERSION = "1.3"


def _scene_graph_sha256(graph: SceneGraph) -> str:
    """Hash the exact graph values passed to the validators."""
    payload = json.dumps(
        graph.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _collision_geometry_sha256(graph: SceneGraph) -> str:
    """Hash collision bytes under stable object/part/index identifiers.

    Absolute storage paths are deliberately excluded: moving an unchanged scene to
    another machine must not change the geometry identity. Missing files are still
    represented, so a certificate cannot share a hash with one whose assets existed.
    """
    digest = hashlib.sha256()
    for obj in sorted(graph.objects, key=lambda item: item.object_id):
        for part in sorted(obj.parts, key=lambda item: item.part_id):
            for index, raw_path in enumerate(part.collision_mesh_paths):
                identity = f"{obj.object_id}\0{part.part_id}\0{index}\0".encode()
                digest.update(identity)
                path = Path(raw_path)
                try:
                    digest.update(path.read_bytes())
                except OSError as exc:
                    digest.update(f"<missing:{type(exc).__name__}>".encode())
    return digest.hexdigest()


def _simulator_version() -> str:
    try:
        return version("mujoco")
    except PackageNotFoundError:
        return "unknown"


def _record_error(errors: dict[str, str], axis: str, exc: Exception) -> None:
    errors[axis] = f"{type(exc).__name__}: {exc}"
    log.exception("%s certification did not run", axis)


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
    errors: dict[str, str] = {}
    supports = None

    # The same measured surfaces stage 6 solved against, so the axis grades the
    # contact the solver was aiming at rather than a bounding-box approximation of it.
    try:
        supports = support.build(graph, settings)
        scale_checks = scale.run(
            graph,
            settings.max_prior_deviation_sigma,
            settings.max_support_gap_m,
            # Burial is penetration, so the scale axis grades it against the same
            # tolerance the stability axis does. See `scale.gap_ok`.
            settings.max_penetration_m,
            supports,
        )
        scale_status = _status(scale_checks)
    except Exception as exc:
        _record_error(errors, "scale", exc)
        scale_checks = []
        scale_status = AxisStatus.NOT_RUN

    # Static equilibrium is deliberately separate from dynamic settling.  The
    # former asks whether the photographed pose *can* be held by its contacts; the
    # latter asks what MuJoCo does after release.  Keeping both makes their
    # disagreement a diagnostic instead of hiding one inside the other.
    try:
        if supports is None:
            supports = support.build(graph, settings)
        balance_checks = balance.run(graph, supports, settings.max_support_gap_m)
        balance_status = _status(balance_checks)
    except Exception as exc:
        _record_error(errors, "balance", exc)
        balance_checks = []
        balance_status = AxisStatus.NOT_RUN

    try:
        stability_checks = stability.run(graph, settings)
        stability_status = _status(stability_checks)
    except Exception as exc:
        _record_error(errors, "stability", exc)
        stability_checks = []
        stability_status = AxisStatus.NOT_RUN

    try:
        inertial_checks = inertial.run(graph, settings.max_inertia_rel_error)
        inertial_status = _status(inertial_checks)
    except Exception as exc:
        _record_error(errors, "inertial", exc)
        inertial_checks = []
        inertial_status = AxisStatus.NOT_RUN

    try:
        cost_check = cost.run(graph, settings.step_time_budget_ms)
        cost_status = AxisStatus.PASS if cost_check.passed else AxisStatus.FAIL
    except Exception as exc:
        _record_error(errors, "cost", exc)
        cost_check = None
        cost_status = AxisStatus.NOT_RUN

    return Certificate(
        certificate_version=CERTIFICATE_VERSION,
        generated_at=datetime.now(UTC),
        scene_graph_sha256=_scene_graph_sha256(graph),
        collision_geometry_sha256=_collision_geometry_sha256(graph),
        simulator_name="MuJoCo",
        simulator_version=_simulator_version(),
        axis_errors=errors,
        scale_status=scale_status,
        balance_status=balance_status,
        stability_status=stability_status,
        inertial_status=inertial_status,
        cost_status=cost_status,
        scale=scale_checks,
        balance=balance_checks,
        stability=stability_checks,
        inertial=inertial_checks,
        cost=cost_check,
        # Recorded on the certificate, not just read from settings, so a stored
        # result stays interpretable after the threshold moves.
        penetration_tolerance_m=settings.max_penetration_m,
        com_displacement_tolerance_m=settings.max_com_displacement_m,
        support_gap_tolerance_m=settings.max_support_gap_m,
        prior_deviation_tolerance_sigma=settings.max_prior_deviation_sigma,
        orientation_drift_tolerance_deg=settings.max_orientation_drift_deg,
    )
