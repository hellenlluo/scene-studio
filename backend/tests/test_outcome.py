"""A completed computation is not necessarily a ready physical scene."""

from app.pipeline import solve
from app.schemas import SceneOutcome
from tests.fixtures.seeded import build_spec


def test_ready_requires_convergence_settling_and_certification():
    spec = build_spec("outcome-ready", "kitchen")
    assert spec.certificate.passed
    assert spec.outcome is SceneOutcome.READY
    assert spec.outcome_reasons == []

    spec.diagnostics.converged = False
    spec.refresh_outcome()

    assert spec.outcome is SceneOutcome.NEEDS_ATTENTION
    assert spec.outcome_reasons == ["solver_not_converged"]


def test_legacy_scene_without_outcome_is_classified_when_loaded():
    spec = build_spec("outcome-legacy", "mug_sunk_into_table")
    stored = spec.model_dump(mode="json")
    stored.pop("outcome")
    stored.pop("outcome_reasons")

    loaded = type(spec).model_validate(stored)

    assert loaded.outcome is SceneOutcome.NEEDS_ATTENTION
    assert "scene_not_settled" in loaded.outcome_reasons
    assert "certification_failed" in loaded.outcome_reasons


def test_final_diagnostics_describe_the_certified_graph():
    spec = build_spec("outcome-final", "mug_sunk_into_table")
    stale = spec.diagnostics.model_copy(update={"settled": True, "max_settle_drift_m": 0.0})

    final = solve.finalise_diagnostics(stale, spec.certificate)
    measured = max(check.com_displacement_m for check in spec.certificate.stability)

    assert final.max_settle_drift_m == measured
    assert not final.settled
