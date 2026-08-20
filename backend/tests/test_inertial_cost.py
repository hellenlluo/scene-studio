"""The inertial and cost certification axes.

Both are short, and both catch things that nothing else does: MuJoCo will happily
simulate an impossible inertia tensor, and a scene that certifies on every other
axis but steps too slowly to train in is not simulation-ready either.
"""

import pytest

from app.certify import cost, inertial
from app.config import get_settings
from app.schemas import InertialProperties, ProxyTier
from tests.fixtures import scenes

REL_TOL = 0.01


@pytest.fixture
def settings():
    return get_settings()


def _by_object(checks):
    return {c.object_id: c for c in checks}


def _replace_mug_inertia(graph, **changes):
    mug = graph.get("mug").root_part
    mug.inertial = mug.inertial.model_copy(update=changes)
    return graph


# --- inertial: the happy path -------------------------------------------------


def test_sound_mass_properties_pass():
    checks = inertial.run(scenes.kitchen(), REL_TOL)
    assert len(checks) == 3
    assert all(c.passed for c in checks)


# --- inertial: each check catches its own failure ------------------------------


def test_mass_that_disagrees_with_density_times_volume_fails():
    graph = _replace_mug_inertia(scenes.kitchen(), mass_kg=99.0)
    check = _by_object(inertial.run(graph, REL_TOL))["mug"]
    assert not check.mass_density_volume_consistent
    assert not check.passed
    # The other two are independent and should still hold.
    assert check.positive_definite
    assert check.triangle_inequality


def test_a_non_watertight_mesh_fails_consistency():
    """trimesh returns a volume for an open mesh without complaining, and that
    volume is meaningless — so the mass derived from it is too, however plausible
    the number looks. Every actuation result downstream depends on that mass."""
    graph = _replace_mug_inertia(scenes.kitchen(), watertight=False)
    assert not _by_object(inertial.run(graph, REL_TOL))["mug"].mass_density_volume_consistent


def test_zero_mass_fails():
    """A massless body is a hard model error in MuJoCo."""
    graph = _replace_mug_inertia(scenes.kitchen(), mass_kg=0.0)
    assert not _by_object(inertial.run(graph, REL_TOL))["mug"].passed


def test_a_non_positive_definite_inertia_fails():
    graph = _replace_mug_inertia(scenes.kitchen(), inertia_diag=(0.0, 1.0, 1.0))
    check = _by_object(inertial.run(graph, REL_TOL))["mug"]
    assert not check.positive_definite
    assert not check.passed


def test_inertia_violating_the_triangle_inequality_fails():
    """No rigid body has principal moments where one exceeds the sum of the other
    two — it does not correspond to any distribution of mass."""
    graph = _replace_mug_inertia(scenes.kitchen(), inertia_diag=(1.0, 1.0, 5.0))
    check = _by_object(inertial.run(graph, REL_TOL))["mug"]
    assert not check.triangle_inequality
    assert not check.passed


def test_tolerance_is_relative_not_absolute():
    """1 g means something different on a mug than on a wardrobe."""
    graph = scenes.kitchen()
    mug = graph.get("mug").root_part
    nudged = mug.inertial.mass_kg * 1.005
    mug.inertial = mug.inertial.model_copy(update={"mass_kg": nudged})
    assert _by_object(inertial.run(graph, 0.01))["mug"].mass_density_volume_consistent
    assert not _by_object(inertial.run(graph, 0.001))["mug"].mass_density_volume_consistent


# --- inertial: what is not checkable ------------------------------------------


def test_objects_without_inertial_properties_are_skipped_not_failed():
    """That is an object the inertia stage has not reached. Absent is not
    inconsistent, and reporting it either way would be a lie — so it produces no
    check and the axis ends up NOT_APPLICABLE."""
    graph = scenes.kitchen()
    for obj in graph.objects:
        for part in obj.parts:
            part.inertial = None
    assert inertial.run(graph, REL_TOL) == []


def test_one_object_missing_inertia_does_not_hide_the_others():
    graph = scenes.kitchen()
    graph.get("mug").root_part.inertial = None
    checks = inertial.run(graph, REL_TOL)
    assert {c.object_id for c in checks} == {"table", "cabinet"}


# --- cost ---------------------------------------------------------------------


def test_a_small_scene_is_well_inside_budget(settings):
    check = cost.run(scenes.kitchen(), settings.step_time_budget_ms)
    assert check.passed
    assert 0.0 < check.mean_step_time_ms < settings.step_time_budget_ms
    assert check.budget_ms == settings.step_time_budget_ms


def test_the_budget_is_actually_enforced():
    check = cost.run(scenes.kitchen(), budget_ms=1e-9)
    assert not check.passed


def test_the_reported_tier_is_the_most_expensive_one_present():
    """The costliest proxy is what drives step time, so that is what the
    certificate reports for the scene."""
    graph = scenes.kitchen()
    assert cost.scene_tier(graph) is ProxyTier.OBB

    graph.get("mug").root_part.proxy_tier = ProxyTier.DECOMPOSED
    assert cost.scene_tier(graph) is ProxyTier.DECOMPOSED

    graph.get("table").root_part.proxy_tier = ProxyTier.CONVEX_HULL
    assert cost.scene_tier(graph) is ProxyTier.DECOMPOSED


def test_timing_is_reproducible(settings):
    """Two runs of the same scene land in the same ballpark. Loose tolerance
    because this is wall-clock on a shared machine, not a benchmark."""
    first = cost.run(scenes.kitchen(), settings.step_time_budget_ms)
    second = cost.run(scenes.kitchen(), settings.step_time_budget_ms)
    assert second.mean_step_time_ms == pytest.approx(first.mean_step_time_ms, rel=1.5)
    assert first.mean_step_time_ms > 0.0


def test_inertial_properties_helpers_agree_with_the_axis():
    """The axis delegates to these, so they are contract rather than convenience."""
    good = InertialProperties(
        density_kg_m3=600.0, volume_m3=0.1, mass_kg=60.0, inertia_diag=(1.0, 1.0, 1.0)
    )
    assert good.is_positive_definite and good.satisfies_triangle_inequality
