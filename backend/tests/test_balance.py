"""Static contact-force feasibility, independent of dynamic settling."""

import pytest

from app.certify import balance
from app.pipeline import support
from app.pipeline.base import PipelineContext
from tests.fixtures import scenes


@pytest.fixture
def ctx(tmp_path):
    image = tmp_path / "room.png"
    image.write_bytes(b"balance test")
    return PipelineContext.create("balance-test", image)


def _run(ctx, graph):
    supports = support.build(graph, ctx.settings)
    return balance.run(graph, supports, ctx.settings.max_support_gap_m)


def test_sound_scene_has_feasible_contact_forces(ctx):
    checks = {check.object_id: check for check in _run(ctx, scenes.kitchen())}

    assert all(check.passed for check in checks.values())
    assert checks["mug"].contact_force_feasible
    assert checks["mug"].contact_count >= 4
    # Exact low collision vertices are retained alongside raster cell centres, so
    # the certificate sees the authored box's full 45 mm support half-width.
    assert checks["mug"].support_margin_m == pytest.approx(0.045)
    assert len(checks["mug"].contact_points_m) == checks["mug"].contact_count
    assert len(checks["mug"].support_polygon_xy_m) == 4
    assert checks["mug"].evidence_source == "collision_proxy"
    assert checks["mug"].centre_of_mass_m is not None


def test_floating_object_has_no_support_wrench(ctx):
    checks = {check.object_id: check for check in _run(ctx, scenes.mug_floating_above_table())}

    assert not checks["mug"].passed
    assert not checks["mug"].contact_force_feasible
    assert checks["mug"].contact_count == 0


def test_com_outside_contact_patch_is_infeasible(ctx):
    graph = scenes.kitchen()
    graph.get("mug").position_m = (0.61, 0.0, 0.80)
    checks = {check.object_id: check for check in _run(ctx, graph)}

    assert not checks["mug"].passed
    assert not checks["mug"].contact_force_feasible
    assert checks["mug"].support_margin_m == pytest.approx(-0.015)
    assert checks["mug"].failure_kind == "partial_support"
    assert checks["mug"].full_base_force_feasible
    assert checks["mug"].full_base_margin_m == pytest.approx(0.045)
    assert 0.0 < checks["mug"].support_coverage_ratio < 1.0


def test_com_outside_complete_base_is_intrinsic_not_partial(ctx):
    graph = scenes.kitchen()
    mug = graph.get("mug")
    inertial = mug.root_part.inertial
    assert inertial is not None
    mug.root_part.inertial = inertial.model_copy(update={"com_m": (0.07, 0.0, 0.0)})

    checks = {check.object_id: check for check in _run(ctx, graph)}
    check = checks["mug"]

    assert not check.passed
    assert check.failure_kind == "intrinsic_imbalance"
    assert not check.full_base_force_feasible
    assert check.full_base_margin_m == pytest.approx(-0.025)
    # The cabinet/table is not clipping the base: all potential base samples are
    # supported. Moving the rigid object sideways therefore cannot change the result.
    assert check.support_coverage_ratio == pytest.approx(1.0)


def test_mass_weighted_com_avoids_bounding_box_false_failure(ctx):
    checks = {check.object_id: check for check in _run(ctx, scenes.mug_overhanging_its_support())}

    assert checks["mug"].passed
    assert checks["mug"].support_margin_m > 0.0
