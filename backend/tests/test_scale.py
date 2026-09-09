"""The scale certification axis.

The cheapest axis: pure geometry over the scene graph, no MuJoCo and no files. It
reads no depth map on purpose — depth is an input to the solver, so scoring the
result against it would let an overfitted solve certify itself.
"""

import math

import pytest

from app.certify import scale
from app.config import get_settings
from tests.fixtures import scenes

MAX_SIGMA = 3.0
MAX_GAP = 0.005
# Burial is graded against the penetration tolerance, not the gap one — see
# `scale.gap_ok`.
MAX_PEN = 0.002


def _by_object(checks):
    return {c.object_id: c for c in checks}


def _run(graph, max_sigma=MAX_SIGMA, max_gap=MAX_GAP, max_pen=MAX_PEN):
    return _by_object(scale.run(graph, max_sigma, max_gap, max_pen))


# --- the happy path -----------------------------------------------------------


def test_a_sound_scene_passes():
    checks = scale.run(scenes.kitchen(), MAX_SIGMA, MAX_GAP, MAX_PEN)
    assert len(checks) == 3
    assert all(c.passed for c in checks)


def test_resting_objects_have_a_closed_support_gap():
    for check in scale.run(scenes.kitchen(), MAX_SIGMA, MAX_GAP, MAX_PEN):
        assert check.support_gap_m == pytest.approx(0.0, abs=1e-9)
        assert check.base_inside_parent


def test_thresholds_come_from_settings():
    settings = get_settings()
    assert settings.max_prior_deviation_sigma > 0
    assert settings.max_support_gap_m > 0


# --- the support gap ----------------------------------------------------------


def test_a_floating_object_has_a_positive_gap():
    check = _run(scenes.mug_floating_above_table())["mug"]
    assert check.support_gap_m == pytest.approx(0.25, abs=1e-6)
    assert not check.passed


def test_a_sunk_object_has_a_negative_gap():
    """Sign carries the diagnosis: floating and buried are different errors and a
    magnitude alone would not say which."""
    check = _run(scenes.mug_sunk_into_table())["mug"]
    assert check.support_gap_m == pytest.approx(-0.05, abs=1e-6)
    assert not check.passed


def test_burial_is_graded_against_the_penetration_tolerance_not_the_gap_one():
    """Floating and buried are different failures measured on the same number, and
    the certificate publishes a different bound for each: clearance against
    `max_support_gap_m`, because a mesh contact is never exactly flush, and burial
    against `max_penetration_m`, because burial *is* penetration.

    One two-sided `abs(gap) <= max_support_gap_m` hands out 5 mm of overlap that the
    next axis fails. Measured on `room2.png`: a bookshelf 4.99 mm into the floor and
    a rug 2.11 mm into it both passed this axis and failed `certify.stability`, and
    `repair` could reach neither — it scored the gap the same way, and
    `_resolve_penetration` cannot push against a floor that is not an object.
    """
    graph = scenes.kitchen()
    mug = graph.get("mug")
    x, y, z = mug.position_m

    mug.position_m = (x, y, z - 0.004)  # inside the 5 mm gap bound, past the 2 mm one
    buried = _run(graph)["mug"]
    assert buried.support_gap_m == pytest.approx(-0.004, abs=1e-9)
    assert not buried.passed

    # The same magnitude of *clearance* is fine, which is the whole asymmetry.
    mug.position_m = (x, y, z + 0.004)
    assert _run(graph)["mug"].passed


def test_the_gap_failure_is_localised_to_the_offending_object():
    """Unlike the stability axis, whose penetration measure is symmetric and fails
    both bodies, this names only the one that is in the wrong place."""
    checks = _run(scenes.mug_sunk_into_table())
    assert not checks["mug"].passed
    assert checks["table"].passed
    assert checks["cabinet"].passed


def test_objects_on_the_floor_are_measured_against_floor_height():
    graph = scenes.kitchen()
    graph.floor_height_m = 0.2
    check = _run(graph)["table"]
    assert check.support_gap_m == pytest.approx(-0.2, abs=1e-6)
    assert not check.passed


# --- prior deviation ----------------------------------------------------------


def test_an_object_far_from_its_class_prior_fails():
    graph = scenes.kitchen()
    graph.get("mug").scale = 3.0  # a 27 cm mug
    check = _run(graph)["mug"]
    assert max(abs(d) for d in check.deviation_sigma) > MAX_SIGMA
    assert not check.passed


def test_deviation_is_reported_per_axis():
    """A scalar would hide which dimension is wrong, and that is exactly what tells
    you whether the scale or the reconstruction is at fault."""
    graph = scenes.kitchen()
    cabinet = graph.get("cabinet")
    check = _run(graph)["cabinet"]
    assert len(check.deviation_sigma) == 3
    # The fixture cabinet is 0.92 tall against a 0.90 prior with sigma 0.05.
    assert check.deviation_sigma[2] == pytest.approx(0.4, abs=1e-6)
    assert check.deviation_sigma[0] == pytest.approx(0.0, abs=1e-6)
    assert cabinet.dims_m[2] == pytest.approx(0.92)


def test_a_zero_sigma_prior_is_a_hard_constraint():
    """A scale anchor enters as E_prior with sigma -> 0. Any mismatch has to read
    as a large deviation rather than dividing by zero."""
    graph = scenes.kitchen()
    mug = graph.get("mug")
    mug.label.prior = mug.label.prior.model_copy(update={"sigma_m": (0.0, 0.0, 0.0)})
    mug.scale = 1.01

    check = _run(graph)["mug"]
    assert all(math.isfinite(d) for d in check.deviation_sigma)
    assert max(abs(d) for d in check.deviation_sigma) > MAX_SIGMA
    assert not check.passed


def test_a_zero_sigma_prior_that_is_matched_exactly_still_passes():
    graph = scenes.kitchen()
    mug = graph.get("mug")
    mug.label.prior = mug.label.prior.model_copy(update={"sigma_m": (0.0, 0.0, 0.0)})
    assert _run(graph)["mug"].passed


# --- footprint containment ----------------------------------------------------


def test_an_object_pushed_off_its_support_fails_containment():
    graph = scenes.kitchen()
    mug = graph.get("mug")
    x, y, z = mug.position_m
    mug.position_m = (x + 1.0, y, z)  # well past the tabletop edge

    check = _run(graph)["mug"]
    assert not check.base_inside_parent
    assert not check.passed


def test_containment_is_tested_on_the_centre_of_mass_not_the_bounding_box():
    """The toppling condition is about weight, and a box centre is not where the
    weight is — see `scenes.mug_overhanging_its_support`. Measured on both rooms the
    box gets it wrong in both directions: `room.png`'s lamp has its box centre
    outside its contact polygon and its mass inside, and settles 0.1 mm; `room2.png`'s
    armchair has its box centre inside and its mass outside, and settles 23.7 mm.
    """
    check = _run(scenes.mug_overhanging_its_support())["mug"]
    assert check.base_inside_parent
    assert check.support_gap_m == pytest.approx(0.0, abs=1e-9)
    assert check.passed


def test_floor_supported_objects_are_trivially_contained():
    """The floor is unbounded, so only the gap is informative there."""
    checks = _run(scenes.kitchen())
    assert checks["table"].base_inside_parent
    assert checks["cabinet"].base_inside_parent


def test_an_unknown_support_parent_degrades_to_the_floor():
    """A reconcile bug, but reporting it as a floor contact keeps the object's
    height visible instead of skipping it silently."""
    graph = scenes.kitchen()
    graph.get("mug").supported_by = "no-such-object"
    check = _run(graph)["mug"]
    assert check.base_inside_parent
    # Its base is at tabletop height, so against the floor it reads as floating.
    assert check.support_gap_m == pytest.approx(0.75, abs=1e-6)


# --- rotation -----------------------------------------------------------------


def test_bounds_account_for_orientation():
    """All eight corners are transformed rather than the extent being scaled
    directly: a rotated box has a different axis-aligned footprint from its own
    dimensions, and a gap computed from the unrotated extent would report a tilted
    object as floating."""
    graph = scenes.kitchen()
    mug = graph.get("mug")
    # 90 degrees about x: the 0.10 z-extent becomes the 0.09 y-extent, so the base
    # rises by half the difference and the object lifts off its support.
    mug.orientation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)

    check = _run(graph)["mug"]
    assert check.support_gap_m == pytest.approx(0.005, abs=1e-6)


def test_a_rotation_that_changes_nothing_changes_nothing():
    graph = scenes.kitchen()
    graph.get("mug").orientation = (1.0, 0.0, 0.0, 0.0)
    assert _run(graph)["mug"].support_gap_m == pytest.approx(0.0, abs=1e-9)
