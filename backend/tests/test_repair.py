"""Minimal correction of a failing scene.

The interesting tests here are not "does it fix things" but the policy questions:
which body moves, when rescaling is allowed, and what happens to a repair that
turns out not to help.
"""

import pytest

from app.certify import certify, repair
from app.pipeline.base import PipelineContext
from app.schemas import Provenance, RepairKind
from tests.fixtures import scenes


@pytest.fixture
def ctx(tmp_path):
    image = tmp_path / "room.png"
    image.write_bytes(b"only the bytes are hashed")
    return PipelineContext.create("repair-test", image)


def _repair(ctx, graph):
    return repair.run(ctx, graph, certify.run(ctx, graph))


def _kept(result):
    return [a for a in result.actions if a.improved]


# --- each strategy fixes its own failure --------------------------------------


def test_a_floating_object_is_snapped_down(ctx):
    graph = scenes.mug_floating_above_table()
    result = _repair(ctx, graph)

    assert result.certificate.passed
    action = _kept(result)[0]
    assert action.kind is RepairKind.SNAP_TO_SUPPORT
    assert action.target_id == "mug"
    assert action.delta_position_m[2] == pytest.approx(-0.25, abs=1e-6)
    assert action.magnitude == pytest.approx(0.25, abs=1e-6)


def test_a_buried_object_is_snapped_up(ctx):
    result = _repair(ctx, scenes.mug_sunk_into_table())
    assert result.certificate.passed
    action = _kept(result)[0]
    assert action.kind is RepairKind.SNAP_TO_SUPPORT
    assert action.delta_position_m[2] == pytest.approx(0.05, abs=1e-6)


def test_overlapping_bodies_are_pushed_apart_along_the_shallowest_axis(ctx):
    """The cabinet overlaps the table by 0.2 m in x and much more in y and z, so
    x is the cheapest way out."""
    result = _repair(ctx, scenes.cabinet_overlapping_table())
    assert result.certificate.passed

    action = _kept(result)[0]
    assert action.kind is RepairKind.RESOLVE_PENETRATION
    assert action.delta_position_m[0] == pytest.approx(0.202, abs=1e-3)
    assert action.delta_position_m[1] == 0.0
    assert action.delta_position_m[2] == 0.0


def test_a_sound_scene_is_left_alone(ctx):
    result = _repair(ctx, scenes.kitchen())
    assert result.actions == []
    assert result.certificate.passed


def test_repair_is_idempotent(ctx):
    """Repairing an already-repaired scene should find nothing left to do."""
    once = _repair(ctx, scenes.mug_floating_above_table())
    twice = repair.run(ctx, once.graph, once.certificate)
    assert twice.actions == []


# --- which body moves ---------------------------------------------------------


def test_an_object_that_supports_others_is_never_the_one_moved(ctx):
    """Moving a support silently drags its dependents, turning one repair into
    several. This holds even when the mass rule would say otherwise: here the
    table is made the lighter of the two, and the cabinet still moves."""
    graph = scenes.cabinet_overlapping_table()
    table = graph.get("table")
    cabinet = graph.get("cabinet")
    table.parts[0].inertial = table.parts[0].inertial.model_copy(update={"mass_kg": 1.0})
    cabinet.parts[0].inertial = cabinet.parts[0].inertial.model_copy(update={"mass_kg": 999.0})
    assert graph.get("mug").supported_by == "table"

    result = _repair(ctx, graph)
    movers = {a.target_id for a in _kept(result)}
    assert "table" not in movers
    assert movers == {"cabinet"}


def test_a_support_contact_is_not_treated_as_a_penetration(ctx):
    """A mug resting on a table is a contact you *want*. Snapping owns those; the
    push-apart strategy has to leave them alone or it would eject every object
    from every surface it sits on."""
    result = _repair(ctx, scenes.mug_sunk_into_table())
    assert all(a.kind is not RepairKind.RESOLVE_PENETRATION for a in result.actions)


# --- rescaling is gated -------------------------------------------------------


def test_rescaling_does_not_fire_when_the_scale_estimate_is_sound(ctx):
    """Scale is the quantity the whole system exists to estimate, so moving it to
    satisfy a contact would discard the measurement. It is only allowed where the
    estimate was already failing on its own terms."""
    for name in ("mug_floating_above_table", "mug_sunk_into_table", "cabinet_overlapping_table"):
        result = _repair(ctx, getattr(scenes, name)())
        assert all(a.kind is not RepairKind.RESCALE for a in result.actions), name


def test_rescaling_is_offered_when_the_deviation_is_already_out_of_tolerance(ctx):
    graph = scenes.kitchen()
    graph.get("mug").scale = 3.0  # a 27 cm mug, far outside its prior
    result = _repair(ctx, graph)

    rescales = [a for a in result.actions if a.kind is RepairKind.RESCALE]
    assert rescales
    assert rescales[0].target_id == "mug"
    assert rescales[0].delta_scale == pytest.approx(1 / 3, rel=0.05)


# --- bookkeeping --------------------------------------------------------------


def test_the_input_graph_is_never_mutated(ctx):
    """The caller has to be able to keep the original — a repair is a proposal
    until re-certification accepts it."""
    graph = scenes.mug_floating_above_table()
    before = graph.get("mug").position_m

    result = _repair(ctx, graph)

    assert graph.get("mug").position_m == before
    assert result.graph.get("mug").position_m != before


def test_repaired_values_are_marked_as_derived(ctx):
    """Not USER: a repair is the system's own correction and the next solve should
    be free to move it, unlike something a person pinned deliberately."""
    result = _repair(ctx, scenes.mug_floating_above_table())
    assert result.graph.get("mug").provenance["position_m"] is Provenance.DERIVED


def test_violation_is_zero_for_a_passing_scene_and_positive_otherwise(ctx):
    settings = ctx.settings
    assert repair._violation(certify.run(ctx, scenes.kitchen()), settings) == 0.0
    assert repair._violation(certify.run(ctx, scenes.mug_sunk_into_table()), settings) > 0.0


def test_every_recorded_action_says_whether_it_helped(ctx):
    """Rejected repairs are reverted but still reported. A silent no-op tells you
    nothing, and knowing what was tried is what you need when a scene will not
    certify."""
    result = _repair(ctx, scenes.cabinet_overlapping_table())
    assert result.actions
    assert all(isinstance(a.improved, bool) for a in result.actions)
    assert all(a.magnitude > 0 for a in result.actions)


# --- the round budget ---------------------------------------------------------


def test_repair_converges_well_inside_the_budget(ctx):
    """The loop exits when a round accepts nothing, so the cap is rarely reached.
    These fixtures each need a single action."""
    for name in ("mug_floating_above_table", "mug_sunk_into_table", "cabinet_overlapping_table"):
        result = _repair(ctx, getattr(scenes, name)())
        assert result.converged, name
        assert result.rounds_used == 1, name


def test_a_sound_scene_still_costs_one_round(ctx):
    result = _repair(ctx, scenes.kitchen())
    assert result.converged
    assert result.rounds_used == 1


def test_exhausting_the_budget_is_reported_as_not_converged(ctx):
    """A truncated repair is not the same as a scene that cannot be repaired, and a
    caller that cannot tell them apart would report the first as the second."""
    settings = ctx.settings.model_copy(update={"max_repair_rounds": 1})
    ctx.settings = settings

    graph = scenes.kitchen()
    graph.get("mug").scale = 3.0
    x, y, z = graph.get("mug").position_m
    graph.get("mug").position_m = (x, y, z + 0.4)

    result = _repair(ctx, graph)
    assert result.rounds_used == 1
    # One round accepted something, so the loop wanted another and was denied.
    assert not result.converged
