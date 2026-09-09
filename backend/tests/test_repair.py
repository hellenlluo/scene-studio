"""Minimal correction of a failing scene.

The interesting tests here are not "does it fix things" but the policy questions:
which body moves, when rescaling is allowed, and what happens to a repair that
turns out not to help.
"""

import pytest

from app.certify import certify, repair
from app.config import get_settings
from app.pipeline.base import PipelineContext
from app.schemas import (
    Certificate,
    Provenance,
    RepairKind,
    ScaleCheck,
    StabilityCheck,
)
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


# --- the snap limit -----------------------------------------------------------


def test_a_correction_too_large_to_be_minimal_is_refused(ctx):
    """The motivating case is a wall-mounted picture misread as floor-supported.
    Snapping it would drop the painting onto the carpet — an actively harmful
    "repair" that also destroys the evidence of what went wrong. At that magnitude
    the support *relation* is the likelier error, not the position."""
    graph = scenes.kitchen()
    mug = graph.get("mug")
    x, y, z = mug.position_m
    mug.position_m = (x, y, z + 1.5)  # as if it were hanging on a wall

    result = _repair(ctx, graph)
    assert all(a.kind is not RepairKind.SNAP_TO_SUPPORT for a in result.actions)
    # And it is left where it was, rather than moved somewhere wrong.
    assert result.graph.get("mug").position_m[2] == pytest.approx(z + 1.5)


def test_a_correction_inside_the_limit_still_happens(ctx):
    """The cap must not disable ordinary repairs — the floating-mug fixture is
    0.25 m, comfortably inside the 0.5 m default."""
    result = _repair(ctx, scenes.mug_floating_above_table())
    assert result.certificate.passed
    assert [a.kind for a in _kept(result)] == [RepairKind.SNAP_TO_SUPPORT]


def test_the_limit_is_configurable(ctx):
    graph = scenes.mug_floating_above_table()  # needs a 0.25 m snap
    tight = ctx.settings.model_copy(update={"max_snap_m": 0.1})
    ctx.settings = tight
    assert not _repair(ctx, graph).certificate.passed


# --- what an action is scored against ------------------------------------------


def test_only_the_target_and_its_dependents_are_scored(tmp_path):
    """A repair is judged on what it can physically reach.

    The whole-scene total is not defeated by noise — `certify` is deterministic to
    the digit — but by sensitivity. Until a scene settles its objects are mid-fall,
    and perturbing any body redirects every trajectory. Measured on `room2.png`,
    snapping the armchair 16.7 mm onto the rug it rests on improved the armchair
    from 23.3 mm of COM drift to 9.8 mm, and was rejected because a book on the far
    side of the room went 565 -> 645 mm. The two are not in contact.
    """
    graph = scenes.kitchen()
    assert repair._affected(graph, "mug") == {"mug"}
    # Everything transitively on the table comes with it.
    assert "mug" in repair._affected(graph, "table")
    assert "cabinet" not in repair._affected(graph, "table")


def test_a_subset_score_ignores_objects_outside_it(ctx):
    """`_violation(ids=...)` is the same sum over fewer checks, which is what makes
    a local comparison meaningful rather than merely smaller."""
    settings = ctx.settings
    cert = certify.run(ctx, scenes.mug_sunk_into_table())

    whole = repair._violation(cert, settings)
    mug_only = repair._violation(cert, settings, {"mug"})
    nothing = repair._violation(cert, settings, set())

    assert nothing == 0.0
    assert 0.0 < mug_only <= whole


# --- the score and the certificate have to agree -------------------------------


def test_the_violation_score_counts_every_criterion_the_check_tests():
    """A criterion `ScaleCheck.passed` tests and `_violation` does not is a repair
    the loop can never earn credit for, and a scene that reaches `score == 0.0` —
    reported as `converged` — with the axis still failing.

    `touching_parent` was the missing one. Three of `room2.png`'s sixteen objects
    fail on it: a plate 591 mm off the table it is recorded as resting on has a
    perfect gap to the floor, is perfectly stable, and is still in the wrong place.
    """
    settings = get_settings()
    failing = Certificate(scale=[ScaleCheck(object_id="plate", deviation_sigma=(0.0, 0.0, 0.0))])
    assert repair._violation(failing, settings) == 0.0

    failing.scale[0].touching_parent = False
    assert repair._violation(failing, settings) > 0.0


# --- the push comes from the measured overlap ---------------------------------


def _pushes(ctx, graph, cert):
    return list(repair._resolve_penetration(graph, cert, ctx.settings))


def test_the_push_is_sized_from_the_measured_depth(ctx):
    """The overlap the stability axis measured on the convex decomposition, not the
    bounding-box intersection. The two agree for solid boxes, which is why the
    fixture still reports 0.202; they do not agree on real geometry, where a box is
    a solid brick from an object's feet to its highest point. Measured on
    `room2.png`, the box version proposed moving a lamp with no overlap at all by
    189.4 mm, and sized a basket's push at 169.1 mm one round and 43.3 mm the next.
    """
    graph = scenes.cabinet_overlapping_table()
    cert = certify.run(ctx, graph)
    depth = max(c.initial_penetration_m for c in cert.stability)

    actions = _pushes(ctx, graph, cert)
    assert len(actions) == 1
    assert actions[0].target_id == "cabinet"
    assert actions[0].magnitude == pytest.approx(depth + ctx.settings.max_penetration_m, abs=1e-9)


def test_nothing_is_proposed_for_an_overlap_the_axis_did_not_name(ctx):
    """A depth with no counterpart is not actionable: which body to move and which
    way is exactly what the counterpart says."""
    graph = scenes.cabinet_overlapping_table()
    cert = certify.run(ctx, graph)
    for check in cert.stability:
        check.penetration_against = None
    assert _pushes(ctx, graph, cert) == []


def test_burial_in_the_floor_is_left_to_snapping(ctx):
    """The floor is not in `graph.objects` and cannot be pushed, so this strategy
    silently had nothing to do — and the gap criterion used to grant 5 mm of burial,
    so snapping had nothing to do either. Measured on `room2.png`, a book 15.3 mm
    into the floor and a bookshelf 5.0 mm into it were failing the stability axis
    with no strategy able to reach either.
    """
    graph = scenes.cabinet_overlapping_table()
    cert = certify.run(ctx, graph)
    for check in cert.stability:
        check.penetration_against = "floor"
    assert _pushes(ctx, graph, cert) == []


def test_a_measured_resting_contact_is_left_alone_even_when_the_relation_disagrees(ctx):
    """`ScaleCheck.resting_on` is what the geometry says holds an object up, and it
    disagrees with `supported_by` exactly when reconcile got the relation wrong.

    Measured on `room2.png`: an armchair recorded as floor-supported but measurably
    resting on a rug was shoved 9.5 mm up off the rug, trading 3.3 mm of overlap for
    a 6.2 mm float that then failed the scale axis. Skipping only the *declared*
    relation makes this strategy's behaviour depend on a field two stages upstream.
    """
    graph = scenes.cabinet_overlapping_table()
    graph.get("mug").supported_by = None  # mislabelled: it is on the table
    cert = certify.run(ctx, graph)
    by_id = {c.object_id: c for c in cert.scale}
    by_id["mug"].resting_on = "table"
    for check in cert.stability:
        if check.object_id in ("mug", "table"):
            check.initial_penetration_m = 0.05
            check.penetration_against = "table" if check.object_id == "mug" else "mug"

    assert {a.target_id for a in _pushes(ctx, graph, cert)} == {"cabinet"}


# --- every failable criterion has a strategy -----------------------------------


def test_each_way_a_scene_can_fail_has_a_repair_that_targets_it():
    """The invariant, asserted rather than assumed.

    A criterion the certificate can fail on with no strategy able to address it
    produces a scene that is reported broken and cannot be acted on. Measured on
    `room2.png` before `slide_onto_support` existed: four of five failing objects
    were outside their contact polygon, snapping moves only in z, and repair
    proposed one action for the whole scene and rejected it.

    Listed explicitly rather than derived, so adding a criterion without a repair
    fails here instead of silently producing unrepairable scenes.
    """
    covered = {
        "support_gap_m": RepairKind.SNAP_TO_SUPPORT,
        "base_inside_parent": RepairKind.SLIDE_ONTO_SUPPORT,
        "touching_parent": RepairKind.REPARENT,
        "initial_penetration_m": RepairKind.RESOLVE_PENETRATION,
        "deviation_sigma": RepairKind.RESCALE,
    }
    failable = {
        name
        for name, field in (ScaleCheck.model_fields | StabilityCheck.model_fields).items()
        if name not in {"object_id", "passed", "resting_on", "penetration_against"}
    }
    # These two are consequences measured after settling, not inputs a repair can
    # target: an object that drifts is fixed by correcting whatever made it drift.
    failable -= {"com_displacement_m", "orientation_drift_deg"}

    assert failable == set(covered), (
        f"criteria with no repair: {sorted(failable - set(covered))}; "
        f"repairs for criteria that no longer exist: {sorted(set(covered) - failable)}"
    )


def test_an_object_touching_nothing_is_still_slid_back_onto_its_support(ctx):
    """The gap between the two strategies, and the plate that fell into it.

    `_slide_onto_support` aims at the polygon of points the object is touching, and
    an object clear of everything touches nothing — so it used to defer to
    `_snap_to_support` as "a gap failure, not a lateral one". That is only true when
    something is underneath to snap *to*. Between them the pair covered every case
    except an object that has drifted off its support entirely, which is exactly
    what reconstruction produces: `room2.png`'s plate drew no proposal of any kind
    across five rounds.

    The fallback asks the parent's own geometry where the object *could* stand —
    `SupportHeights.nearest_support_xy`, the same query the solver's containment
    term uses — which needs no existing contact.
    """
    graph = scenes.mug_adrift_from_its_support()
    before = certify.run(ctx, graph)
    mug = next(c for c in before.scale if c.object_id == "mug")
    assert not mug.base_inside_parent, "the fixture should start off its support"

    result = _repair(ctx, graph)

    slides = [a for a in _kept(result) if a.kind is RepairKind.SLIDE_ONTO_SUPPORT]
    assert slides and slides[0].target_id == "mug", (
        f"nothing slid the mug back: {[(a.kind.value, a.target_id) for a in result.actions]}"
    )
    after = next(c for c in result.certificate.scale if c.object_id == "mug")
    assert after.base_inside_parent, "it is over the table again"


# --- scoping to an edit -------------------------------------------------------


def test_only_ids_confines_repair_to_the_objects_named(ctx):
    """What lets a commit repair without rearranging the room.

    The edit endpoint used to refuse repair outright, because an edit that moved
    objects the user did not touch is a surprising thing for a drag to do. Bounding
    the proposals answers that directly, and it is the whole mechanism — the
    accept/reject machinery needs no change, because `_violation` already scores
    over `_affected(target)` rather than the whole scene.
    """
    graph = scenes.cabinet_overlapping_table()
    unscoped = _repair(ctx, graph)
    assert {a.target_id for a in unscoped.actions} - {"mug"}, (
        "the fixture must fail on something other than the mug, or this proves nothing"
    )

    scoped = repair.run(ctx, graph, certify.run(ctx, graph), only_ids={"mug"})
    assert {a.target_id for a in scoped.actions} <= {"mug"}


def test_only_ids_of_an_untouched_object_changes_nothing(ctx):
    """An empty scope is a no-op, not a licence to repair everything."""
    graph = scenes.cabinet_overlapping_table()
    result = repair.run(ctx, graph, certify.run(ctx, graph), only_ids=set())

    assert result.actions == []
    assert result.graph == graph
