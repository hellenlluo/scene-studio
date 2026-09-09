"""Minimal correction of a failing scene, then re-certification.

Repair is underdetermined, and that is the difficulty rather than a detail. Given
two objects occupying the same space, the fix could be moving either body,
rescaling either, or upgrading a collision proxy too coarse to sit flush. Nothing
in the failure says which.

The policy here is **cheapest and least destructive first**:

1. `SNAP_TO_SUPPORT` — close the gap to whatever the object rests on. Pure
   translation along one axis, and it fixes both floating and buried objects.
2. `RESOLVE_PENETRATION` — push overlapping bodies apart along their shallowest
   axis of overlap.
3. `RESCALE` — only where the object's scale deviation was *already* out of
   tolerance.

Rescaling is last on purpose and should stay reluctant: scale is the variable the
whole system exists to estimate, so moving it to satisfy a contact quietly
discards the measurement. It is only allowed where the estimate was independently
failing anyway, which is the one case where it is a correction rather than a
cover-up.

`UPGRADE_PROXY_TIER` is in the vocabulary but not implemented — it needs mesh
geometry the pipeline does not produce yet, and a repair that cannot run is worse
than one that is honestly absent.

**The loop terminates on its own.** Every accepted repair strictly decreases a
violation score bounded below by zero, so it cannot oscillate and needs no iteration
cap for correctness. `Settings.max_repair_rounds` is a *cost* budget: it only binds
when repairs cascade, and `RepairResult.converged` says which reason the loop
stopped for, because a truncated repair is not the same as a scene that cannot be
repaired.

**Every action is measured, and rejected ones are still reported.** A repair is
applied to a trial graph, the scene is re-certified, and the change is kept only if
total violation went down. Otherwise it is reverted — but recorded with
`improved=False`, because a silent no-op tells you nothing and knowing what was
tried and did not work is exactly what you need when a scene will not certify.
"""

import logging

import numpy as np

from app.config import Settings
from app.geometry import world_aabb
from app.pipeline import support
from app.pipeline.base import PipelineContext
from app.schemas import (
    Certificate,
    Provenance,
    RepairAction,
    RepairKind,
    RepairResult,
    SceneGraph,
)

from . import certify, scale

__all__ = ["RepairResult", "run"]

log = logging.getLogger(__name__)

# A repair has to buy more than floating-point noise to be worth keeping.
MIN_IMPROVEMENT = 1e-6


def _affected(graph: SceneGraph, target_id: str) -> set[str]:
    """The target and everything transitively resting on it.

    The set a repair can *physically* reach. Moving a table moves the mug on it and
    the tray under the mug; it does not move a lamp across the room, whatever a
    whole-scene re-simulation says about the lamp afterwards.
    """
    reached = {target_id}
    frontier = [target_id]
    while frontier:
        parent = frontier.pop()
        for obj in graph.objects:
            if obj.supported_by == parent and obj.object_id not in reached:
                reached.add(obj.object_id)
                frontier.append(obj.object_id)
    return reached


def _violation(cert: Certificate, settings: Settings, ids: set[str] | None = None) -> float:
    """How far past its thresholds the scene sits, in units of threshold.

    Normalising by the threshold is what makes the axes comparable: 3 mm of
    penetration and 4 degrees of drift are not otherwise on the same scale, and a
    repair has to be judged against the total rather than one favoured number.
    Zero means everything passes.

    `ids` narrows it to a subset, which is how an action is scored — see `run`.
    None is the whole scene, which is what the caller wants for reporting.
    """

    def counted(object_id: str) -> bool:
        return ids is None or object_id in ids

    total = 0.0
    for check in cert.scale:
        if not counted(check.object_id):
            continue
        # Asymmetric, because `ScaleCheck.passed` is — see `scale.gap_violation`.
        total += scale.gap_violation(
            check.support_gap_m, settings.max_support_gap_m, settings.max_penetration_m
        )
        worst_sigma = max(abs(d) for d in check.deviation_sigma)
        total += max(0.0, worst_sigma / settings.max_prior_deviation_sigma - 1.0)
        total += 0.0 if check.base_inside_parent else 1.0
        # Every criterion `ScaleCheck.passed` tests has to appear here, or the score
        # and the certificate disagree about what is wrong. This one was missing, so
        # a repair that put an object back on the thing it is recorded as resting on
        # earned nothing and was reverted, and a scene could reach `score == 0.0` —
        # reported as `converged` — with the axis still failing. Three of `room2.png`'s
        # sixteen objects fail on this alone.
        total += 0.0 if check.touching_parent else 1.0
    for check in cert.stability:
        if not counted(check.object_id):
            continue
        total += max(0.0, check.com_displacement_m / settings.max_com_displacement_m - 1.0)
        total += max(0.0, check.orientation_drift_deg / settings.max_orientation_drift_deg - 1.0)
        total += max(0.0, check.initial_penetration_m / settings.max_penetration_m - 1.0)
    for check in cert.inertial:
        if not counted(check.object_id):
            continue
        total += 0.0 if check.passed else 1.0
    return total


def _apply(graph: SceneGraph, action: RepairAction) -> SceneGraph:
    """Return a copy of the graph with the action applied.

    A copy rather than a mutation because the caller has to be able to throw the
    result away when re-certification says the repair did not help.
    """
    trial = graph.model_copy(deep=True)
    obj = trial.get(action.target_id)
    if obj is None:
        return trial

    if any(action.delta_position_m):
        # Everything resting on it comes too. A rigid translation of a support is a
        # translation of the assembly — lift a table and the mug on it goes up with
        # it — and applying one to the parent alone drives it through its own
        # children. Measured on `room2.png`: `snap_to_support` raised a bookshelf
        # 11.7 mm to close its own gap, left the vase on its shelf where it was, and
        # buried it 11.2 mm in the shelf; the vase then cost 39% of the scene's
        # violation, and repair spent later rounds snapping it back out again.
        #
        # `_resolve_penetration` already refuses to move a support for exactly this
        # reason. Carrying the dependents is the other way to be right about it, and
        # the one that lets a floating support still be corrected.
        for object_id in _affected(trial, action.target_id):
            moved = trial.get(object_id)
            if moved is None:
                continue
            moved.position_m = tuple(
                float(a + b) for a, b in zip(moved.position_m, action.delta_position_m, strict=True)
            )
            moved.provenance["position_m"] = Provenance.DERIVED
    if action.kind is RepairKind.REPARENT:
        # Bookkeeping only — nothing moves, so no dependants follow and no
        # provenance changes on a position nobody touched.
        obj.supported_by = action.new_parent_id

    if action.delta_scale != 1.0:
        obj.scale *= action.delta_scale
        obj.provenance["scale"] = Provenance.DERIVED
    return trial


# --- the three strategies -----------------------------------------------------


def _snap_to_support(graph: SceneGraph, cert: Certificate, settings: Settings):
    """Close the gap to whatever an object rests on.

    The cheapest repair and the most common one: one translation along z, no
    change to scale or to any other object.

    Bounded by `max_snap_m`. A correction large enough to move an object across the
    room is not a minimal correction, and at that magnitude the support *relation*
    is the likelier error — so it is reported rather than acted on.

    Triggering on `scale.gap_ok` rather than on `abs(gap)` is what makes this the
    repair for burial in a support, including burial in the floor. It is the only
    one: `_resolve_penetration` needs two objects and the floor is not one.
    """
    for check in cert.scale:
        gap = check.support_gap_m
        if scale.gap_ok(gap, settings.max_support_gap_m, settings.max_penetration_m):
            continue
        if abs(gap) > settings.max_snap_m:
            # Not a placement error at this magnitude — the support relation itself
            # is wrong, and translating the object would bury the evidence. A
            # wall-mounted picture misread as floor-supported is the motivating
            # case: snapping it would drop the painting onto the carpet.
            log.info(
                "%s is %.2f m from its support, beyond the %.2f m snap limit; "
                "the support relation is the likelier error",
                check.object_id,
                gap,
                settings.max_snap_m,
            )
            continue
        yield RepairAction(
            kind=RepairKind.SNAP_TO_SUPPORT,
            target_id=check.object_id,
            axis_repaired="scale",
            delta_position_m=(0.0, 0.0, -gap),
            magnitude=abs(gap),
        )


def _resolve_penetration(graph: SceneGraph, cert: Certificate, settings: Settings):
    """Push overlapping bodies apart, on the overlap the certificate measured.

    Driven by `StabilityCheck` — the pair the axis named and the depth it measured
    on the convex decomposition — rather than re-derived from bounding boxes. The
    box version asked a different question and got three different wrong answers on
    `room2.png`:

    * It proposed moving a **lamp with no measured overlap at all** by 189.4 mm,
      because a failing object's box happened to intersect the lamp's. Boxes are
      solid bricks from an object's feet to its highest point, so they intersect
      constantly where the meshes do not — the same error `app.pipeline.penetration`
      exists to keep out of the solver.
    * It sized the push from the box overlap, so the distance had nothing to do with
      the overlap being repaired: 169.1 mm proposed for a basket, 43.3 mm for the
      same pair one round later.
    * It skipped only *declared* support relations, so it shoved an armchair 9.5 mm
      up off the rug it measurably rests on, trading 3.3 mm of overlap for a 6.2 mm
      float that then failed the scale axis. `ScaleCheck.resting_on` is what the
      geometry says holds an object up, and it disagrees with `supported_by` exactly
      when reconcile got the relation wrong.

    Which body moves is the underdetermined part. The rule: never move an object
    that other things rest on, since that drags its dependents along and turns one
    repair into several. Among equals, move the lighter one, on the grounds that a
    mug is more likely to be misplaced than a counter.
    """
    # What the geometry says each object rests on, alongside what it claims to.
    resting_on = {check.object_id: check.resting_on for check in cert.scale}
    supports = {obj.supported_by for obj in graph.objects if obj.supported_by}

    seen: set[tuple[str, str]] = set()
    for check in cert.stability:
        # Against the tolerance, not against zero. `max_penetration_m` is non-zero
        # precisely because a resting contact has a little overlap in it — measured, a
        # floor lamp standing on the floor reports 4.7 micrometres, six orders of
        # magnitude under the 5 mm support tolerance. Triggering on any positive value
        # proposes a separation for objects that are simply touching.
        if check.initial_penetration_m <= settings.max_penetration_m:
            continue

        first = graph.get(check.object_id)
        second = graph.get(check.penetration_against) if check.penetration_against else None
        if first is None or second is None:
            # `"floor"`, or a counterpart the axis could not name. An object sunk into
            # the surface beneath it is buried in its own support, and `_snap_to_support`
            # owns that — it can see it now that the gap criterion grades burial against
            # `max_penetration_m`. The floor is not in `graph.objects` and cannot be
            # pushed, so there was never anything for this strategy to do: measured on
            # `room2.png`, a book 15.3 mm into the floor and a bookshelf 5.0 mm into it
            # were failing the stability axis with no strategy able to reach either.
            continue

        # A resting contact is wanted, not a violation, and snapping drives those two
        # surfaces together. Tested against the measured relation as well as the
        # declared one; see the docstring.
        if second.object_id in (first.supported_by, resting_on.get(first.object_id)):
            continue
        if first.object_id in (second.supported_by, resting_on.get(second.object_id)):
            continue

        key = (min(first.object_id, second.object_id), max(first.object_id, second.object_id))
        if key in seen:
            continue
        seen.add(key)

        movable = [o for o in (first, second) if o.object_id not in supports]
        if not movable:
            movable = [first, second]
        mover = min(movable, key=_mass)
        other = second if mover is first else first

        low_a, high_a = world_aabb(mover)
        low_b, high_b = world_aabb(other)
        # The boxes still choose the *direction* — the shallowest axis of overlap is
        # the cheapest way out, and a direction is all they are being asked for. Only
        # axes the two actually overlap on qualify; z is the fallback, because lifting
        # is the one separation that never drives a body further into a third.
        overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
        usable = np.where(overlap > 0.0, overlap, np.inf)
        axis = int(np.argmin(usable)) if np.isfinite(usable).any() else 2

        # The distance comes from the measured depth, plus the tolerance so the result
        # lands clear of the bound rather than exactly on it.
        push = check.initial_penetration_m + settings.max_penetration_m
        direction = 1.0 if low_a[axis] > low_b[axis] else -1.0
        delta = [0.0, 0.0, 0.0]
        delta[axis] = direction * push

        yield RepairAction(
            kind=RepairKind.RESOLVE_PENETRATION,
            target_id=mover.object_id,
            axis_repaired="stability",
            delta_position_m=(delta[0], delta[1], delta[2]),
            magnitude=push,
        )


def _mass(obj) -> float:
    return sum(p.inertial.mass_kg for p in obj.parts if p.inertial) or float("inf")


def _rescale_toward_prior(graph: SceneGraph, cert: Certificate, settings: Settings):
    """Pull an object back toward its class prior.

    Last resort, and gated on the scale deviation *already* being out of tolerance.
    Scale is the quantity the whole system exists to estimate; rescaling to satisfy
    a contact would quietly discard that measurement. Where the estimate was
    failing on its own terms, correcting it is a repair rather than a cover-up.
    """
    for check in cert.scale:
        worst = max(abs(d) for d in check.deviation_sigma)
        if worst <= settings.max_prior_deviation_sigma:
            continue
        obj = graph.get(check.object_id)
        if obj is None or obj.label.prior is None:
            # Nothing to pull toward. An object with no prior cannot have exceeded
            # one, so this should be unreachable — but rescaling toward a missing
            # target would be the worst possible way to find that out.
            continue

        current = np.asarray(obj.dims_m, dtype=float)
        target = np.asarray(obj.label.prior.dims_m, dtype=float)
        if np.any(current <= 0):
            continue
        # One isotropic factor, so the best it can do is agree on average.
        factor = float(np.mean(target / current))
        if not np.isfinite(factor) or factor <= 0 or abs(factor - 1.0) < 1e-9:
            continue

        yield RepairAction(
            kind=RepairKind.RESCALE,
            target_id=check.object_id,
            axis_repaired="scale",
            delta_scale=factor,
            magnitude=abs(factor - 1.0),
        )


def _slide_onto_support(graph: SceneGraph, cert: Certificate, settings: Settings):
    """Move an object sideways until its weight is over what it is standing on.

    The lateral counterpart of `_snap_to_support`, and the strategy whose absence
    left most of a scene unrepairable: measured on `room2.png`, four of five failing
    objects were outside their contact polygon and every repair pass proposed a
    single action for the whole scene, because snapping moves only in z and pushing
    penetrations apart only applies to objects that overlap.

    The correction is the shortest vector from the footprint centre to the polygon
    of points the object is actually touching — the same polygon `certify.scale`
    tests against, so this closes exactly the criterion that failed rather than an
    approximation of it. Bounded by `max_snap_m` for the reason snapping is: past
    that, an object is not slightly misplaced, it is somewhere else.
    """
    heights = support.build(graph, settings)
    for check in cert.scale:
        if check.base_inside_parent:
            continue
        obj = graph.get(check.object_id)
        if obj is None or obj.supported_by is None:
            # Nothing to be over. An object on the floor is always supported.
            continue

        contact = heights.contact(
            obj, graph.objects, graph.floor_height_m, settings.max_support_gap_m
        )
        low, high = world_aabb(obj)
        centre = (low[:2] + high[:2]) / 2.0
        target = _nearest_in_hull(centre, contact.contacts_xy)
        if target is None:
            # Touching nothing, so there is no contact hull to aim at. This used to
            # defer to `_snap_to_support` as "a gap failure, not a lateral one",
            # which is only true when something is underneath to snap *to*. An
            # object that has drifted clear of its support has neither, and snapping
            # translates in z alone, so between them the two strategies covered
            # every case except this one — measured on `room2`, a plate 138 mm
            # outside the side table it is recorded as resting on drew no proposal
            # of any kind across five repair rounds.
            #
            # So fall back to the parent's own geometry: `nearest_support_xy` is the
            # same query the solver's containment term aims at, and it answers
            # "where on this parent could the object stand" without needing the
            # object to already be standing there. On that plate it returns a point
            # 228 mm away, inside `max_snap_m`.
            parent = graph.get(obj.supported_by)
            base = heights.base_of(obj)
            target = (
                None
                if parent is None
                else heights.nearest_support_xy(
                    parent,
                    low,
                    high,
                    float(low[2]) if base is None else base,
                    settings.support_burial_slack_m,
                )
            )
        if target is None:
            continue

        delta = target - centre
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6 or distance > settings.max_snap_m:
            continue

        yield RepairAction(
            kind=RepairKind.SLIDE_ONTO_SUPPORT,
            target_id=check.object_id,
            axis_repaired="scale",
            delta_position_m=(float(delta[0]), float(delta[1]), 0.0),
            magnitude=distance,
        )


def _nearest_in_hull(point: np.ndarray, hull_points: np.ndarray):
    """The closest point on the convex hull of `hull_points`, or None if there is no
    hull to speak of. Returns `point` itself when it is already inside."""
    if len(hull_points) < 3:
        return None
    try:
        from scipy.spatial import ConvexHull, Delaunay

        hull = hull_points[ConvexHull(hull_points).vertices]
        if Delaunay(hull).find_simplex(point) >= 0:
            return point
    except Exception:
        return None

    # Closest point on each hull edge; the nearest of those is the nearest on the
    # hull. Small enough that a loop is clearer than a vectorised distance.
    best, best_d = None, float("inf")
    for a, b in zip(hull, np.roll(hull, -1, axis=0), strict=True):
        edge = b - a
        length = float(edge @ edge)
        t = 0.0 if length == 0 else float(np.clip((point - a) @ edge / length, 0.0, 1.0))
        candidate = a + t * edge
        d = float(np.linalg.norm(candidate - point))
        if d < best_d:
            best, best_d = candidate, d
    return best


def _reparent(graph: SceneGraph, cert: Certificate, settings: Settings):
    """Record an object against the thing it is measurably resting on.

    The cheapest repair there is: nothing moves. `touching_parent` fails when an
    object does not meet the thing it is recorded as resting on, and when it is
    measurably resting on something *else* the geometry was right all along and the
    bookkeeping was wrong.

    Only when the measurement disagrees with the record, and only when the object is
    genuinely in contact with the alternative — an object touching nothing is a
    placement failure, and moving its label would bury the evidence rather than fix
    anything.
    """
    heights = support.build(graph, settings)
    for check in cert.scale:
        if check.touching_parent:
            continue
        obj = graph.get(check.object_id)
        if obj is None:
            continue
        contact = heights.contact(
            obj, graph.objects, graph.floor_height_m, settings.max_support_gap_m
        )
        if contact.gap is None or abs(contact.gap) > settings.max_support_gap_m:
            continue  # resting on nothing; not this strategy's problem
        if contact.resting_on == obj.supported_by:
            continue

        yield RepairAction(
            kind=RepairKind.REPARENT,
            target_id=check.object_id,
            axis_repaired="scale",
            new_parent_id=contact.resting_on,
            magnitude=0.0,
        )


# One per criterion `certify.scale` and `certify.stability` can fail on. A
# criterion with no strategy produces a scene that is reported broken and cannot
# be acted on; `test_repair` asserts the pairing.
_STRATEGIES = (
    _snap_to_support,
    _slide_onto_support,
    _reparent,
    _resolve_penetration,
    _rescale_toward_prior,
)


def run(
    ctx: PipelineContext,
    graph: SceneGraph,
    certificate: Certificate,
    only_ids: set[str] | None = None,
) -> RepairResult:
    """Repair `graph`, optionally confining the correction to `only_ids`.

    `None` repairs the whole scene, which is what the explicit repair endpoint
    wants: the user asked for the scene to be fixed and every failing object is
    fair game.

    A set is the commit path. Repair now runs on every edit, and an unbounded
    repair there would move objects the user never touched — the surprise the edit
    endpoint had until now avoided by not repairing at all. Scoped to the edited
    objects and their dependents, it cleans up after the edit and stops there.

    Filtering proposals is enough; the accept/reject machinery needs no change,
    because `_violation` is already scored over `_affected(target)` rather than the
    whole scene. An action outside the set is never proposed, so it is never
    applied and never reported.
    """
    settings = ctx.settings
    current_graph = graph.model_copy(deep=True)
    current_cert = certificate
    score = _violation(current_cert, settings)
    actions: list[RepairAction] = []

    rounds_used = 0
    converged = False
    for _ in range(settings.max_repair_rounds):
        rounds_used += 1
        accepted_this_round = False
        # Objects whose measurements an accepted action has already invalidated this
        # round. Every strategy derives its proposals from the certificate the round
        # opened with, and accepting one makes the rest of them stale: `_apply`
        # carries the target's dependents with it, and pushing two bodies apart moves
        # what a third was overlapping. Measured on `room2.png`: after a bookshelf
        # was snapped 109.9 mm down, the book resting on it — which came along for
        # the ride — was still snapped by the gap measured before the shelf moved,
        # and finished 15.3 mm inside the floor. Deferring to the next round costs a
        # round and re-certifies first, which is the only way to know the delta is
        # still the right one.
        stale: set[str] = set()

        for strategy in _STRATEGIES:
            for action in strategy(current_graph, current_cert, settings):
                if only_ids is not None and action.target_id not in only_ids:
                    continue
                # Scored on what the action can reach, not on the whole scene.
                #
                # The scene-wide total sounds fairer and is not, because it is not
                # noise that defeats it — `certify` is deterministic to the digit —
                # but sensitivity. Until a scene settles its objects are mid-fall,
                # and perturbing any body redirects every trajectory. Measured on
                # `room2.png`: snapping the armchair 16.7 mm onto the rug it rests
                # on improved the armchair from 23.3 mm of COM drift to 9.8 mm, and
                # was rejected because a book on the far side of the room went from
                # 565 mm to 645 mm. The two are not in contact. Judging the action
                # on the armchair and its dependents keeps the part of that
                # comparison which means something.
                affected = _affected(current_graph, action.target_id)
                if affected & stale:
                    continue
                before = _violation(current_cert, settings, affected)

                trial = _apply(current_graph, action)
                trial_cert = certify.run(ctx, trial)
                trial_score = _violation(trial_cert, settings, affected)

                if trial_score < before - MIN_IMPROVEMENT:
                    current_graph, current_cert = trial, trial_cert
                    score = _violation(trial_cert, settings)
                    action.improved = True
                    accepted_this_round = True
                    # Everything the move reached, plus whatever those bodies are now
                    # overlapping — a penetration is measured between two objects, so
                    # moving one restates the other's number too.
                    stale |= affected
                    stale |= {
                        c.object_id for c in trial_cert.stability if c.penetration_against in stale
                    }
                else:
                    # Reverted, but still recorded: knowing what was tried and did
                    # not work is exactly what you need when a scene will not
                    # certify, and a silent no-op tells you nothing.
                    action.improved = False
                actions.append(action)

        # The normal exit. Running out of rounds is the abnormal one, and the
        # caller needs to be able to tell those apart: a truncated repair is not
        # the same as a scene that cannot be repaired.
        if not accepted_this_round or score == 0.0:
            converged = True
            break

    return RepairResult(
        graph=current_graph,
        certificate=current_cert,
        actions=actions,
        rounds_used=rounds_used,
        converged=converged,
    )
