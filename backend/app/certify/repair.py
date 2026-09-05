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
from app.pipeline.base import PipelineContext
from app.schemas import (
    Certificate,
    Provenance,
    RepairAction,
    RepairKind,
    RepairResult,
    SceneGraph,
)

from . import certify

__all__ = ["RepairResult", "run"]

log = logging.getLogger(__name__)

# A repair has to buy more than floating-point noise to be worth keeping.
MIN_IMPROVEMENT = 1e-6


def _violation(cert: Certificate, settings: Settings) -> float:
    """How far past its thresholds the whole scene sits, in units of threshold.

    Normalising by the threshold is what makes the axes comparable: 3 mm of
    penetration and 4 degrees of drift are not otherwise on the same scale, and a
    repair has to be judged against the total rather than one favoured number.
    Zero means everything passes.
    """
    total = 0.0
    for check in cert.scale:
        total += max(0.0, abs(check.support_gap_m) / settings.max_support_gap_m - 1.0)
        worst_sigma = max(abs(d) for d in check.deviation_sigma)
        total += max(0.0, worst_sigma / settings.max_prior_deviation_sigma - 1.0)
        total += 0.0 if check.base_inside_parent else 1.0
    for check in cert.stability:
        total += max(0.0, check.com_displacement_m / settings.max_com_displacement_m - 1.0)
        total += max(0.0, check.orientation_drift_deg / settings.max_orientation_drift_deg - 1.0)
        total += max(0.0, check.initial_penetration_m / settings.max_penetration_m - 1.0)
    for check in cert.inertial:
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
        obj.position_m = tuple(
            float(a + b) for a, b in zip(obj.position_m, action.delta_position_m, strict=True)
        )
        obj.provenance["position_m"] = Provenance.DERIVED
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
    """
    for check in cert.scale:
        gap = check.support_gap_m
        if abs(gap) <= settings.max_support_gap_m:
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
    """Push overlapping bodies apart along their shallowest axis of overlap.

    Which body moves is the underdetermined part. The rule here: never move an
    object that other things rest on, since that silently drags its dependents
    with it and turns one repair into several. Among equals, move the lighter one,
    on the grounds that a mug is more likely to be misplaced than a counter.
    """
    # Against the tolerance, not against zero. `max_penetration_m` is non-zero
    # precisely because a resting contact has a little overlap in it — measured, a
    # floor lamp standing on the floor reports 4.7 micrometres, six orders of
    # magnitude under the 5 mm support tolerance. Triggering on any positive value
    # proposes a separation for objects that are simply touching, which then fails
    # to improve anything and spends a round of the repair budget finding that out.
    # The violation score below already compares against the tolerance; only the
    # candidate selection did not.
    failing = {
        c.object_id for c in cert.stability if c.initial_penetration_m > settings.max_penetration_m
    }
    if not failing:
        return

    bounds = {obj.object_id: world_aabb(obj) for obj in graph.objects}
    supports = {obj.supported_by for obj in graph.objects if obj.supported_by}

    seen: set[tuple[str, str]] = set()
    for first in graph.objects:
        for second in graph.objects:
            if first.object_id >= second.object_id:
                continue
            if first.object_id not in failing and second.object_id not in failing:
                continue
            # A support relation is a *wanted* contact; snapping handles those.
            if second.supported_by == first.object_id or first.supported_by == second.object_id:
                continue

            low_a, high_a = bounds[first.object_id]
            low_b, high_b = bounds[second.object_id]
            overlap = np.minimum(high_a, high_b) - np.maximum(low_a, low_b)
            if np.any(overlap <= settings.max_penetration_m):
                continue  # separated on at least one axis, so not interpenetrating

            axis = int(np.argmin(overlap))
            push = float(overlap[axis]) + settings.max_penetration_m

            movable = [o for o in (first, second) if o.object_id not in supports]
            if not movable:
                movable = [first, second]
            mover = min(movable, key=_mass)
            other = second if mover is first else first

            direction = (
                1.0 if bounds[mover.object_id][0][axis] > bounds[other.object_id][0][axis] else -1.0
            )
            delta = [0.0, 0.0, 0.0]
            delta[axis] = direction * push

            key = (first.object_id, second.object_id)
            if key in seen:
                continue
            seen.add(key)

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


_STRATEGIES = (_snap_to_support, _resolve_penetration, _rescale_toward_prior)


def run(ctx: PipelineContext, graph: SceneGraph, certificate: Certificate) -> RepairResult:
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

        for strategy in _STRATEGIES:
            for action in strategy(current_graph, current_cert, settings):
                trial = _apply(current_graph, action)
                trial_cert = certify.run(ctx, trial)
                trial_score = _violation(trial_cert, settings)

                if trial_score < score - MIN_IMPROVEMENT:
                    current_graph, current_cert, score = trial, trial_cert, trial_score
                    action.improved = True
                    accepted_this_round = True
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
