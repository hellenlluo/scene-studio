"""Stage 6: coupled optimisation over scale and pose. The core stage.

Reconcile places each object from depth alone, one at a time. That is the best any
object can do in isolation, and it is not enough: objects that individually match
their own depth measurement still sink into each other, float above their supports,
and drift when gravity is applied. Those violations carry information, and this
stage is where they are allowed to move the estimate.

Variables are per-object isotropic scale and position — 4 per object. Orientation is
held: yaw comes from the depth cloud's principal axis in stage 5, and refining it
here would need a residual that actually sees rotation, which the axis-aligned terms
below do not.

    E = w_depth * E_depth   fitted extent and centre against the depth measurement
      + w_supp  * E_supp    signed gap on each support edge
      + w_pen   * E_pen     pairwise overlap
      + w_phys  * E_phys    displacement measured by settling the scene in MuJoCo

**Solved as block coordinate descent, not one optimisation.** The first three terms
are smooth and go to `scipy.optimize.least_squares`. `E_phys` is not smooth — it
comes from stepping a physics engine — so it enters as a residual computed *between*
Gauss-Newton passes and held fixed within them. A coupled solve that fails to
converge is worse than two that do, and this ordering degrades gracefully to
sequential-with-feedback if the loop is stopped after one round.

**E_phys is the coupling made concrete.** Settle the scene, measure where each
object ends up, and pull the estimate toward the settled pose. An object that slides
when gravity is applied was mis-placed, and the distance it slid is the correction.
That is physics feedback moving scale and position, which is the claim the whole
approach rests on — and the iteration count is the evidence for it.

Scale is optimised as **log scale**: it keeps the value positive without a
constraint, and makes a doubling and a halving the same size of step, which they are
not in linear space.

Not yet here: `E_prior` (no measured priors table, so the term has no weight),
`E_sil` (needs a renderer), and anisotropic terms of any kind.
"""

import logging

import numpy as np
from scipy import optimize

from app.geometry import world_aabb
from app.pipeline import penetration, support
from app.pipeline.base import PipelineContext
from app.schemas import (
    DepthObservation,
    DepthResult,
    Provenance,
    ScaleAnchor,
    SceneGraph,
    SceneObject,
    SegmentResult,
    SolveDiagnostics,
    SolveResult,
    SolveWeights,
)

__all__ = ["SolveResult", "run"]

log = logging.getLogger(__name__)

# Outer block-coordinate rounds: smooth solve, then settle, then re-solve against
# what settling revealed. The loop exits early the moment both flags are true, so
# this only binds on a scene that never gets there.
#
# Five because the useful work is done by four and there is nothing after it.
# Measured on `room`, which contains one object that topples: moved per round went
# 199, 40, 16, 6.9 mm and then *plateaued at exactly 8.0 mm*, with the settling
# drift alternating 260, 331, 260, 331 forever — a period-2 limit cycle, not slow
# convergence. The solver places the lamp, `_settle` reports it fallen, the physics
# term pulls the estimate toward the fallen pose, the depth and support terms pull
# it back, and it falls the other way. Ten rounds cost twice the time and produced
# an identical scene.
#
# So the cap is not a convergence budget, it is a cutoff for the pathological case.
# A scene with nothing unstable in it exits early and never reaches this.
MAX_ROUNDS = 5
ROUND_TOLERANCE_M = 2e-3

# Weight on the settled-pose residual. Deliberately below the depth weight: settling
# reveals that a placement is wrong, but the settled pose is not itself a
# measurement of where the object belongs — a mug knocked off a table lands on the
# floor, and the floor is not the answer.
PHYSICS_WEIGHT = 0.5


def _pack(objects: list[SceneObject]) -> np.ndarray:
    return np.concatenate([[np.log(obj.scale), *obj.position_m] for obj in objects]).astype(float)


def _unpack(objects: list[SceneObject], x: np.ndarray) -> None:
    for index, obj in enumerate(objects):
        block = x[index * 4 : index * 4 + 4]
        obj.scale = float(np.exp(block[0]))
        obj.position_m = (float(block[1]), float(block[2]), float(block[3]))


# How far a variable may move from where it started. Bounds rather than penalties,
# so the search cannot trade a physics violation for a scale one and report success:
# without them the penetration term will push an object arbitrarily far to stop it
# overlapping.
_SCALE_FACTOR = 4.0
_POSITION_RANGE_M = 1.0


def _bounds(objects: list[SceneObject], observations) -> tuple[np.ndarray, np.ndarray]:
    """Keep the search near the measurement rather than unbounded.

    **An object with no depth observation has its scale frozen.** Nothing else in
    the objective carries absolute size: `E_prior` has no priors table yet, and the
    support and penetration terms only care about surfaces meeting. Left free, the
    solver discovers it can close a contact by shrinking the object as easily as by
    moving it, and does — measured on a hand-authored fixture with no observations,
    a mug re-solved to a quarter of its size with a support gap of 1e-10 and a
    straight face. No measurement of a thing's size is not a licence to change it.

    This never binds on pipeline output, where every object carries an observation.
    It binds on the fixtures, and on any scene assembled by hand.
    """
    low, high = [], []
    for obj in objects:
        logged = np.log(obj.scale)
        measured = obj.object_id in observations
        span = np.log(_SCALE_FACTOR) if measured else 0.0
        low += [logged - span, *(p - _POSITION_RANGE_M for p in obj.position_m)]
        high += [logged + span, *(p + _POSITION_RANGE_M for p in obj.position_m)]
    # `least_squares` rejects a zero-width interval, so a frozen scale is given the
    # narrowest one it will accept rather than an exactly equal pair.
    lower, upper = np.asarray(low), np.asarray(high)
    upper = np.maximum(upper, lower + 1e-12)
    return lower, upper


class _Penetrations:
    """The FCL objects plus the pair list, so neither is rebuilt per evaluation.

    The pairs depend only on the support graph, which does not change during a
    solve, and rebuilding them inside the residual would be the whole cost of the
    term.
    """

    def __init__(self, graph: SceneGraph, settings):
        self._depths = penetration.build(graph, settings)
        self.pairs = penetration.pairs(graph)
        # Support contacts, which `pairs` deliberately leaves out. Kept separately
        # because they are scored differently: see the residual.
        self.support_pairs = [
            (obj.object_id, obj.supported_by) for obj in graph.objects if obj.supported_by
        ]
        self.tolerance = settings.max_penetration_m

    def between(self, first, second) -> float:
        if first is None or second is None:
            return 0.0
        return self._depths.between(first, second)


def _residuals(
    x: np.ndarray,
    graph: SceneGraph,
    free: list[SceneObject],
    observations: dict,
    weights: SolveWeights,
    anchors: list[ScaleAnchor],
    settled: dict[str, np.ndarray] | None,
    supports: support.SupportHeights,
    penetrations: _Penetrations,
    edits: dict[str, tuple[np.ndarray | None, np.ndarray | None]],
) -> np.ndarray:
    _unpack(free, x)
    ctx_slack = supports.burial_slack_m

    boxes = {obj.object_id: world_aabb(obj) for obj in graph.objects}
    out: list[float] = []

    for obj in graph.objects:
        low, high = boxes[obj.object_id]
        centre, extent = (low + high) / 2.0, high - low

        observed = observations.get(obj.object_id)
        if observed is not None:
            # Both halves matter: extent alone fixes size but lets the object drift,
            # centre alone fixes position but lets it grow without bound.
            #
            # Either half may be aimed at the user's edit instead of at the depth
            # measurement — see `_edit_priors`. Substituted rather than added, so the
            # residual vector keeps its length and the two never pull against each
            # other: a term that both held the measurement and honoured the edit
            # would just split the difference, which is the behaviour being fixed.
            want_centre, want_extent = edits.get(obj.object_id, (None, None))
            if want_centre is None:
                want_centre = observed.centre
            if want_extent is None:
                want_extent = observed.extent
            out += list(weights.depth * (centre - want_centre))
            out += list(weights.depth * (extent - want_extent))

        # Support: the gap to whatever this rests on, floor or object.
        support_top = graph.floor_height_m
        if obj.supported_by and obj.supported_by in boxes:
            parent = graph.get(obj.supported_by)
            # The height of the parent's real collision surface under this object's
            # footprint. The box top is the fallback, not the answer: it is the top
            # of a sofa's *backrest* rather than its seat, and it comes from stored
            # `dims_m` rather than from the mesh — 8.5 mm out on a reconstructed rug.
            # See `app.pipeline.support`.
            # The child's own base is passed so the surface picked is the one it
            # stands on rather than the parent's highest — see `support.under`.
            child_base = supports.base_of(obj)
            measured = (
                supports.under(
                    parent,
                    low,
                    high,
                    base=child_base if child_base is not None else float(low[2]),
                    slack=ctx_slack,
                )
                if parent is not None
                else None
            )
            support_top = measured if measured is not None else float(boxes[obj.supported_by][1][2])
        # The child's underside, measured the same way. Both sides have to come from
        # the same geometry or the residual closes onto something that is not the
        # contact: with a measured surface but a stored box-bottom, a table settled
        # 2.9 mm clear of the rug it was supposedly resting on, which is exactly the
        # amount its `dims_m` box disagrees with its collision mesh.
        measured_base = supports.base_of(obj)
        base = measured_base if measured_base is not None else float(low[2])
        # One query for the whole contact — the minimum clearance over this object's
        # own underside — rather than a lowest-point-anywhere minus a
        # highest-surface-anywhere. For a solid box those agree; for a pedestal they
        # are two different places, and the side table's 9.2 mm "gap" was one low
        # foot compared against a rug sampled under the empty air beneath its top.
        measured_gap = supports.gap_to(
            graph.get(obj.supported_by) if obj.supported_by else None,
            obj,
            graph.floor_height_m,
        )
        gap = measured_gap if measured_gap is not None else base - support_top
        # Deliberately *not* dead-zoned, unlike penetration below. This term is not
        # only a penalty, it is the coupling: `E_supp` is what carries one object's
        # scale to the next, which is the claim the whole per-object formulation
        # rests on. Silencing it inside the tolerance silences it almost everywhere,
        # and a scale anchor on a table then stops reaching the mug on it — measured,
        # the mug ended 819 mm off its own support. A term that transmits a
        # constraint has to stay linear; only a term that merely forbids something
        # can go quiet once it is satisfied.
        out.append(weights.support * gap)

        # Containment: the lateral counterpart of the term above. `support` closes
        # the vertical gap and is indifferent to *where* the contact is, so without
        # this an object satisfies it just as well hovering the correct 0 mm over
        # empty air a metre to the left of the table it is recorded as resting on.
        #
        # Aimed at the nearest point that actually has surface under it, not at the
        # parent's bounding box. The box version had to ship disabled: a bookshelf's
        # AABB is mostly solid shelf, so pulling a vase's centre "inside" it drove
        # the vase into the carcass — 34 mm of burial at the weight that fixed the
        # placement. Now that the grid keeps every surface in a column, "where could
        # this rest" is answerable directly and the pull goes somewhere the object
        # can stand.
        #
        # Silent once the centre is over surface, so a correctly placed object pays
        # nothing and this cannot recentre a scene that was already right. Guarded on
        # the weight rather than multiplied by it: a zero weight has to add no
        # residuals at all, because `least_squares` reaches a different local minimum
        # when the residual vector merely changes length.
        if weights.containment and obj.supported_by and obj.supported_by in boxes:
            parent = graph.get(obj.supported_by)
            over = (
                None
                if parent is None
                else supports.under(
                    parent, low, high, centre_only=True, base=base, slack=supports.burial_slack_m
                )
            )
            # Always two residuals, zero when satisfied, never absent. `least_squares`
            # needs a fixed-length residual vector, and this term switches off exactly
            # when the object crosses onto its support — emitting it conditionally
            # changed the vector from 17 entries to 15 mid-solve.
            offset = np.zeros(2)
            if parent is not None and over is None:
                target = supports.nearest_support_xy(
                    parent, low, high, base, supports.burial_slack_m
                )
                if target is not None:
                    offset = centre[:2] - target
            out += list(weights.containment * offset)

        if settled is not None and obj.object_id in settled:
            out += list(PHYSICS_WEIGHT * (np.asarray(obj.position_m) - settled[obj.object_id]))

    # Penetration, over pairs. One-sided: touching is fine, overlapping is not.
    #
    # Measured against the convex decomposition, not the bounding boxes. A box is
    # a solid brick from an object's feet to its highest point, so a rug lying
    # under a sofa reads as buried inside it — on `room.png` that was a phantom
    # 14.5 mm the solver spent every iteration trying to resolve while both the
    # decomposition and MuJoCo reported the pair as not touching at all. See
    # `app.pipeline.penetration`.
    for first, second in penetrations.pairs:
        depth = penetrations.between(graph.get(first), graph.get(second))
        out.append(weights.penetration * max(0.0, depth - penetrations.tolerance))

    # Support contacts, scored only past the tolerance. `penetration.pairs` drops
    # them because the support term is already driving those two surfaces together
    # and penalising the same contact twice has the solver fighting itself — true of
    # a *resting* contact, and the reason this is one-sided rather than a pair added
    # back to the list above. It is not true of an object buried in its own support,
    # which nothing was scoring at all: `E_supp` measures the vertical gap under the
    # footprint, which is a different quantity from mesh overlap and can read as
    # satisfied while the meshes are centimetres into each other. Measured on
    # `room2.png`, a 0.77 kg vase sat 15.8 mm inside the bookshelf it rests on, and
    # the solver had no term that could see it — `certify.stability` did, and MuJoCo
    # ejected it at 15.9 m/s and diverged after 0.446 s.
    for child, parent in penetrations.support_pairs:
        depth = penetrations.between(graph.get(child), graph.get(parent))
        out.append(weights.penetration * max(0.0, depth - penetrations.tolerance))

    # An anchor is E_prior with sigma -> 0: a hard pull on one axis.
    for anchor in anchors:
        obj = graph.get(anchor.object_id)
        if obj is None:
            continue
        low, high = boxes[anchor.object_id]
        out.append(100.0 * float((high - low)[anchor.axis] - anchor.value_m))

    return np.asarray(out, dtype=float)


def _edit_priors(
    graph: SceneGraph,
) -> dict[str, tuple[np.ndarray | None, np.ndarray | None]]:
    """Where the *user* says each object is, for the objects they have moved.

    **Depth measures the scene once, at reconstruction. After that the person
    looking at it is the better witness to where a thing goes.** The stored
    `DepthObservation` is a fact about the photo and stays one — nothing here
    rewrites it — but it stops being the target the depth term aims at for a
    property the user has since set by hand.

    Without this a drag is not merely outvoted, it is erased. Measured on `room2`,
    dragging the lamp and re-solving kept **0%** of a 0.5 m move and 0% of a 1.0 m
    move; a 2.0 m move survived only 81%, and only because `_POSITION_RANGE_M`
    stops the solver reaching back far enough to finish the job. The cause is
    arithmetic rather than tuning: depth contributes six residuals per object, three
    of them the centre, against one for support, so the stale centre wins whatever
    `weights.depth` is set to.

    Per property, not per object, because the two measurements are independent. A
    drag says nothing about size, so `extent` keeps coming from depth; a typed scale
    says nothing about position, so `centre` does. Only the half the user actually
    set is replaced.

    The target is the object's world AABB at solve entry, which makes the residual
    exactly zero to begin with. It grows only as support, penetration or settling
    push the object off the mark — so the edit holds unless physics says it cannot,
    which is the whole intent. Note this is a *soft* prior at `weights.depth`, not a
    pin: a drop that would leave an object floating or buried still gets corrected.
    `ScaleAnchor` remains the hard constraint, for a typed dimension the user
    genuinely knows.
    """
    priors: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {}
    for obj in graph.objects:
        moved = obj.provenance.get("position_m") == Provenance.USER
        resized = obj.provenance.get("scale") == Provenance.USER
        if not moved and not resized:
            continue
        low, high = world_aabb(obj)
        priors[obj.object_id] = (
            (low + high) / 2.0 if moved else None,
            high - low if resized else None,
        )
    return priors


def _settle(ctx: PipelineContext, graph: SceneGraph) -> dict[str, np.ndarray]:
    """Where each object ends up after gravity, as a target to pull toward."""
    import mujoco

    from app.export import mjcf

    model = mjcf.load_model(graph)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    for _ in range(round(ctx.settings.settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    settled = {}
    for obj in graph.objects:
        body = model.body(mjcf.body_name(obj.object_id, obj.root_part.part_id)).id
        # The body frame moves with the free joint, so its world position is the
        # settled equivalent of `SceneObject.position_m`.
        settled[obj.object_id] = np.asarray(data.xpos[body], dtype=float)
    return settled


def _measure(
    ctx: PipelineContext,
    graph: SceneGraph,
    depth: DepthResult | None,
    segments: SegmentResult | None,
) -> dict:
    """The depth measurement per object, from the depth map or from the graph.

    First time through the pipeline both are supplied and this reads the depth map.
    A re-solve — a user committing an edit, say — has neither: the `.npy` and every
    mask PNG would have to still be on disk under the key that produced them, which
    is a lot to require of a scene the client is looking at right now. So the six
    numbers stage 6 actually reads are carried on the graph, and this prefers the
    map when it is there and falls back to them when it is not.
    """
    from app.pipeline.reconcile import Observation, observe

    if depth is None or segments is None:
        return {
            obj.object_id: Observation(
                centre=np.asarray(obj.observation.centre_m, dtype=float),
                extent=np.asarray(obj.observation.extent_m, dtype=float),
                footprint_xy=np.empty((0, 2)),
            )
            for obj in graph.objects
            if obj.observation is not None
        }

    observations, _, _ = observe(ctx, segments, depth)
    # `observe` reports in camera coordinates; reconcile recentred the graph out of
    # them. Without this the residual would be the frame offset rather than the fit
    # error, and the solver would happily drag every object back to where the
    # photographer was standing.
    offset = np.asarray(graph.world_offset_m)
    if offset.any():
        observations = {
            key: value._replace(centre=value.centre + offset) for key, value in observations.items()
        }
    return observations


def run(
    ctx: PipelineContext,
    graph: SceneGraph,
    depth: DepthResult | None,
    segments: SegmentResult | None,
    weights: SolveWeights,
    anchors: list[ScaleAnchor] | None = None,
    free_ids: set[str] | None = None,
) -> SolveResult:
    """Solve the scene, starting from wherever it currently is.

    **The graph's current pose is the initial guess, not a constraint.** That is
    what makes committing a user edit work without any pinning machinery: move an
    object, re-run, and the solver refines from where you put it rather than from
    where reconstruction did. `_bounds` allows a metre either way of the starting
    position, so an edit also shifts the region searched.

    **Seeding alone turned out not to be enough, and `_edit_priors` is the rest of
    it.** Starting from the user's pose does not help when a term still points at
    the old one: the stale depth centre pulled a dragged lamp all the way home, so
    the seed decided only which side of the answer the solver approached from.
    Where the user has set a property by hand, the depth term now aims at what they
    set rather than at what reconstruction measured — still softly, at
    `weights.depth`, so support and penetration can overrule an impossible drop.
    `ScaleAnchor` remains the only hard constraint, for a dimension the user has
    actually measured.
    """
    anchors = anchors or []
    working = graph.model_copy(deep=True)
    if not working.objects:
        # Trivially both: there is nothing to move and nothing to settle. Reporting
        # `settled=False` here would say an empty scene is physically unresolved.
        return SolveResult(
            graph=working, diagnostics=SolveDiagnostics(converged=True, settled=True)
        )

    observations = _measure(ctx, working, depth, segments)
    # Carried on the graph so a later re-solve needs neither the depth map nor the
    # masks; see `DepthObservation`.
    for obj in working.objects:
        found = observations.get(obj.object_id)
        if found is not None:
            obj.observation = DepthObservation(
                centre_m=tuple(float(v) for v in found.centre),
                extent_m=tuple(float(v) for v in found.extent),
            )
    # Which objects the solve may move. `None` is the whole scene, which is what a
    # pipeline run wants. An *edit* wants only what the user touched and whatever
    # rests on it: a full re-solve for a one-object drag moves objects nobody
    # touched — measured on `room2.png`, a commit took the scale axis from 6
    # failures to 9, and with 6 of 16 objects sitting within 4 mm of the tolerance
    # line, that reads as objects flickering between certified and not for no
    # visible reason. The edit endpoint already declines to run `repair` for exactly
    # this reason; re-solving everything was doing the same thing less visibly.
    #
    # The trade is real: freezing the rest gives up the coupling that propagates
    # scale between objects, which is the whole point of the formulation. That is
    # right for a drag, whose intent is local, and wrong for a pipeline run.
    free = [obj for obj in working.objects if free_ids is None or obj.object_id in free_ids]
    if not free:
        return SolveResult(
            graph=working, diagnostics=SolveDiagnostics(converged=True, settled=True)
        )

    lower, upper = _bounds(free, observations)
    # Read once, before anything moves: these are targets, and re-reading them from a
    # partly-solved graph would make each round chase the last one's answer.
    edits = _edit_priors(working)

    settled: dict[str, np.ndarray] | None = None
    total_iterations = 0
    converged = False
    is_settled = False
    max_drift = 0.0
    result = None

    for round_index in range(MAX_ROUNDS):
        # Rebuilt per round rather than once: a grid is exact under the parent's
        # translation and scale, both of which it divides out, but is built from the
        # parent's *collision meshes at their current pose*, so a round that moved a
        # parent's parts wants a fresh one. Cheap — a few thousand rays per support.
        supports = support.build(working, ctx.settings)
        penetrations = _Penetrations(working, ctx.settings)
        # Clipped, because a round trip through the bounds can land just outside
        # them. `_bounds` freezes an unobserved object's scale to an interval 1e-12
        # wide, `_unpack` stores exp(x) and `_pack` reads log(scale) back, and that
        # is not exactly the identity: measured, a frozen log-scale bounded at 1e-12
        # came back as 1.0000889e-12 and `least_squares` refused the round with
        # "Initial guess is outside of provided bounds". Clipping is right rather
        # than merely quiet — the value is inside the interval to within float
        # precision, and the alternative is failing a solve over one ulp.
        before = np.clip(_pack(free), lower, upper)
        result = optimize.least_squares(
            _residuals,
            before,
            bounds=(lower, upper),
            args=(
                working,
                free,
                observations,
                weights,
                anchors,
                settled,
                supports,
                penetrations,
                edits,
            ),
            # Finite differences: the residual runs a physics-free geometry pass, so
            # an analytic Jacobian would be a second implementation of it to keep in
            # step for no gain at this problem size.
            jac="2-point",
            max_nfev=200 * len(before),
        )
        _unpack(free, result.x)
        total_iterations += int(result.nfev)

        moved = float(np.max(np.abs(result.x - before)))
        # Physics feedback for the next round: settle and see what slid.
        settled = _settle(ctx, working)
        drift = max(
            (
                float(np.linalg.norm(np.asarray(o.position_m) - settled[o.object_id]))
                for o in working.objects
                if o.object_id in settled
            ),
            default=0.0,
        )
        log.info(
            "round %d: cost %.4f, moved %.4f m, settling drift %.4f m",
            round_index + 1,
            float(result.cost),
            moved,
            drift,
        )
        converged = moved < ROUND_TOLERANCE_M
        is_settled = drift < ROUND_TOLERANCE_M
        max_drift = drift
        # Both, because the loop has nothing left to do only when the optimiser has
        # stopped moving *and* physics agrees with where it stopped. Reported apart
        # so a scene held back by one unstable object is not mistaken for a failed
        # solve; see `SolveDiagnostics`.
        if converged and is_settled:
            break

    # Only what this solve actually estimated, and never over a `USER` mark.
    #
    # Both restrictions matter. Stamping a frozen object claims an estimate that was
    # never made — it has no Jacobian column and no variance to report. Stamping over
    # `USER` is worse: `_edit_priors` reads that flag, so the blanket version erased
    # its own input, and a second commit would have gone back to being outvoted by
    # depth with nothing in the graph left to say why.
    free_set = {obj.object_id for obj in free}
    for obj in working.objects:
        if obj.object_id not in free_set:
            continue
        for field in ("scale", "position_m"):
            if obj.provenance.get(field) != Provenance.USER:
                obj.provenance[field] = Provenance.DERIVED

    return SolveResult(
        graph=working,
        diagnostics=SolveDiagnostics(
            iterations=total_iterations,
            converged=converged,
            settled=is_settled,
            max_settle_drift_m=max_drift,
            residual_by_term=_term_costs(
                working,
                working.objects,
                observations,
                weights,
                anchors,
                settled,
                supports,
                penetrations,
                edits,
            ),
            scale_variance=_scale_variance(free, result),
        ),
    )


def _term_costs(
    graph, free, observations, weights, anchors, settled, supports, penetrations, edits
) -> dict[str, float]:
    """Each term's share of the final cost, for seeing which one is binding."""
    x = _pack(graph.objects)
    full = float(
        np.sum(
            _residuals(
                x,
                graph,
                graph.objects,
                observations,
                weights,
                anchors,
                settled,
                supports,
                penetrations,
                edits,
            )
            ** 2
        )
    )
    bare = SolveWeights(depth=0.0, support=0.0, penetration=0.0, containment=0.0)
    return {
        "total": full,
        "depth": full
        - float(
            np.sum(
                _residuals(
                    x,
                    graph,
                    graph.objects,
                    observations,
                    bare.model_copy(
                        update={
                            "support": weights.support,
                            "penetration": weights.penetration,
                            "containment": weights.containment,
                        }
                    ),
                    anchors,
                    settled,
                    supports,
                    penetrations,
                    edits,
                )
                ** 2
            )
        ),
    }


def _scale_variance(free: list[SceneObject], result) -> dict[str, float]:
    """Inverse-Hessian diagonal for the scale variables.

    Free from choosing Gauss-Newton: `J^T J` approximates the Hessian, and the
    diagonal of its inverse is the per-variable variance. This is what tells you
    *which* objects an anchor never reached — the ones whose scale the data barely
    constrains — and it is v2's primary confidence signal.
    """
    if result is None or result.jac is None:
        return {}
    try:
        hessian = result.jac.T @ result.jac
        covariance = np.linalg.pinv(hessian)
    except np.linalg.LinAlgError:
        return {}
    # Indexed against the objects the solve was free to move, which is what the
    # Jacobian's columns are. Keyed on those only: an object held fixed has no
    # column and so no variance to report, which is honest — nothing was estimated.
    return {
        obj.object_id: float(abs(covariance[index * 4, index * 4]))
        for index, obj in enumerate(free)
    }
