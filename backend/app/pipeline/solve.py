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
    DepthResult,
    Provenance,
    ScaleAnchor,
    SceneGraph,
    SegmentResult,
    SolveDiagnostics,
    SolveResult,
    SolveWeights,
)

__all__ = ["SolveResult", "run"]

log = logging.getLogger(__name__)

# Outer block-coordinate rounds: smooth solve, then settle, then re-solve against
# what settling revealed. Two is usually enough; the loop exits early when a round
# stops changing anything.
MAX_ROUNDS = 4
ROUND_TOLERANCE_M = 2e-3

# Weight on the settled-pose residual. Deliberately below the depth weight: settling
# reveals that a placement is wrong, but the settled pose is not itself a
# measurement of where the object belongs — a mug knocked off a table lands on the
# floor, and the floor is not the answer.
PHYSICS_WEIGHT = 0.5


def _pack(graph: SceneGraph) -> np.ndarray:
    return np.concatenate([[np.log(obj.scale), *obj.position_m] for obj in graph.objects]).astype(
        float
    )


def _unpack(graph: SceneGraph, x: np.ndarray) -> None:
    for index, obj in enumerate(graph.objects):
        block = x[index * 4 : index * 4 + 4]
        obj.scale = float(np.exp(block[0]))
        obj.position_m = (float(block[1]), float(block[2]), float(block[3]))


def _bounds(graph: SceneGraph, observations) -> tuple[np.ndarray, np.ndarray]:
    """Keep the search near the measurement rather than unbounded.

    Scale is allowed a factor of four either way and position a metre. Without
    bounds the penetration term can push an object arbitrarily far to stop it
    overlapping, which trades a physics violation for a scale one and reports
    success.
    """
    low, high = [], []
    for obj in graph.objects:
        low += [np.log(obj.scale) - np.log(4.0), *(p - 1.0 for p in obj.position_m)]
        high += [np.log(obj.scale) + np.log(4.0), *(p + 1.0 for p in obj.position_m)]
    return np.asarray(low), np.asarray(high)


class _Penetrations:
    """The FCL objects plus the pair list, so neither is rebuilt per evaluation.

    The pairs depend only on the support graph, which does not change during a
    solve, and rebuilding them inside the residual would be the whole cost of the
    term.
    """

    def __init__(self, graph: SceneGraph, settings):
        self._depths = penetration.build(graph, settings)
        self.pairs = penetration.pairs(graph)

    def between(self, first, second) -> float:
        if first is None or second is None:
            return 0.0
        return self._depths.between(first, second)


def _residuals(
    x: np.ndarray,
    graph: SceneGraph,
    observations: dict,
    weights: SolveWeights,
    anchors: list[ScaleAnchor],
    settled: dict[str, np.ndarray] | None,
    supports: support.SupportHeights,
    penetrations: _Penetrations,
) -> np.ndarray:
    _unpack(graph, x)

    boxes = {obj.object_id: world_aabb(obj) for obj in graph.objects}
    out: list[float] = []

    for obj in graph.objects:
        low, high = boxes[obj.object_id]
        centre, extent = (low + high) / 2.0, high - low

        observed = observations.get(obj.object_id)
        if observed is not None:
            # Both halves matter: extent alone fixes size but lets the object drift,
            # centre alone fixes position but lets it grow without bound.
            out += list(weights.depth * (centre - observed.centre))
            out += list(weights.depth * (extent - observed.extent))

        # Support: the gap to whatever this rests on, floor or object.
        support_top = graph.floor_height_m
        if obj.supported_by and obj.supported_by in boxes:
            parent = graph.get(obj.supported_by)
            # The height of the parent's real collision surface under this object's
            # footprint. The box top is the fallback, not the answer: it is the top
            # of a sofa's *backrest* rather than its seat, and it comes from stored
            # `dims_m` rather than from the mesh — 8.5 mm out on a reconstructed rug.
            # See `app.pipeline.support`.
            measured = supports.under(parent, low, high) if parent is not None else None
            support_top = measured if measured is not None else float(boxes[obj.supported_by][1][2])
        # The child's underside, measured the same way. Both sides have to come from
        # the same geometry or the residual closes onto something that is not the
        # contact: with a measured surface but a stored box-bottom, a table settled
        # 2.9 mm clear of the rug it was supposedly resting on, which is exactly the
        # amount its `dims_m` box disagrees with its collision mesh.
        measured_base = supports.base_of(obj)
        base = measured_base if measured_base is not None else float(low[2])
        out.append(weights.support * (base - support_top))

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
        out.append(weights.penetration * depth)

    # An anchor is E_prior with sigma -> 0: a hard pull on one axis.
    for anchor in anchors:
        obj = graph.get(anchor.object_id)
        if obj is None:
            continue
        low, high = boxes[anchor.object_id]
        out.append(100.0 * float((high - low)[anchor.axis] - anchor.value_m))

    return np.asarray(out, dtype=float)


def _settle(ctx: PipelineContext, graph: SceneGraph) -> dict[str, np.ndarray]:
    """Where each object ends up after gravity, as a target to pull toward."""
    import mujoco

    from app.export import mjcf

    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
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


def run(
    ctx: PipelineContext,
    graph: SceneGraph,
    depth: DepthResult,
    segments: SegmentResult,
    weights: SolveWeights,
    anchors: list[ScaleAnchor] | None = None,
) -> SolveResult:
    from app.pipeline.reconcile import observe

    anchors = anchors or []
    working = graph.model_copy(deep=True)
    if not working.objects:
        return SolveResult(graph=working, diagnostics=SolveDiagnostics(converged=True))

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
    lower, upper = _bounds(working, observations)

    settled: dict[str, np.ndarray] | None = None
    total_iterations = 0
    converged = False
    result = None

    for round_index in range(MAX_ROUNDS):
        # Rebuilt per round rather than once: a grid is exact under the parent's
        # translation and scale, both of which it divides out, but is built from the
        # parent's *collision meshes at their current pose*, so a round that moved a
        # parent's parts wants a fresh one. Cheap — a few thousand rays per support.
        supports = support.build(working, ctx.settings)
        penetrations = _Penetrations(working, ctx.settings)
        before = _pack(working)
        result = optimize.least_squares(
            _residuals,
            before,
            bounds=(lower, upper),
            args=(working, observations, weights, anchors, settled, supports, penetrations),
            # Finite differences: the residual runs a physics-free geometry pass, so
            # an analytic Jacobian would be a second implementation of it to keep in
            # step for no gain at this problem size.
            jac="2-point",
            max_nfev=200 * len(before),
        )
        _unpack(working, result.x)
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
        if moved < ROUND_TOLERANCE_M and drift < ROUND_TOLERANCE_M:
            converged = True
            break

    for obj in working.objects:
        obj.provenance["scale"] = Provenance.DERIVED
        obj.provenance["position_m"] = Provenance.DERIVED

    return SolveResult(
        graph=working,
        diagnostics=SolveDiagnostics(
            iterations=total_iterations,
            converged=converged,
            residual_by_term=_term_costs(
                working, observations, weights, anchors, settled, supports, penetrations
            ),
            scale_variance=_scale_variance(working, result),
        ),
    )


def _term_costs(
    graph, observations, weights, anchors, settled, supports, penetrations
) -> dict[str, float]:
    """Each term's share of the final cost, for seeing which one is binding."""
    x = _pack(graph)
    full = float(
        np.sum(
            _residuals(x, graph, observations, weights, anchors, settled, supports, penetrations)
            ** 2
        )
    )
    bare = SolveWeights(depth=0.0, support=0.0, penetration=0.0)
    return {
        "total": full,
        "depth": full
        - float(
            np.sum(
                _residuals(
                    x,
                    graph,
                    observations,
                    bare.model_copy(
                        update={"support": weights.support, "penetration": weights.penetration}
                    ),
                    anchors,
                    settled,
                    supports,
                    penetrations,
                )
                ** 2
            )
        ),
    }


def _scale_variance(graph: SceneGraph, result) -> dict[str, float]:
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
    return {
        obj.object_id: float(abs(covariance[index * 4, index * 4]))
        for index, obj in enumerate(graph.objects)
    }
