"""Scale certification: are the fitted sizes plausible, and do things rest on things?

Three checks, all computed from the scene graph alone. No depth map, no MuJoCo,
no I/O — which makes this the cheapest axis and the only one that can run on a
graph with no geometry files at all.

**Depth is deliberately absent.** It is an *input* to the thing being checked: the
solver minimises E_depth, so a certifier that also scored against depth would let
a solve which overfit it certify beautifully while being wrong.

Being straight about what this is, though: every check here corresponds to a term
the solver was already minimising, so it is not validation against outside truth.
It is a **feasibility check** — did the optimiser actually achieve the constraints
it was given, or did it trade one away to satisfy another? An object that ended up
four sigma from its class prior because depth pulled hard is a genuine warning,
and that is what this catches. Independent validation needs a tape measure, and
that is evaluation rather than certification.
"""

import numpy as np

from app.geometry import quat_to_matrix, world_aabb
from app.pipeline.support import SupportHeights
from app.schemas import ScaleCheck, SceneGraph, SceneObject, Vec3

__all__ = ["gap_ok", "gap_violation", "run"]


def gap_ok(gap: float, max_support_gap_m: float, max_penetration_m: float) -> bool:
    """Whether a signed support gap is acceptable. Two bounds, not one.

    Floating and buried are different failures measured on the same number, and the
    certificate publishes a different tolerance for each: clearance is graded against
    `max_support_gap_m` (5 mm), which exists because a mesh contact is never exactly
    flush, and burial is penetration, which the stability axis grades against
    `max_penetration_m` (2 mm). A single `abs(gap) <= max_support_gap_m` therefore
    hands out 5 mm of overlap that the next axis fails, and two axes disagree about
    one contact. Measured on `room2.png`: a bookshelf 4.99 mm into the floor and a rug
    2.11 mm into it both passed here and failed `certify.stability`, and because
    `repair._violation` scored them zero on this axis and `_resolve_penetration`
    cannot push against a floor that is not in `graph.objects`, nothing in repair
    could see them either.

    Shared with `repair` so the criterion and the repair trigger cannot drift apart.
    """
    return -max_penetration_m <= gap <= max_support_gap_m


def gap_violation(gap: float, max_support_gap_m: float, max_penetration_m: float) -> float:
    """How far past its bound the gap sits, in units of that bound. Zero inside.

    The scoring counterpart of `gap_ok`, normalised the way `repair._violation`
    normalises everything else, and asymmetric for the same reason.
    """
    limit = max_support_gap_m if gap >= 0.0 else max_penetration_m
    return max(0.0, abs(gap) / limit - 1.0)


def _weight_centre_xy(obj: SceneObject, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """World XY of the object's centre of mass, or its box centre without one.

    The toppling condition is about *weight*, and the bounding-box centre is not
    where the weight is. Measured on the two rooms, it gets the answer wrong in both
    directions: `room.png`'s lamp has its box centre outside its contact polygon and
    its centre of mass inside — it settles 0.1 mm and drifts 0.1 deg, so it is
    resting and the box says toppling — while `room2.png`'s armchair has its box
    centre inside and its centre of mass outside, and it settles 23.7 mm and fails
    the stability axis. A shade overhanging a base or a backrest behind four legs is
    the ordinary case, not a corner one.

    Composed exactly as `export.mjcf` composes it, so this axis and the simulation
    agree about where the mass is: the part origin is scaled, `com_m` is an offset
    within the part frame. Checked against `data.subtree_com` on both rooms, worst
    disagreement 5.4 micrometres.

    Falls back to the box centre for an object with no inertial properties, which is
    stage 9 not having run — the hand-authored fixtures live there.
    """
    rotation = quat_to_matrix(obj.orientation)
    origin = np.asarray(obj.position_m, dtype=float)
    total, moment = 0.0, np.zeros(3)
    for part in obj.parts:
        inertial = part.inertial
        if inertial is None or inertial.mass_kg <= 0.0:
            continue
        local = np.asarray(part.origin_m, dtype=float) * obj.scale + np.asarray(
            inertial.com_m, dtype=float
        )
        moment += inertial.mass_kg * (origin + rotation @ local)
        total += inertial.mass_kg
    if total <= 0.0:
        return (low[:2] + high[:2]) / 2.0
    return (moment / total)[:2]


def _deviation_sigma(fitted: Vec3, prior_dims: Vec3, prior_sigma: Vec3) -> Vec3:
    """(fitted - prior) / sigma, per axis.

    A sigma of zero is a hard constraint — a scale anchor, or a prior the VLM
    claimed certainty about — so any mismatch has to read as a large deviation
    rather than a division by zero.
    """
    out = []
    for f, p, s in zip(fitted, prior_dims, prior_sigma, strict=True):
        out.append((f - p) / max(s, 1e-9))
    return (out[0], out[1], out[2])


def run(
    graph: SceneGraph,
    max_deviation_sigma: float,
    max_support_gap_m: float,
    max_penetration_m: float,
    supports: SupportHeights | None = None,
) -> list[ScaleCheck]:
    """Per-object scale and support checks.

    `supports` carries the measured surface heights stage 6 solved against. Passing
    it is what keeps the axis and the solver talking about the same contact: with
    the axis on stored `dims_m` boxes and the solver on collision meshes, a table
    the solver had correctly placed on a rug was reported 23.4 mm clear of it,
    because the two disagree about where both surfaces are. Optional so the pure
    fixtures still work without geometry on disk, and it degrades to the box.
    """
    bounds = {obj.object_id: world_aabb(obj) for obj in graph.objects}

    checks: list[ScaleCheck] = []
    for obj in graph.objects:
        prior = obj.label.prior
        # No prior means there is nothing to deviate from. Reporting zeros would
        # claim the object matched a prior it never had, which reads as a pass.
        # Zeros are correct here for a different reason: the deviation *component*
        # is vacuously satisfied, and the support checks below still apply.
        deviation = (
            _deviation_sigma(obj.dims_m, prior.dims_m, prior.sigma_m)
            if prior is not None
            else (0.0, 0.0, 0.0)
        )

        low, high = bounds[obj.object_id]
        parent = graph.get(obj.supported_by) if obj.supported_by else None

        if parent is None:
            # Supported by the floor, which is unbounded — so containment is
            # trivially true and only the gap is informative. An object naming a
            # support that is not in the graph lands here too; that is a
            # reconcile bug, and reporting it as a floor contact at least makes
            # the object's height visible rather than skipping it silently.
            support_top = graph.floor_height_m
            inside = True
        else:
            parent_low, parent_high = bounds[parent.object_id]
            support_top = float(parent_high[2])

        # The centre of *mass*, not of the bounding box: this is the toppling test.
        centre_xy = _weight_centre_xy(obj, low, high)

        # Geometry, asked without reference to any declared parent. An object rests
        # on whatever is beneath it, and often on more than one thing at once: an
        # armchair with three legs on a rug and one on the floor is ordinary and
        # stable, and so is a book overlapping another book by most of its face.
        # Measured against a single parent those read as defects — 6.2 mm against the
        # rug or 21 mm against the floor for the same unchanged chair, so the object
        # changed colour whenever the parent choice flipped two stages upstream.
        contact = (
            supports.contact(obj, graph.objects, graph.floor_height_m, max_support_gap_m)
            if supports is not None
            else None
        )
        # `supports` being present is not the same as it having anything to say. The
        # hand-authored fixtures carry no collision meshes, so `support.build` returns
        # an empty set of grids and undersides — and a contact query with no contact
        # points reports nothing to stand on, which read as every object toppling.
        if contact is not None and contact.gap is not None:
            gap = contact.gap
            resting_on = contact.resting_on
            # The toppling condition proper: is the weight over the contacts. A
            # bounding-box test only approximates this, and on a bookshelf — mostly
            # air and shelf edges — it approximated it wrongly, passing a vase with
            # no shelf under its centre at all.
            inside = contact.over_support(centre_xy) if parent is not None else True
            # And the semantic half, kept separate: is it on the thing it is recorded
            # as being on. A book that fell to the floor has a fine gap and is
            # perfectly stable, and is still not on its table.
            touching = (
                supports.touches(parent, obj, max_support_gap_m) if parent is not None else True
            )
        else:
            # No measurable geometry: fall back to the stored boxes, which is the
            # tier the fixtures live at and the degradation path for a scene whose
            # meshes went astray.
            gap = float(low[2]) - support_top
            resting_on = parent.object_id if parent is not None else None
            inside = (
                True
                if parent is None
                else bool(
                    np.all(centre_xy >= parent_low[:2]) and np.all(centre_xy <= parent_high[:2])
                )
            )
            touching = True

        checks.append(
            ScaleCheck(
                object_id=obj.object_id,
                deviation_sigma=deviation,
                support_gap_m=gap,
                base_inside_parent=inside,
                touching_parent=touching,
                resting_on=resting_on,
                passed=(
                    all(abs(d) <= max_deviation_sigma for d in deviation)
                    and gap_ok(gap, max_support_gap_m, max_penetration_m)
                    and inside
                    and touching
                ),
            )
        )
    return checks
