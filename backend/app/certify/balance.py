"""Static balance certification from contact-force feasibility.

This is intentionally a certificate axis before it is a solver term.  A photograph
of an ordinary indoor scene is evidence that its objects can be held at rest, but
the old physics residual answered a different question: where MuJoCo moved them
after release.  Here the observed pose itself is tested.

The contact representation currently records upward support samples in XY, not
arbitrary 3-D contact normals.  Under that honest scope, equilibrium under gravity
reduces to finding nonnegative normal forces whose resultant passes through the
centre of mass.  For contact points ``p_i`` and normal-force fractions ``n_i``:

    sum(n_i) = 1
    sum(n_i * (p_i.x - com.x)) = 0
    sum(n_i * (p_i.y - com.y)) = 0
    n_i >= 0

This small linear feasibility problem is the horizontal-contact form of contact-
wrench equilibrium.  Absolute mass cancels, so this axis makes no unsupported
claim to infer mass or friction.  The signed support margin is also reported:
positive inside the contact hull, negative outside, and zero on its boundary.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from app.geometry import world_aabb, world_center_of_mass
from app.pipeline.support import SupportHeights
from app.schemas import BalanceCheck, SceneGraph, SceneObject

__all__ = ["run"]

_NUMERICAL_MARGIN_M = 1e-9


@dataclass(frozen=True)
class _ContactEvidence:
    points_m: np.ndarray
    source: str
    resting_on: str | None


def _box_contact_points(graph: SceneGraph, obj: SceneObject, tolerance_m: float) -> np.ndarray:
    """OBB-tier fallback contact patch for scenes without collision meshes.

    Real reconstructed scenes use ``SupportHeights.contact``.  Hand-authored and
    degraded scenes still need a declared, auditable tier, so their patch is the
    XY intersection between child and declared support, or the child's footprint
    on the floor.
    """
    low, high = world_aabb(obj)
    if obj.supported_by:
        parent = graph.get(obj.supported_by)
        if parent is None:
            return np.empty((0, 3))
        parent_low, parent_high = world_aabb(parent)
        if abs(float(low[2] - parent_high[2])) > tolerance_m:
            return np.empty((0, 3))
        contact_low = np.maximum(low[:2], parent_low[:2])
        contact_high = np.minimum(high[:2], parent_high[:2])
        if np.any(contact_high <= contact_low):
            return np.empty((0, 3))
        contact_z = float(parent_high[2])
    else:
        if abs(float(low[2] - graph.floor_height_m)) > tolerance_m:
            return np.empty((0, 3))
        contact_low, contact_high = low[:2], high[:2]
        contact_z = float(graph.floor_height_m)

    return np.array(
        [
            [contact_low[0], contact_low[1], contact_z],
            [contact_low[0], contact_high[1], contact_z],
            [contact_high[0], contact_low[1], contact_z],
            [contact_high[0], contact_high[1], contact_z],
        ],
        dtype=float,
    )


def _contact_points(
    graph: SceneGraph,
    obj: SceneObject,
    supports: SupportHeights,
    tolerance_m: float,
) -> _ContactEvidence:
    contact = supports.contact(obj, graph.objects, graph.floor_height_m, tolerance_m)
    points = np.asarray(contact.contacts_m, dtype=float)
    source = "collision_proxy"
    # Fall back only when no collision underside exists.  A real mesh producing
    # one or two contact samples is a genuinely degenerate support, not permission
    # to replace it with a much larger bounding box and certify it anyway.
    if len(points) < 3 and supports.base_of(obj) is None:
        points = _box_contact_points(graph, obj, tolerance_m)
        source = "obb_fallback"
    points = np.empty((0, 3)) if not len(points) else np.unique(points.reshape(-1, 3), axis=0)
    return _ContactEvidence(points, source, contact.resting_on)


def _hull(points: np.ndarray) -> tuple[np.ndarray, ConvexHull] | None:
    if len(points) < 3:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    vertices = points[hull.vertices]
    return vertices, ConvexHull(vertices)


def _force_feasible(points: np.ndarray, centre_xy: np.ndarray) -> bool:
    """Whether nonnegative normal forces can balance gravity and its moment."""
    if len(points) < 3:
        return False
    offsets = points - centre_xy
    equalities = np.vstack([np.ones(len(points)), offsets[:, 0], offsets[:, 1]])
    result = linprog(
        c=np.zeros(len(points)),
        A_eq=equalities,
        b_eq=np.array([1.0, 0.0, 0.0]),
        bounds=(0.0, None),
        method="highs",
    )
    return bool(result.success)


def _support_margin(hull: ConvexHull, centre_xy: np.ndarray) -> float:
    """Signed distance to the nearest supporting line of the convex contact hull."""
    equations = hull.equations
    normals = equations[:, :2]
    offsets = equations[:, 2]
    distances = -(normals @ centre_xy + offsets) / np.linalg.norm(normals, axis=1)
    return float(np.min(distances))


def run(
    graph: SceneGraph,
    supports: SupportHeights,
    contact_tolerance_m: float,
) -> list[BalanceCheck]:
    checks: list[BalanceCheck] = []
    for obj in graph.objects:
        centre_m = world_center_of_mass(obj)
        centre_xy = centre_m[:2]
        evidence = _contact_points(graph, obj, supports, contact_tolerance_m)
        points = evidence.points_m[:, :2]
        hull_data = _hull(points)
        if hull_data is None:
            feasible = False
            margin = 0.0
            hull_points = points
        else:
            hull_points, hull = hull_data
            feasible = _force_feasible(hull_points, centre_xy)
            margin = _support_margin(hull, centre_xy)

        checks.append(
            BalanceCheck(
                object_id=obj.object_id,
                centre_of_mass_xy_m=(float(centre_xy[0]), float(centre_xy[1])),
                centre_of_mass_m=tuple(float(value) for value in centre_m),
                contact_count=len(points),
                contact_points_m=[
                    tuple(float(value) for value in point) for point in evidence.points_m
                ],
                support_polygon_xy_m=[(float(point[0]), float(point[1])) for point in hull_points],
                evidence_source=evidence.source,
                resting_on=evidence.resting_on,
                contact_force_feasible=feasible,
                support_margin_m=margin,
                passed=feasible and margin >= -_NUMERICAL_MARGIN_M,
            )
        )
    return checks
