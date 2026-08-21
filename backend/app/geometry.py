"""World-space geometry over the scene graph.

Shared rather than per-consumer because three things want the same answers and
would otherwise each grow their own slightly-different copy: the scale
certification axis needs world bounds and support gaps, stage 6's `E_supp` needs
the surface an object should be resting on, and scene variation needs the set of
surfaces an object could plausibly be moved to.

Everything here is pure — the scene graph in, numbers out. No MuJoCo, no meshes
loaded from disk, no I/O.
"""

import math

import numpy as np

from app.schemas import PartGeometry, SceneGraph, SceneObject, SupportSurface

__all__ = [
    "candidate_hosts",
    "footprint_xy",
    "part_corners",
    "quat_to_matrix",
    "support_surfaces",
    "world_aabb",
]

# Local box faces as (axis, sign). +Z is index 2, +1.
_FACES = [(axis, sign) for axis in range(3) for sign in (1, -1)]


def quat_to_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """MuJoCo quaternion order: w, x, y, z."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def part_corners(obj: SceneObject, part: PartGeometry) -> np.ndarray:
    """The part's eight box corners in world space, as an 8x3 array.

    Ordered by the sign pattern (-,-,-), (-,-,+), (-,+,-), ... so a face can be
    selected by filtering on one coordinate's sign.
    """
    rotation = quat_to_matrix(obj.orientation)
    origin = np.asarray(obj.position_m, dtype=float)
    centre = np.asarray(part.origin_m, dtype=float) * obj.scale
    half = np.asarray(part.dims_m, dtype=float) * obj.scale / 2.0

    corners = np.empty((8, 3))
    for index, signs in enumerate(np.ndindex(2, 2, 2)):
        offset = centre + half * (np.asarray(signs) * 2 - 1)
        corners[index] = origin + rotation @ offset
    return corners


def world_aabb(obj: SceneObject) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned world bounds over every part.

    All eight corners of each part are transformed rather than the extent being
    scaled directly, because a rotated box has a larger axis-aligned footprint
    than its own dimensions — and a support gap computed from the unrotated
    extent would silently report a tilted object as floating.
    """
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for part in obj.parts:
        corners = part_corners(obj, part)
        low = np.minimum(low, corners.min(axis=0))
        high = np.maximum(high, corners.max(axis=0))
    return low, high


def footprint_xy(obj: SceneObject) -> tuple[float, float]:
    """The object's axis-aligned width and depth in world space."""
    low, high = world_aabb(obj)
    return (float(high[0] - low[0]), float(high[1] - low[1]))


def _face_corners(corners: np.ndarray, axis: int, sign: int) -> np.ndarray:
    """The four corners on one face of the box.

    `part_corners` orders by sign pattern, so the face is the four entries whose
    bit for `axis` matches `sign`.
    """
    want = 1 if sign > 0 else 0
    keep = [i for i in range(8) if ((i >> (2 - axis)) & 1) == want]
    return corners[keep]


def _polygon_area_xy(points: np.ndarray) -> float:
    """Shoelace over the XY projection, with the four corners put in ring order.

    `_face_corners` returns them in bit order, which for a rectangle is
    (a, b, c, d) with c and d swapped relative to a traversal — hence the
    reordering rather than a straight shoelace.
    """
    ring = points[[0, 1, 3, 2]][:, :2]
    x, y = ring[:, 0], ring[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _xy_overlap(a_low, a_high, b_low, b_high) -> float:
    """Area of the XY intersection of two axis-aligned boxes."""
    width = min(a_high[0], b_high[0]) - max(a_low[0], b_low[0])
    depth = min(a_high[1], b_high[1]) - max(a_low[1], b_low[1])
    return float(max(0.0, width) * max(0.0, depth))


def _dependents(graph: SceneGraph, object_id: str) -> set[str]:
    """Everything resting on this object, directly or transitively."""
    found: set[str] = set()
    frontier = [object_id]
    while frontier:
        current = frontier.pop()
        for obj in graph.objects:
            if obj.supported_by == current and obj.object_id not in found:
                found.add(obj.object_id)
                frontier.append(obj.object_id)
    return found


def support_surfaces(
    graph: SceneGraph,
    object_id: str,
    max_tilt_deg: float = 10.0,
    min_area_m2: float = 0.0025,
) -> list[SupportSurface]:
    """Upward-facing faces of one object that something could rest on.

    A face qualifies when its world normal is within `max_tilt_deg` of straight up.
    That tolerance is not cosmetic: reconstruction rarely returns a perfectly level
    tabletop, and requiring an exact +Z normal would find no surfaces at all on
    real output. Too generous and a sloped surface gets offered as a shelf.

    Only box faces are handled. Every part currently falls back to an OBB, so this
    is complete for the geometry the pipeline produces today; a mesh tier would
    cluster near-horizontal faces by height and plug in here.

    `free_area_m2` subtracts whatever already rests on the object, using
    axis-aligned footprints. That over-subtracts for a rotated occupant, which is
    the safe direction — it under-reports available space rather than offering a
    surface that turns out to be full.
    """
    obj = graph.get(object_id)
    if obj is None:
        return []

    cos_limit = math.cos(math.radians(max_tilt_deg))
    rotation = quat_to_matrix(obj.orientation)
    up = np.array([0.0, 0.0, 1.0])

    occupants = [
        world_aabb(other)
        for other in graph.objects
        if other.supported_by == object_id and other.object_id != object_id
    ]

    surfaces: list[SupportSurface] = []
    for part in obj.parts:
        corners = part_corners(obj, part)
        for axis, sign in _FACES:
            local_normal = np.zeros(3)
            local_normal[axis] = sign
            normal = rotation @ local_normal
            if float(np.dot(normal, up)) < cos_limit:
                continue

            face = _face_corners(corners, axis, sign)
            area = _polygon_area_xy(face)
            if area < min_area_m2:
                continue

            face_low = face.min(axis=0)
            face_high = face.max(axis=0)
            occupied = sum(_xy_overlap(face_low, face_high, low, high) for low, high in occupants)

            surfaces.append(
                SupportSurface(
                    object_id=object_id,
                    part_id=part.part_id,
                    # Centroid, because a face tilted within tolerance has no single
                    # height. The solver refines the contact anyway.
                    height_m=float(face[:, 2].mean()),
                    polygon_xy=[(float(p[0]), float(p[1])) for p in face[[0, 1, 3, 2]]],
                    normal=(float(normal[0]), float(normal[1]), float(normal[2])),
                    area_m2=area,
                    free_area_m2=max(0.0, area - occupied),
                )
            )

    surfaces.sort(key=lambda s: s.height_m, reverse=True)
    return surfaces


def candidate_hosts(
    graph: SceneGraph,
    object_id: str,
    max_tilt_deg: float = 10.0,
    clearance: float = 1.0,
) -> list[SupportSurface]:
    """Every surface in the scene that could geometrically hold this object.

    Geometry only — nothing here knows whether an apple *belongs* on a toilet
    cistern. The point is to hand a VLM a short list of placements that are at
    least possible, so its judgment is spent entirely on plausibility and it can
    never propose something that cannot physically work.

    Excludes the object's own surfaces and those of everything resting on it,
    transitively. Direct children are not enough: with a mug on a tray on a table,
    putting the table on the mug is a cycle two hops away, and a support graph with
    a cycle has no floor to settle against.
    """
    obj = graph.get(object_id)
    if obj is None:
        return []

    width, depth = footprint_xy(obj)
    needed = width * depth * clearance
    excluded = _dependents(graph, object_id) | {object_id}

    hosts: list[SupportSurface] = []
    for other in graph.objects:
        if other.object_id in excluded:
            continue
        for surface in support_surfaces(graph, other.object_id, max_tilt_deg):
            if surface.free_area_m2 < needed:
                continue
            # The surface also has to be wide enough in both directions, not just
            # large enough in total: a long thin shelf can have ample area and
            # still not take a wide object.
            span = np.ptp(np.asarray(surface.polygon_xy), axis=0)
            if span[0] < width or span[1] < depth:
                continue
            hosts.append(surface)

    hosts.sort(key=lambda s: s.height_m, reverse=True)
    return hosts
