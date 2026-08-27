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
    "matrix_to_quat",
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


def matrix_to_quat(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix to a w, x, y, z quaternion. Inverse of `quat_to_matrix`.

    Shepperd's method: pick the branch whose divisor is largest rather than always
    dividing by `w`. The naive form loses precision as `w -> 0` and divides by zero
    at exactly 180 degrees, which is not a corner case here — a principal-axis
    frame is whatever the eigenvectors say it is, and half-turns are common.

    The sign is chosen so `w >= 0`. A quaternion and its negation are the same
    rotation, so this is only a convention, but it is the one MuJoCo prints and it
    keeps a test that compares against a literal from failing on a valid answer.
    """
    m = np.asarray(m, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    else:
        axis = int(np.argmax(np.diag(m)))
        i, j, k = axis, (axis + 1) % 3, (axis + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        components = [0.0, 0.0, 0.0]
        components[i] = 0.25 * s
        components[j] = (m[j, i] + m[i, j]) / s
        components[k] = (m[k, i] + m[i, k]) / s
        q = ((m[k, j] - m[j, k]) / s, *components)

    if q[0] < 0.0:
        q = tuple(-v for v in q)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


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


def recentre(graph: SceneGraph) -> SceneGraph:
    """Put the scene on the standard origin: base-centred, floor on the ground plane.

    **The convention, stated once.** A scene's origin is the centre of its horizontal
    bounding box, at floor level: x and y centred on the objects' extent, z = 0 on the
    fitted floor. This is the same pivot convention a DCC tool gives you for a placeable
    asset — Blender's "origin to bounds centre" with the origin dropped to the base —
    and it is what makes a scene behave like an asset: it drops onto a floor, it rotates
    about its own middle, and two scenes can be compared without first hunting for them.

    **Centre rather than a corner.** The other candidate is the minimum corner of the
    bounding box, which gives all-positive coordinates and is what tile and level-block
    workflows use. Centre wins here on sensitivity: it is `(min + max) / 2`, so it
    responds to each extreme with weight one half, where a corner responds to one
    extreme with weight one and ignores the other entirely. When the object set changes
    — and it does, segmentation is not deterministic run to run — the corner therefore
    moves twice as far. Measured on room.jpg by dropping each object in turn: mean shift
    0.111 m for the centre against 0.138 m for the corner, worst case 0.54 m against
    1.08 m, the worst case being exactly the factor of two.

    A corner is also not the same corner twice. Which object is at minimum x and y
    depends on the scene's yaw, and yaw comes from the camera — so two photographs of
    one room from different positions put the origin at different physical corners,
    which is precisely the comparison this convention exists to make possible.

    Rotation does not distinguish them, and it is worth saying so because it sounds
    like it should. Whatever the reference point, this function puts it on the origin,
    so a rotation followed by recentring cancels the pivot entirely — both conventions
    end up at the origin by construction.

    **Why a convention is needed at all**, when importers normally preserve authored
    coordinates and never recentre. Because there are no authored coordinates here.
    Reconstruction happens in camera coordinates: depth backprojection puts the origin
    at the camera's optical centre, so a scene sits wherever the photographer happened
    to stand — measured at 2.4 m along +Y on room.jpg, and a different offset for every
    photo. That is not an origin anyone chose; it is an artefact of how the scene was
    measured. Preserving it would be preserving noise.

    Applied to the graph rather than to the exported glTF's root node, which is where
    this lived first. The root-node version centred the picture and left the numbers
    alone, which is fine for looking and wrong for editing: a transform gizmo reads a
    world position off the rendered object and writes it back to `position_m`, so a
    hidden offset between the two would move an object by the offset on every drag.

    Horizontal only, and the floor is left where the plane fit put it — z is shared
    with MJCF's ground plane and the viewer's grid, so shifting it would put objects
    above a floor that does not move.

    The camera pose travels with the scene so it still points at what it pointed at, and
    the offset accumulates onto `world_offset_m` — accumulated, not assigned, so calling
    this twice is a no-op rather than a second shift. `solve` reads it to get back to
    the camera coordinates its depth observations still live in.

    Objects keep their positions relative to each other; this is a change of frame, not
    a change of scene.

    Note what this deliberately does *not* try to fix: a scene centred this way can
    still sit off-centre on screen, because a 3/4 view projects along a diagonal and an
    L-shaped room is not centred on it. That is a framing problem and it belongs to the
    camera, which is where every editor puts it. See `frontend/src/scene/aim.ts`.
    """
    if not graph.objects:
        return graph

    corners = np.vstack([np.vstack(world_aabb(obj)) for obj in graph.objects])
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2.0
    shift = np.array([-centre[0], -centre[1], 0.0])
    if np.allclose(shift, 0.0):
        return graph

    objects = [
        obj.model_copy(
            update={"position_m": tuple(float(v) for v in np.asarray(obj.position_m) + shift)}
        )
        for obj in graph.objects
    ]
    camera = graph.camera
    if camera is not None:
        camera = camera.model_copy(
            update={"position_m": tuple(float(v) for v in np.asarray(camera.position_m) + shift)}
        )
    return graph.model_copy(
        update={
            "objects": objects,
            "camera": camera,
            "world_offset_m": tuple(float(v) for v in np.asarray(graph.world_offset_m) + shift),
        }
    )
