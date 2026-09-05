"""Stage 5: merge reconstruction and depth into one gravity-aligned scene graph.

Unglamorous, and nothing works without it. Reconstruction returns meshes in their
own frame at their own scale; depth returns metres but no shape. Neither is a
scene. This puts them in one world frame with gravity along -Z.

**The division of labour is the point.** Position and size come from *depth*; shape
comes from the *mesh*. That is not arbitrary — measured on one room, the
backprojected cloud gives a sofa 2.09 m wide against SAM 3D's own estimate of about
1.0 m, and a floor lamp 1.81 m tall. Metric depth carries real size information and
a generative reconstruction's scale does not, which is the whole reason this project
fuses the two rather than trusting either.

`AssetFrame.normalization_scale` and `ObjectAssets.camera_translation` are therefore
kept but not used for placement. They are hypotheses to cross-check against, and on
this room they disagree with depth by a factor of two.

Three things it does **not** do yet, each a real gap rather than an oversight:

* **No interpenetration resolution.** Single-view depth sees the front shell only,
  so extent away from the camera is inferred. Overlaps are left for certification
  to find rather than pre-empted.
* **Support comes from the VLM**, which called the sofa floor-standing in one run
  and rug-supported in another. `app.geometry` can measure contact once objects are
  placed, and measurement should win.
"""

import logging
from typing import NamedTuple

import cv2
import numpy as np

from app.geometry import recentre, world_aabb
from app.pipeline.base import PipelineContext
from app.schemas import (
    CameraPose,
    DepthResult,
    LabelResult,
    Material,
    ObjectLabel,
    ReconstructionResult,
    SceneGraph,
    SceneObject,
    SegmentResult,
)

__all__ = ["Observation", "SceneGraph", "observe", "run"]

log = logging.getLogger(__name__)

# Mask edges straddle the depth discontinuity between an object and whatever is
# behind it, so the extreme values in a masked region belong to the background.
# Measured: the telephone's raw extent came out at 0.61 m against a real ~0.25 m.
# Clipping to this percentile band rejects the boundary without touching the object.
EXTENT_LOW, EXTENT_HIGH = 5.0, 95.0

# A mask smaller than this is mostly boundary, so its depth statistics are noise.
MIN_MASK_PX = 200

# Floor search: RANSAC over the lower part of the image, where the floor is if it
# is anywhere. The whole cloud would let a large wall win on inlier count.
FLOOR_IMAGE_FRACTION = 0.45
FLOOR_ITERATIONS = 200
FLOOR_INLIER_M = 0.03

# +90 degrees about X: takes the meshes' +Y up to the scene's +Z up. That the mesh
# frame is Y-up is determined from the geometry rather than assumed — a reconstructed
# rug is 1.00 x 0.007 x 1.00, so its thin axis is Y, and a rug lies flat.
#
# The sign matters and is easy to get backwards. R_x(t) sends (0,1,0) to (0,cos t,
# sin t), so +Y reaches +Z at t = +90 and -Z at t = -90. It was -90 here, which stood
# every object in the scene on its head; the error was invisible in the numbers
# because a bounding box is symmetric about its centre, and only showed up on screen.
_Y_UP_TO_Z_UP_HALF = np.pi / 4

# Below this the horizontal footprint is nearly square, so its principal axis is
# noise and rotating by it would spin the object at random.
MIN_YAW_ANISOTROPY = 1.15


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    """Yaw about world +Z, composed with the mesh's Y-up to Z-up correction.

    Quaternion product written out rather than pulled from a rotation library: both
    factors are axis-aligned, so the general form collapses to four terms and the
    ordering (yaw applied *after* the up-correction) stays visible.
    """
    cz, sz = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    cx, sx = np.cos(_Y_UP_TO_Z_UP_HALF), np.sin(_Y_UP_TO_Z_UP_HALF)
    return (float(cz * cx), float(cz * sx), float(sz * sx), float(sz * cx))


def _observed_yaw(points_xy: np.ndarray) -> float | None:
    """The dominant horizontal direction of an object's visible surface.

    Principal component of the masked points projected onto the floor plane. Single
    view means this is a shell rather than a solid, but the shell's widest
    horizontal spread is still the object's width axis, which is what yaw is
    measured against.

    Returns None when the footprint is close to circular: the principal axis of a
    round rug or a square stool is numerical noise, and rotating by it would spin
    the object at random for no gain.
    """
    if len(points_xy) < 3:
        return None
    centred = points_xy - points_xy.mean(axis=0)
    covariance = centred.T @ centred / len(centred)
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 1e-12 or np.sqrt(values[1] / values[0]) < MIN_YAW_ANISOTROPY:
        return None
    major = vectors[:, 1]
    return float(np.arctan2(major[1], major[0]))


def _backproject(depth: np.ndarray, intrinsics) -> np.ndarray:
    """Depth map to a camera-frame point cloud, shaped (H, W, 3).

    OpenCV convention: X right, Y **down**, Z forward.
    """
    height, width = depth.shape
    vs, us = np.mgrid[0:height, 0:width]
    x = (us - intrinsics.cx) * depth / intrinsics.fx
    y = (vs - intrinsics.cy) * depth / intrinsics.fy
    return np.stack([x, y, depth], axis=-1)


def _fit_floor(points: np.ndarray) -> tuple[np.ndarray, float]:
    """RANSAC a floor plane, returning its unit normal and offset.

    Restricted to the lower part of the image because a wall is also a plane and a
    big one will out-vote the floor on inlier count. Falls back to assuming the
    camera is level, which is wrong but recoverable, rather than to a plane fitted
    through furniture, which is not.
    """
    if len(points) < 3:
        return np.array([0.0, -1.0, 0.0]), 0.0

    rng = np.random.default_rng(0)  # deterministic: the same photo must reconcile alike
    best_normal, best_offset, best_inliers = None, 0.0, 0

    for _ in range(FLOOR_ITERATIONS):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        offset = float(normal @ sample[0])

        # The floor's normal is roughly the camera's up axis (-Y in OpenCV). This
        # rejects walls, which are the main competitor.
        if abs(normal[1]) < 0.7:
            continue

        inliers = int(np.sum(np.abs(points @ normal - offset) < FLOOR_INLIER_M))
        if inliers > best_inliers:
            best_normal, best_offset, best_inliers = normal, offset, inliers

    if best_normal is None:
        log.warning("no floor plane found; assuming the camera is level")
        return np.array([0.0, -1.0, 0.0]), float(np.percentile(points[:, 1], 95))

    # Point the normal up (-Y is up in OpenCV).
    if best_normal[1] > 0:
        best_normal, best_offset = -best_normal, -best_offset
    log.info("floor plane: %d inliers of %d", best_inliers, len(points))
    return best_normal, best_offset


def _world_basis(floor_normal: np.ndarray) -> np.ndarray:
    """A rotation taking camera coordinates to a Z-up world with gravity along -Z.

    Built from the floor normal so a tilted camera is corrected, rather than
    assuming the photographer held it level.
    """
    up = floor_normal / np.linalg.norm(floor_normal)
    # Camera forward (+Z), projected onto the floor plane, becomes world +Y.
    forward = np.array([0.0, 0.0, 1.0])
    forward = forward - (forward @ up) * up
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0]) - (np.array([1.0, 0.0, 0.0]) @ up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    return np.stack([right, forward, up])


class Observation(NamedTuple):
    """What depth says about one object, in world coordinates.

    Shared with stage 6 rather than recomputed there: the solver's `E_depth` term
    scores against exactly the measurement this stage placed the object from, and
    two implementations of "what depth says" drifting apart would make the solver
    optimise toward a target reconcile never used.
    """

    centre: np.ndarray
    extent: np.ndarray
    footprint_xy: np.ndarray


def observe(ctx: PipelineContext, segments: SegmentResult, depth: DepthResult):
    """Per-object metric observations from the depth map, plus the world frame.

    Returns `(observations, basis, floor_z)`. Separated from `run` so the solver can
    ask the same question without re-deriving the floor.
    """
    depth_map = np.load(depth.depth_path)
    cloud = _backproject(depth_map, depth.intrinsics)

    height = depth_map.shape[0]
    lower = cloud[int(height * (1.0 - FLOOR_IMAGE_FRACTION)) :].reshape(-1, 3)
    lower = lower[np.isfinite(lower).all(axis=1)]
    floor_normal, floor_offset = _fit_floor(lower)

    basis = _world_basis(floor_normal)
    floor_z = float((basis @ (floor_normal * floor_offset))[2])
    flat = cloud.reshape(-1, 3)

    observations: dict[str, Observation] = {}
    for mask in segments.masks:
        found = _observe(mask.mask_path, depth_map.shape, flat, basis, floor_z)
        if found is not None:
            observations[mask.object_id] = found
    return observations, basis, floor_z


def resolve_supports(graph: SceneGraph, settings) -> SceneGraph:
    """Replace claimed support relations the geometry contradicts.

    `ObjectLabel.support_parent` is a VLM guess made two stages before any 3D
    exists — stage 3 looks at a flat photo with numbered outlines and judges "is
    the sofa on the rug" from one viewpoint. `schemas.ObjectLabel` has always said
    reconcile overrides that with measured contact once geometry exists; this is
    that override, and until it existed the guess went straight through to the
    solver.

    It matters because the guess is not stable. Across consecutive runs of the same
    photo the same model called a sofa floor-supported and then rug-supported; on
    the run where it said rug, the sofa's footprint centre was 0.19 m clear of the
    rug's near edge, the scale axis failed `base_inside_parent`, and the solver
    spent the round driving the sofa down onto a surface it was not over.

    **The nearest measurable surface wins; the claim survives only where nothing is
    measurable.** An earlier version kept the guess whenever it was merely viable,
    which sounds conservative and is too weak to be useful: an object stacked on a
    high shelf but labelled with the low one stayed on the low one, because half a
    metre of gap is "viable" under the threshold below. Since the schema promises
    *measured* contact, the measurement decides whenever there is one.

    The fallback matters as much as the rule. Nothing has been solved at this point,
    so this reads the raw reconstruction, and an object badly enough placed that no
    surface sits under its centre is a geometry failure rather than a labelling one
    — refuting a claim is evidence, failing to confirm one is not. In that case the
    guess is left alone and the solver still has a contact to close.

    A parent has to satisfy two things, matching what `certify.scale` will later
    check rather than inventing a second rule:

    - the child's footprint *centre* over the parent's surface, which is the
      toppling condition; a corner overhanging a neighbour is not resting on it
    - a gap within `max_snap_m`, the same distance repair already refuses to move
      an object, beyond which the relation is more likely wrong than the position
    """
    if not graph.objects:
        return graph

    from app.pipeline import support

    ids = {obj.object_id for obj in graph.objects}
    heights = support.build(graph, settings, candidates=ids)
    boxes = {obj.object_id: world_aabb(obj) for obj in graph.objects}

    def surface_under(child: SceneObject, parent: SceneObject) -> float | None:
        low, high = boxes[child.object_id]
        return heights.under(parent, low, high, centre_only=True)

    def base_of(obj: SceneObject) -> float:
        measured = heights.base_of(obj)
        return measured if measured is not None else float(boxes[obj.object_id][0][2])

    objects = []
    for obj in graph.objects:
        base = base_of(obj)

        # Every candidate parent, plus the floor as `None`, scored by how far the
        # object is from resting on it.
        scored: list[tuple[float, str | None]] = [(abs(base - graph.floor_height_m), None)]
        for other in graph.objects:
            if other.object_id == obj.object_id:
                continue
            # Only something whose own base is lower can hold this up. Cheap, and it
            # makes the resulting graph acyclic by construction rather than by a
            # cycle check afterwards.
            if base_of(other) >= base:
                continue
            surface = surface_under(obj, other)
            if surface is not None:
                scored.append((abs(base - surface), other.object_id))

        claimed = obj.supported_by
        viable = [entry for entry in scored if entry[0] <= settings.max_snap_m]

        if not viable:
            # Nothing is convincingly under this object, including the floor. The
            # claim is unverifiable rather than refuted, and replacing a considered
            # guess with a worse one is not an improvement — this is the badly
            # reconstructed object, not the badly labelled one.
            objects.append(obj)
            continue

        chosen = min(viable)[1]
        if chosen != claimed:
            log.info(
                "%s: support %s -> %s (claimed contact not supported by geometry)",
                obj.object_id,
                claimed,
                chosen,
            )
        objects.append(obj.model_copy(update={"supported_by": chosen}))

    return graph.model_copy(update={"objects": objects})


def _placeholder_label(object_id: str) -> ObjectLabel:
    """A label for an object the VLM never described.

    Better than dropping it: unlabelled geometry still occupies space, and anything
    resting on it needs it to exist.
    """
    return ObjectLabel(object_id=object_id, category="other", material=Material.OTHER)


def run(
    ctx: PipelineContext,
    reconstruction: ReconstructionResult,
    segments: SegmentResult,
    depth: DepthResult,
    labels: LabelResult,
) -> SceneGraph:
    observations, basis, floor_z = observe(ctx, segments, depth)

    by_id = {label.object_id: label for label in labels.labels}
    known = {asset.object_id for asset in reconstruction.objects}
    # Masks come from the SegmentResult rather than a filename convention: segment
    # owns where it writes them, and guessing the path here would be a second place
    # that has to agree.

    objects: list[SceneObject] = []
    for asset in reconstruction.objects:
        label = by_id.get(asset.object_id) or _placeholder_label(asset.object_id)
        if label.support_parent not in known:
            # A support naming an object reconstruction dropped is a dangling
            # reference; the floor is the honest fallback.
            label = label.model_copy(update={"support_parent": None})

        observed = observations.get(asset.object_id)
        if observed is None:
            log.warning("%s: no usable depth under its mask; skipped", asset.object_id)
            continue
        centre, extent, footprint = observed

        part = asset.parts[0]
        # Mesh extents are in the mesh's own Y-up frame; the world extent to match
        # them against has already been rotated, so permute rather than compare
        # blind.
        mesh_extent = np.array([part.dims_m[0], part.dims_m[2], part.dims_m[1]])
        scale = _fit_scale(mesh_extent, extent)

        # Yaw is measured relative to the mesh's own long horizontal axis, not to
        # the world: aligning a mesh that is already long in Y to a world direction
        # would rotate it by an extra 90 degrees.
        observed_yaw = _observed_yaw(footprint)
        yaw = 0.0
        if observed_yaw is not None:
            mesh_yaw = 0.0 if mesh_extent[0] >= mesh_extent[1] else np.pi / 2.0
            yaw = observed_yaw - mesh_yaw
            # A principal axis has no sign, so yaw and yaw+pi fit equally well.
            # Neither can be told apart from a single-view shell — a sofa's front
            # and back look the same to a covariance matrix — so take the smaller
            # rotation and leave the ambiguity documented rather than guessed.
            yaw = (yaw + np.pi / 2.0) % np.pi - np.pi / 2.0

        objects.append(
            SceneObject(
                object_id=asset.object_id,
                label=label,
                frame=asset.frame,
                parts=asset.parts,
                scale=scale,
                position_m=(float(centre[0]), float(centre[1]), float(centre[2])),
                orientation=_yaw_quaternion(yaw),
                supported_by=label.support_parent,
                degradation_reason=asset.failure_reason if asset.failed else None,
            )
        )

    objects = _ground(objects)
    log.info("reconciled %d of %d objects", len(objects), len(reconstruction.objects))
    # Last, so it sees the final placements. Everything downstream — solve, certify,
    # the exports, the viewer — works in the recentred frame; `world_offset_m` is what
    # lets solve get back to the camera coordinates its depth observations live in.
    graph = recentre(SceneGraph(objects=objects, camera=_camera_pose(basis, floor_z, depth)))
    log.info("recentred by %s", np.round(graph.world_offset_m, 3))
    # After recentring, so the geometry it measures is the geometry everything
    # downstream will use.
    return resolve_supports(graph, ctx.settings)


def _ground(objects: list[SceneObject]) -> list[SceneObject]:
    """Drop each object vertically until its base rests on its support.

    Depth places an object at the centre of the 5th-to-95th percentile of the points
    under its mask. That is the right estimator for x, y and for *size*, and the
    wrong one for height above the floor, for two reasons that both push the same
    way. The bottom 5% is clipped away, and the bottom of a floor-standing object is
    exactly the part in contact — so the clip lands on the one surface whose position
    is known a priori. And `_fit_scale` takes the median of three axis ratios, so
    when the vertical ratio is the odd one out the mesh is stretched past its
    observed extent and overhangs the centre in both directions.

    Measured on room.jpg, floor-supported objects came out anywhere from 177 mm below
    the floor to 37 mm above it — a scene where nothing is level and every object
    fails the scale axis on support gap before the solver has seen it.

    Resting contact is a hard constraint, not an observation to be averaged, so it is
    imposed rather than fitted. Only z moves; x and y keep the depth estimate, which
    has no comparable bias. `repair._snap_to_support` still exists and still runs —
    it catches what later stages break, whereas this stops the scene from being born
    broken.

    Support tops are read as the parent's world AABB z-max, matching what
    `certify.scale` measures the gap against, so agreement here is by construction
    rather than by coincidence of two similar formulas.
    """
    by_id = {obj.object_id: obj for obj in objects}
    placed: dict[str, SceneObject] = {}

    def resolve(object_id: str, seen: frozenset[str]) -> SceneObject:
        if object_id in placed:
            return placed[object_id]
        obj = by_id[object_id]
        parent_id = obj.supported_by

        # A support cycle is a reconcile bug, but hanging is a worse response than
        # treating the object as floor-standing and carrying on.
        if parent_id in seen:
            log.warning(
                "%s: support cycle through %s; grounding on the floor", object_id, parent_id
            )
            parent_id = None

        if parent_id is None or parent_id not in by_id:
            support_top = 0.0
        else:
            support_top = float(world_aabb(resolve(parent_id, seen | {object_id}))[1][2])

        low = world_aabb(obj)[0][2]
        x, y, z = obj.position_m
        placed[object_id] = obj.model_copy(
            update={"position_m": (x, y, float(z + support_top - low))}
        )
        return placed[object_id]

    # Original order preserved: object order is not semantic, but a stage that
    # reshuffles its output makes every downstream diff unreadable.
    return [resolve(obj.object_id, frozenset()) for obj in objects]


def _camera_pose(basis: np.ndarray, floor_z: float, depth: DepthResult) -> CameraPose:
    """Where the photo was taken from, in world coordinates.

    Exact rather than guessed: the camera is at the origin of the frame this stage
    rotates and shifts, so it lands at (0, 0, -floor_z) — and that z is the
    photographer's eye height above the fitted floor, which doubles as a sanity
    check on the plane.
    """
    forward = basis @ np.array([0.0, 0.0, 1.0])
    up = basis @ np.array([0.0, -1.0, 0.0])
    height = -floor_z
    fov = 2.0 * np.degrees(np.arctan(depth.intrinsics.cy / depth.intrinsics.fy))
    log.info("camera %.2f m above the floor, %.0f deg vertical fov", height, fov)
    return CameraPose(
        position_m=(0.0, 0.0, float(height)),
        forward=tuple(float(v) for v in forward),
        up=tuple(float(v) for v in up),
        vertical_fov_deg=float(fov),
    )


def _observe(mask_path, shape, flat_cloud, basis, floor_z):
    """Metric centre and extent of one object, from the depth under its mask."""
    if mask_path is None:
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    selected = flat_cloud[(mask > 127).reshape(-1)]
    selected = selected[np.isfinite(selected).all(axis=1)]
    if len(selected) < MIN_MASK_PX:
        return None

    world = selected @ basis.T
    world[:, 2] -= floor_z

    low = np.percentile(world, EXTENT_LOW, axis=0)
    high = np.percentile(world, EXTENT_HIGH, axis=0)
    # The horizontal footprint comes back too, for the yaw estimate.
    return Observation((low + high) / 2.0, high - low, world[:, :2])


def _fit_scale(mesh_extent: np.ndarray, observed_extent: np.ndarray) -> float:
    """One isotropic factor bringing a normalised mesh to an observed size.

    The median of the per-axis ratios rather than the mean: depth is least reliable
    along the axis pointing away from the camera, where only the front shell is
    visible, and a median lets the two well-observed axes outvote it.
    """
    usable = mesh_extent > 1e-6
    if not usable.any():
        return 1.0
    ratios = observed_extent[usable] / mesh_extent[usable]
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    return float(np.median(ratios)) if len(ratios) else 1.0
