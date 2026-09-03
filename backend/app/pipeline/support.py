"""How high is the surface under an object, measured on real geometry.

Stage 6's `E_supp` needs one number per support edge: the height of the surface
the child rests on. It used to take the top of the parent's bounding box, which is
wrong in two different ways and both were measured on `room.png`.

**A box's top is not a support surface.** A sofa's bounding box is topped by its
backrest, not its seat — 0.63 m against 0.40 m. Every cushion placed on that sofa
was seeded 23 cm too high, fell most of that under gravity, and rotated 165 deg on
the way down, which is what the stability axis reported.

**The box is not even the mesh's box.** `geometry.world_aabb` builds it from the
stored `PartGeometry.dims_m` and `origin_m` rather than measuring anything, and on
a reconstructed rug those disagree with the collision geometry by 10 mm at the
bottom and 8.5 mm at the top. Driving the table's stored box-bottom onto the rug's
stored box-top left the two real surfaces 12.4 mm apart, so the table fell 17 mm
and took the mug and book on it down with it. That one is not a modelling
approximation, it is two sources of truth for the same shape.

So: rasterise the parent's *collision* geometry from above into a height grid, and
read the height under the child's footprint out of it. Collision geometry rather
than visual, because that is what MuJoCo will grade the result against, and the
whole failure above is the solver and the certifier disagreeing about shape.

**Built in the parent's oriented unit-scale frame, which is what makes it exact.**
The naive version bakes the parent's world pose into the grid and goes stale the
moment the solver moves the parent — and the parent is a variable too, so it goes
stale inside a single Gauss-Newton pass rather than between rounds. Storing
heights in a frame that has the parent's rotation applied but neither its scale
nor its translation costs nothing and makes the lookup exact under both of the
things the solver actually changes:

    oriented_xy = (world_xy - parent.position_xy) / parent.scale
    world_z     = parent.position_z + parent.scale * grid(oriented_xy)

Only a change of *orientation* would invalidate a grid, and orientation is not a
stage-6 variable — yaw comes from the depth cloud's principal axis in stage 5 and
is held. So a grid is built once per solve round and is correct for every residual
evaluation inside it.

Sampled over the footprint rather than at its centre, taking the highest hit. An
object hanging over the edge of its support rests on the part that is actually
underneath it, and the centre alone would read the empty air beyond the edge as
"no surface" and fall back to the box.
"""

import logging
from dataclasses import dataclass

import numpy as np
import trimesh

from app.config import Settings
from app.export.gltf import geometry_for
from app.geometry import quat_to_matrix
from app.schemas import PartGeometry, SceneGraph, SceneObject

__all__ = ["SupportHeights", "build"]

log = logging.getLogger(__name__)

# Where to probe under the child, as fractions of its half-extent from the centre.
# The centre plus four points toward the corners: enough to find the supported part
# of an overhanging object without turning one residual into a mesh query storm.
_PROBES = ((0.0, 0.0), (0.6, 0.6), (0.6, -0.6), (-0.6, 0.6), (-0.6, -0.6))


def _oriented_transform(obj: SceneObject, part: PartGeometry) -> np.ndarray:
    """Part-local geometry into the parent's oriented, unit-scale, untranslated frame.

    Deliberately *not* `gltf.node_transform`, which also applies scale and
    position. Leaving those out is what lets the grid outlive a solver step; see
    the module docstring.
    """
    rotation = np.eye(4)
    rotation[:3, :3] = quat_to_matrix(obj.orientation)
    offset = np.eye(4)
    offset[:3, 3] = np.asarray(part.origin_m, dtype=float)
    return rotation @ offset


def _collision_mesh(obj: SceneObject) -> trimesh.Trimesh | None:
    """Every part's collision geometry, in the oriented unit-scale frame.

    Falls back to `gltf.geometry_for` — the visual mesh, or a box from the OBB
    extent — when the inertia stage has not run. That keeps this usable on the
    hand-authored fixtures, which carry no collision paths, and it degrades to
    exactly the behaviour it is replacing rather than to nothing.
    """
    pieces: list[trimesh.Trimesh] = []
    for part in obj.parts:
        transform = _oriented_transform(obj, part)
        if part.collision_mesh_paths:
            for path in part.collision_mesh_paths:
                try:
                    loaded = trimesh.load(path, force="mesh")
                except Exception as exc:  # a proxy that went astray is not fatal
                    log.warning("support: cannot read %s (%s)", path, exc)
                    continue
                if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
                    piece = loaded.copy()
                    piece.apply_transform(transform)
                    pieces.append(piece)
        else:
            piece = geometry_for(part).copy()
            piece.apply_transform(transform)
            pieces.append(piece)

    if not pieces:
        return None
    return trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]


@dataclass(frozen=True)
class _Grid:
    """Top-surface heights over the oriented frame's XY, `nan` where nothing is."""

    origin_xy: np.ndarray
    cell_m: float
    heights: np.ndarray

    def sample(self, x: float, y: float) -> float | None:
        col = int((x - self.origin_xy[0]) / self.cell_m)
        row = int((y - self.origin_xy[1]) / self.cell_m)
        if not (0 <= row < self.heights.shape[0] and 0 <= col < self.heights.shape[1]):
            return None
        value = float(self.heights[row, col])
        return None if np.isnan(value) else value


def _rasterise(mesh: trimesh.Trimesh, cell_m: float, max_cells: int) -> _Grid | None:
    """Cast rays straight down on a grid and keep the highest hit in each cell.

    Rays rather than a triangle rasteriser because the geometry is a pile of
    overlapping convex hulls from CoACD, and "the highest surface at this xy" is
    exactly a ray query — projecting triangles would mean resolving that overlap by
    hand for no gain. Measured on a 1382-face rug, 2304 rays takes 0.03 s, which is
    once per support per round.
    """
    low, high = mesh.bounds
    span = np.maximum(high[:2] - low[:2], cell_m)
    counts = np.clip(np.ceil(span / cell_m).astype(int), 1, max_cells)
    # Widen the cell rather than clip the extent, so a large support stays fully
    # covered at coarser resolution instead of losing its edges.
    cell = float(max(cell_m, *(span / counts)))

    xs = low[0] + (np.arange(counts[0]) + 0.5) * cell
    ys = low[1] + (np.arange(counts[1]) + 0.5) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    origins = np.stack(
        [grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, high[2] + 1.0)], axis=1
    )
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))

    points, ray_index, _ = mesh.ray.intersects_location(origins, directions)
    if not len(points):
        return None

    heights = np.full(grid_x.size, -np.inf)
    np.maximum.at(heights, ray_index, points[:, 2])
    heights[np.isinf(heights)] = np.nan
    return _Grid(
        origin_xy=np.asarray(low[:2], dtype=float),
        cell_m=cell,
        heights=heights.reshape(grid_x.shape),
    )


class SupportHeights:
    """Height grids for the objects something rests on, queried in world space.

    Also carries each object's own *base* — the lowest point of its collision
    geometry. Both sides of a support contact have to be measured the same way or
    the residual closes onto a number that is not the contact: driving a table's
    stored box-bottom onto a rug's measured surface still left the table's real
    underside 2.8 mm high, because the box and the mesh disagree by that much.
    """

    def __init__(self, grids: dict[str, _Grid], bases: dict[str, float]):
        self._grids = grids
        self._bases = bases

    def base_of(self, obj: SceneObject) -> float | None:
        """World z of the lowest point of this object's collision geometry.

        One scalar, not a grid: the same oriented unit-scale trick applies, and
        orientation is not a stage-6 variable, so `position_z + scale * offset` is
        exact under everything the solver moves.
        """
        offset = self._bases.get(obj.object_id)
        if offset is None:
            return None
        return float(obj.position_m[2] + obj.scale * offset)

    def under(self, parent: SceneObject, low: np.ndarray, high: np.ndarray) -> float | None:
        """World-space surface height under a child's footprint, or None.

        `low` and `high` are the child's world AABB — only its XY extent is read,
        to place the probes. None means the grid has nothing under any probe, which
        is the honest answer when the child is not over its support at all, and
        leaves the caller to fall back.
        """
        grid = self._grids.get(parent.object_id)
        if grid is None:
            return None

        centre_xy = (np.asarray(low[:2]) + np.asarray(high[:2])) / 2.0
        half_xy = (np.asarray(high[:2]) - np.asarray(low[:2])) / 2.0
        position = np.asarray(parent.position_m, dtype=float)

        best: float | None = None
        for fx, fy in _PROBES:
            world_xy = centre_xy + half_xy * (fx, fy)
            oriented = (world_xy - position[:2]) / parent.scale
            height = grid.sample(float(oriented[0]), float(oriented[1]))
            if height is not None and (best is None or height > best):
                best = height
        if best is None:
            return None
        return float(position[2] + parent.scale * best)


def build(graph: SceneGraph, settings: Settings) -> SupportHeights:
    """Grids for every object that something in this graph rests on.

    Only those: a grid costs a mesh load and a few thousand rays, and an object
    nothing is resting on will never be asked.
    """
    wanted = {obj.supported_by for obj in graph.objects if obj.supported_by}
    # Bases are wanted for every object that rests on something, grids only for the
    # objects being rested *on*. An object can of course be both.
    needs_base = {obj.object_id for obj in graph.objects if obj.supported_by}

    grids: dict[str, _Grid] = {}
    bases: dict[str, float] = {}
    for obj in graph.objects:
        if obj.object_id not in wanted and obj.object_id not in needs_base:
            continue
        mesh = _collision_mesh(obj)
        if mesh is None:
            continue
        if obj.object_id in needs_base:
            bases[obj.object_id] = float(mesh.bounds[0][2])
        if obj.object_id in wanted:
            grid = _rasterise(mesh, settings.support_grid_cell_m, settings.support_grid_max_cells)
            if grid is not None:
                grids[obj.object_id] = grid
    log.info("support: %d height grids, %d bases", len(grids), len(bases))
    return SupportHeights(grids, bases)
