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
from app.geometry import dependents, quat_to_matrix
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


# How many underside points a contact query may probe per object.
#
# The raster gives one per grid cell — 22,684 across `room2.png` — and `gap_to` runs
# inside the solver's residual, so that number is multiplied by the finite-difference
# Jacobian and again by every round: measured, one pass over the scene cost 22 ms and
# a re-solve took 318 s. The gap is a *minimum* over those points, and a minimum does
# not need every sample of a surface, only coverage of one. Striding preserves the
# spatial spread the raster's row-major order already has.
MAX_CONTACT_SAMPLES = 128


@dataclass(frozen=True)
class _Grid:
    """Every surface height in each column of the oriented frame's XY.

    Every one, not just the topmost, because a top surface is not what a child
    rests on whenever the parent has anything above it. A bookshelf's column
    contains the top of the unit, each shelf, and the floor of the carcass; a
    basket's contains its rim and its inside floor. Keeping only the maximum
    answers "how high is this object" when the question is "what is this child
    standing on", and the two differ by a whole shelf.
    """

    origin_xy: np.ndarray
    cell_m: float
    shape: tuple[int, int]
    # Flat cell index -> ascending surface heights in that column. Sparse: a column
    # the rays missed is simply absent.
    columns: dict[int, np.ndarray]
    # The same columns as flat arrays, for the nearest-support query. Precomputed
    # because that query runs inside the solver's residual, and rebuilding it per
    # evaluation would cost more than the term is worth.
    cell_xy: np.ndarray  # (N, 2) centre of each occupied column
    cell_lo: np.ndarray  # (N,) lowest surface in it
    cell_hi: np.ndarray  # (N,) highest surface in it

    def _column(self, x: float, y: float) -> np.ndarray | None:
        """Every surface height in this column, ascending, or None outside the grid."""
        col = int((x - self.origin_xy[0]) / self.cell_m)
        row = int((y - self.origin_xy[1]) / self.cell_m)
        if not (0 <= row < self.shape[0] and 0 <= col < self.shape[1]):
            return None
        return self.columns.get(row * self.shape[1] + col)

    def lowest(self, x: float, y: float) -> float | None:
        """The bottom of this column — the lowest surface the rays found in it.

        What `sample` needs but cannot ask, because its fallback has already decided
        what to do when nothing lies at or below the child: whether this object has
        any geometry *under* a given height at all. An object entirely above a point
        is not something that point can rest on and not something it can be buried
        in; it is a neighbour overlapping in plan view. See `SupportHeights.contact`.
        """
        heights = self._column(x, y)
        return None if heights is None else float(heights[0])

    def sample(self, x: float, y: float, below: float | None = None, slack: float = 0.0):
        """The surface height in this column, or None if the column is empty.

        `below` is the child's base in this frame. Given it, the answer is the
        highest surface at or below that base — the thing the child is standing on
        — rather than the highest surface outright. `slack` is how far a surface may
        sit *above* the base and still count, which is what keeps an object that
        reconstruction buried a few centimetres attached to the shelf it is buried
        in rather than snapping to the one beneath.

        With `below` unset this is the old top-surface query, which is still what a
        caller wants when it has no child to speak of.
        """
        heights = self._column(x, y)
        if heights is None:
            return None
        if below is None:
            return float(heights[-1])
        allowed = heights[heights <= below + slack]
        # Nothing at or below it means there is no surface here to rest on, and the
        # honest answer is to say so.
        #
        # This used to fall back to the column's *lowest* surface, on the reasoning
        # that it was "the nearest thing it could be resting on". It is not: for a
        # child under a tabletop the lowest surface in the column is the tabletop's
        # *underside*, and the support term is a signed residual driven to zero, so
        # it dutifully pulled the object up until its top met the table's bottom.
        # Measured: a book dragged below a side table reported a gap of -446.7 mm
        # against that underside and re-solved to hang beneath the table — which is
        # exactly what a user who dragged books around and committed the edit saw.
        # An object cannot rest on a downward-facing surface, so offering one as
        # support does not produce a large gap, it produces a confidently wrong
        # direction. Returning None lets the caller fall back to the parent's box
        # top, which is at least the right side of it.
        return float(allowed[-1]) if len(allowed) else None

    def nearest_xy(self, x: float, y: float, base: float, slack: float):
        """Centre of the nearest column that could hold something at `base`, or None.

        "Could hold" is deliberately loose — the child's base lies anywhere within
        that column's vertical span, widened by `slack`. A tighter test would have
        to pick which shelf, and the caller only wants a direction to pull in.
        """
        if not len(self.cell_xy):
            return None
        usable = (base >= self.cell_lo - slack) & (base <= self.cell_hi + slack)
        if not usable.any():
            return None
        candidates = self.cell_xy[usable]
        deltas = candidates - np.array([x, y])
        return candidates[int(np.argmin((deltas * deltas).sum(axis=1)))]


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

    # Every hit per column, ascending — one ray through a bookshelf returns one
    # height per shelf it passes through, and which of them matters depends on the
    # child asking.
    order = np.lexsort((points[:, 2], ray_index))
    cells, starts = np.unique(ray_index[order], return_index=True)
    zs = np.split(points[order, 2], starts[1:])
    columns = {int(c): z for c, z in zip(cells, zs, strict=True)}

    rows, cols = np.divmod(cells, grid_x.shape[1])
    return _Grid(
        origin_xy=np.asarray(low[:2], dtype=float),
        cell_m=cell,
        shape=(int(grid_x.shape[0]), int(grid_x.shape[1])),
        columns=columns,
        cell_xy=np.stack([low[0] + (cols + 0.5) * cell, low[1] + (rows + 0.5) * cell], axis=1),
        cell_lo=np.array([z[0] for z in zs]),
        cell_hi=np.array([z[-1] for z in zs]),
    )


@dataclass(frozen=True)
class Contact:
    """What an object is resting on, measured without reference to any declared parent."""

    gap: float | None
    """Smallest clearance from the underside to whatever is beneath it. Negative is
    overlap. None when the object has no measurable underside."""

    resting_on: str | None
    """The object providing that nearest surface, or None for the floor."""

    contacts_xy: np.ndarray
    """(k, 2) world XY of the underside points actually in contact. The support
    polygon: an object is stable when its weight falls inside this, which is the
    toppling condition a bounding-box test only approximates."""

    def over_support(self, centre_xy: np.ndarray) -> bool:
        """Whether that centre falls within the contact points' footprint.

        Three points are the minimum that can enclose anything; with fewer, an
        object is balanced rather than resting and the honest answer is no.
        """
        if len(self.contacts_xy) < 3:
            return False
        try:
            from scipy.spatial import ConvexHull, Delaunay

            hull = self.contacts_xy[ConvexHull(self.contacts_xy).vertices]
            return bool(Delaunay(hull).find_simplex(np.asarray(centre_xy)) >= 0)
        except Exception:
            # Degenerate contacts — collinear feet, say — have no interior to be
            # inside of, and a collapsed hull raises rather than returning empty.
            return False


class SupportHeights:
    """Height grids for the objects something rests on, queried in world space.

    Also carries each object's own *base* — the lowest point of its collision
    geometry. Both sides of a support contact have to be measured the same way or
    the residual closes onto a number that is not the contact: driving a table's
    stored box-bottom onto a rug's measured surface still left the table's real
    underside 2.8 mm high, because the box and the mesh disagree by that much.
    """

    def __init__(
        self,
        grids: dict[str, _Grid],
        bases: dict[str, float],
        burial_slack_m: float = 0.0,
        undersides: dict[str, np.ndarray] | None = None,
    ):
        self._grids = grids
        self._bases = bases
        # object_id -> (k, 3) points on its own underside, in its oriented unit-scale
        # frame. What `gap_to` probes with, so both sides of a contact are read at the
        # same places. See that method.
        self._undersides = undersides or {}
        # Carried here rather than passed at every call site, so the solver, the
        # certifier and reconcile cannot disagree about which surface a contact is.
        self.burial_slack_m = burial_slack_m

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

    def under(
        self,
        parent: SceneObject,
        low: np.ndarray,
        high: np.ndarray,
        centre_only: bool = False,
        base: float | None = None,
        slack: float = 0.0,
    ) -> float | None:
        """World-space surface height under a child's footprint, or None.

        `low` and `high` are the child's world AABB — only its XY extent is read,
        to place the probes. None means the grid has nothing under any probe, which
        is the honest answer when the child is not over its support at all, and
        leaves the caller to fall back.

        `centre_only` asks the narrower question "is this object's middle over that
        surface", which is the toppling condition rather than the contact one. The
        support *term* wants the contact and probes the whole footprint; deciding
        *which* object something rests on wants the toppling test, because an object
        overhanging a neighbour by one corner is not resting on it.

        `base` is the child's own underside in world z, and passing it is what makes
        this correct for a parent that has geometry above the contact. Without it
        the answer is the parent's topmost surface, which for a book on a middle
        shelf is the top of the bookcase and for a magazine in a basket is the rim —
        in both cases a surface the child is nowhere near and, in the basket's case,
        one the solver then drives it up through the wall to reach.
        """
        grid = self._grids.get(parent.object_id)
        if grid is None:
            return None

        centre_xy = (np.asarray(low[:2]) + np.asarray(high[:2])) / 2.0
        half_xy = (np.asarray(high[:2]) - np.asarray(low[:2])) / 2.0
        position = np.asarray(parent.position_m, dtype=float)

        # The child's base in the parent's oriented unit-scale frame, so it is
        # comparable with the heights the grid stores.
        oriented_base = None if base is None else (base - position[2]) / parent.scale
        oriented_slack = slack / parent.scale

        best: float | None = None
        for fx, fy in ((0.0, 0.0),) if centre_only else _PROBES:
            world_xy = centre_xy + half_xy * (fx, fy)
            oriented = (world_xy - position[:2]) / parent.scale
            height = grid.sample(
                float(oriented[0]), float(oriented[1]), oriented_base, oriented_slack
            )
            if height is not None and (best is None or height > best):
                best = height
        if best is None:
            return None
        return float(position[2] + parent.scale * best)

    def contact(
        self,
        child: SceneObject,
        candidates: list[SceneObject],
        floor_z: float,
        tolerance: float,
    ) -> "Contact":
        """What this object is actually touching, and whether that holds it up.

        Parent-free on purpose. An object rests on whatever is beneath it, and in a
        real scene that is routinely more than one thing: measured on `room2.png`,
        28 of the armchair's 33 foot samples sit over the rug and 5 over bare floor.
        A single-parent gap answers that two ways and both are wrong — 6.2 mm
        against the rug, ignoring the leg on the floor, or 21 mm against the floor,
        ignoring the three legs on the rug. Neither is a defect: a chair with one leg
        off a rug is an ordinary stable object, and so is a book overlapping another
        book by most but not all of its face.

        So the geometry here answers only geometric questions — is it floating, is it
        buried, is its weight over its contacts. Whether it is on *the object the
        photo shows it on* is a separate, semantic question, answered by `touches`.

        Parent-free is not candidate-free: the one set of objects that cannot be
        holding this one up is the set resting on it. Nothing else here excludes
        them, and the `burial_slack_m` window is wide enough to let one in — measured
        on `room2.png`, a book was reported resting 98.87 mm *inside* the book
        standing on it, a shave under the 100 mm slack, which is the signature of a
        bound doing the choosing. Its real clearance to the side table it rests on
        was 4.66 mm. That is the same "a book on a table does not hold the table up"
        error `gap_to` documents, arriving through a different door: the band filter
        below drops the child's own upward-facing geometry, not a whole object above
        it. It also cost `certify.scale` a false failure and `certify.repair` a
        98.9 mm snap proposed and rejected in every one of its five rounds, and in
        `reconcile.resolve_supports` — the other caller — adopting it would have
        closed a cycle in the support graph.
        """
        points = self._undersides.get(child.object_id)
        if points is None or not len(points):
            base = self.base_of(child)
            gap = None if base is None else base - floor_z
            return Contact(gap=gap, resting_on=None, contacts_xy=np.empty((0, 2)))

        position = np.asarray(child.position_m, dtype=float)
        world = position + child.scale * points

        # Only the lowest band of the underside — where contact actually happens.
        #
        # The raster covers every downward-facing surface, which for a pedestal table
        # means its feet *and* the underside of its top, 600 mm higher. Taking the
        # minimum clearance over all of them let a book resting on the tabletop count
        # as support for the tabletop's own underside, and the table was reported
        # resting 98 mm inside that book. A book on a table does not hold the table
        # up. The band is `burial_slack_m` deep so a tilted or slightly buried object
        # keeps all of its feet rather than just the lowest one.
        floor_band = world[:, 2] <= world[:, 2].min() + self.burial_slack_m
        world = world[floor_band]

        # Itself, and everything transitively resting on it. See the docstring.
        excluded = dependents(candidates, child.object_id) | {child.object_id}

        best_gap: float | None = None
        resting_on: str | None = None
        touching: list[np.ndarray] = []
        for x, y, z in world:
            # Nearest surface to this point, not the highest one under it.
            #
            # Both bounds have bitten. Choosing the *highest* within a generous
            # window let anything within 100 mm above a point claim to support it — a
            # side table reported resting 100.0 mm inside a book, exactly the slack,
            # which is the signature of a bound doing the choosing. Tightening the
            # window instead broke the opposite case: a mug buried 50 mm in a table
            # has that table's surface *above* its underside, so it was rejected and
            # the floor won, reporting +700 mm of clearance on an object sitting
            # inside its own support.
            #
            # Nearest-with-a-generous-window handles both. The buried mug finds the
            # table 50 mm up rather than the floor 700 mm down; the side table finds
            # the floor at its feet rather than a book a hand's width above it.
            surface, owner = float(floor_z), None
            nearest = abs(z - floor_z)
            for parent in candidates:
                if parent.object_id in excluded:
                    continue
                grid = self._grids.get(parent.object_id)
                if grid is None:
                    continue
                origin = np.asarray(parent.position_m, dtype=float)
                ox, oy = (np.array([x, y]) - origin[:2]) / parent.scale
                height = grid.sample(
                    float(ox),
                    float(oy),
                    (z - origin[2]) / parent.scale,
                    self.burial_slack_m / parent.scale,
                )
                if height is None:
                    continue
                world_h = origin[2] + parent.scale * height
                if world_h > z + self.burial_slack_m:
                    continue  # above the point by more than reconstruction error
                # And it has to straddle the point to be under it at all. A surface
                # above the point means one of two things — the child is buried in
                # this object, or this object is simply a neighbour overlapping in
                # plan view — and `grid.sample`'s "under the whole parent" fallback
                # returns the column's lowest surface either way. Burial always
                # straddles: a mug 50 mm into a table has the tabletop above its
                # underside and the table's legs below it. A neighbour does not.
                #
                # Measured on `room2.png`: a vase resting on a shelf at +2.55 mm over
                # 1640 of its 1658 underside points was reported 93.97 mm *inside* the
                # plant standing beside it on that shelf, on the strength of 51 points
                # near its rim where the plant's foliage overlaps in plan view. The
                # plant's column is entirely above those points — lowest surface
                # 1.9970 against a point at 1.9509 — and `certify.stability` measured
                # the pair as not touching at all. `contact` takes the minimum gap
                # over the underside, so those 51 points outvoted the other 1640.
                #
                # Threshold-free, and it costs one array lookup: the column is already
                # in hand. Compare `burial_slack_m`, which is the bound that let this
                # in and is doing its own job correctly one level down, choosing which
                # shelf in the column the child stands on.
                lowest = grid.lowest(float(ox), float(oy))
                if lowest is None or origin[2] + parent.scale * lowest > z:
                    continue
                delta = abs(z - world_h)
                if delta < nearest:
                    surface, owner, nearest = world_h, parent.object_id, delta
            gap = float(z) - surface
            if best_gap is None or gap < best_gap:
                best_gap, resting_on = gap, owner
            if abs(gap) <= tolerance:
                touching.append(np.array([x, y]))

        return Contact(
            gap=best_gap,
            resting_on=resting_on,
            contacts_xy=np.asarray(touching) if touching else np.empty((0, 2)),
        )

    def touches(self, parent: SceneObject, child: SceneObject, tolerance: float) -> bool:
        """Whether the child meets *this* parent anywhere, within tolerance.

        The semantic half of the question `contact` deliberately does not answer. A
        book that fell from its table to the floor has a perfectly good clearance to
        the floor and is perfectly stable; what is wrong with it is that it is not on
        the table, and only a check that names the table can say so.
        """
        gap = self.gap_to(parent, child, 0.0)
        return gap is not None and abs(gap) <= tolerance

    def gap_to(self, parent: SceneObject, child: SceneObject, floor_z: float) -> float | None:
        """Signed distance from the child to the first thing it would touch.

        Negative is overlap, positive is clearance, and the answer is the *minimum*
        over the child's own underside — the place it lands first, which is the only
        one that decides whether it is resting.

        This exists because measuring the two sides separately does not work. The
        old pair was `base_of` — the lowest point *anywhere* on the child — against
        `under` — the highest surface *anywhere beneath the child's bounding box* —
        and nothing makes those the same (x, y). For a solid box they coincide. For a
        pedestal they do not: measured on `room2.png`, the side table's contact patch
        is 0.107 x 0.037 m inside a 0.56 x 0.58 m footprint, so 99% of what was being
        probed is the empty air under a round top, and its 9.2 mm "gap" was one low
        foot compared against a rug sampled somewhere else entirely.

        `parent` of None means the floor, which is a plane and so needs no probing.
        """
        points = self._undersides.get(child.object_id)
        if points is None or not len(points):
            base = self.base_of(child)
            return None if base is None else base - floor_z

        position = np.asarray(child.position_m, dtype=float)
        world = position + child.scale * points
        if parent is None:
            return float(np.min(world[:, 2]) - floor_z)

        grid = self._grids.get(parent.object_id)
        if grid is None:
            return None
        origin = np.asarray(parent.position_m, dtype=float)
        slack = self.burial_slack_m / parent.scale

        best: float | None = None
        for x, y, z in world:
            ox, oy = (np.array([x, y]) - origin[:2]) / parent.scale
            height = grid.sample(float(ox), float(oy), (z - origin[2]) / parent.scale, slack)
            if height is None:
                continue
            gap = float(z) - (origin[2] + parent.scale * height)
            if best is None or gap < best:
                best = gap
        return best

    def nearest_support_xy(
        self, parent: SceneObject, low: np.ndarray, high: np.ndarray, base: float, slack: float
    ):
        """World XY of the nearest point on `parent` that could hold this child up.

        None when the parent has no grid, or nothing in it at a plausible height.
        The child's own footprint centre is returned when it is already over
        something, so a caller can treat "no correction needed" and "pull here" the
        same way.
        """
        grid = self._grids.get(parent.object_id)
        if grid is None:
            return None
        centre_xy = (np.asarray(low[:2]) + np.asarray(high[:2])) / 2.0
        position = np.asarray(parent.position_m, dtype=float)
        oriented = (centre_xy - position[:2]) / parent.scale
        oriented_base = (base - position[2]) / parent.scale
        found = grid.nearest_xy(
            float(oriented[0]), float(oriented[1]), oriented_base, slack / parent.scale
        )
        if found is None:
            return None
        return position[:2] + parent.scale * found


def build(
    graph: SceneGraph, settings: Settings, candidates: set[str] | None = None
) -> SupportHeights:
    """Grids for every object that something in this graph rests on.

    Only those: a grid costs a mesh load and a few thousand rays, and an object
    nothing is resting on will never be asked.
    """
    wanted = (
        candidates
        if candidates is not None
        else {obj.supported_by for obj in graph.objects if obj.supported_by}
    )
    # Bases and undersides for *every* object; grids only for the ones being rested
    # on. An object can of course be both.
    #
    # This used to be only the objects that declared a parent, which was right while
    # the gap was measured against that parent — a floor-standing object had nothing
    # to measure against but the floor plane. `contact` changed that: it asks what is
    # beneath an object without consulting any declaration, so it needs the underside
    # of everything. Measured on `room2.png`, the armchair silently kept the old
    # bounding-box arithmetic and reported +21.2 mm on the floor while the side table
    # beside it — which happened to declare the rug — measured +0.0 mm on the rug.
    # Same geometry, same rug, and the only difference was one field.
    needs_base = {obj.object_id for obj in graph.objects}

    grids: dict[str, _Grid] = {}
    bases: dict[str, float] = {}
    undersides: dict[str, np.ndarray] = {}
    for obj in graph.objects:
        if obj.object_id not in wanted and obj.object_id not in needs_base:
            continue
        mesh = _collision_mesh(obj)
        if mesh is None:
            continue
        if obj.object_id in needs_base:
            bases[obj.object_id] = float(mesh.bounds[0][2])
            # The child's own underside: the same downward raster, read from the
            # bottom of each column instead of the top. One point per occupied cell,
            # so a pedestal contributes its feet and a book its whole bottom face.
            own = _rasterise(mesh, settings.support_grid_cell_m, settings.support_grid_max_cells)
            if own is not None and len(own.cell_xy):
                points = np.column_stack([own.cell_xy, own.cell_lo])
                # Only the lowest band: contact happens at the feet, and a pedestal's
                # raster also covers the underside of its top half a metre higher.
                band = points[:, 2] <= points[:, 2].min() + settings.support_burial_slack_m
                points = points[band]
                if len(points) > MAX_CONTACT_SAMPLES:
                    # The lowest point is kept explicitly, not left to the stride.
                    # The gap is a *minimum* over these samples, so dropping the
                    # argmin does not lose resolution, it changes the answer.
                    lowest = points[int(np.argmin(points[:, 2]))]
                    points = points[:: max(1, len(points) // MAX_CONTACT_SAMPLES)]
                    if not np.any(np.all(points == lowest, axis=1)):
                        points = np.vstack([lowest, points])
                undersides[obj.object_id] = points
        if obj.object_id in wanted:
            grid = _rasterise(mesh, settings.support_grid_cell_m, settings.support_grid_max_cells)
            if grid is not None:
                grids[obj.object_id] = grid
    log.info("support: %d height grids, %d bases", len(grids), len(bases))
    return SupportHeights(
        grids,
        bases,
        settings.support_burial_slack_m,
        undersides,
    )
