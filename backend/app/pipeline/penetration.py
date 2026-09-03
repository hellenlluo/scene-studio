"""How far two objects actually overlap, measured on the collision decomposition.

Stage 6's `E_pen` needs one number per object pair: how deep they interpenetrate,
zero when they merely touch. It used to take the axis-aligned bounding boxes, and
on real output that is not an approximation of the answer, it is a different
question. Measured on `room.png` after everything else was fixed:

    solver E_pen (AABB vs AABB)   penalises sofa vs rug by 14.5 mm
    the decomposition             no intersection between any pair, anywhere
    MuJoCo at t=0                 no sofa/rug contact at all

A sofa's bounding box is a solid brick from its feet to the top of its backrest,
so a rug lying under it is "inside" the sofa by whatever the rug is thick. The
solver spent every iteration pushing the two apart to resolve a collision that
does not exist, while the certifier — which uses the decomposition — reported the
scene as clean. Same class of bug as the support term reading a box top: the
solver and the certifier disagreeing about what shape an object is.

**Convex pieces, because MuJoCo has no other kind.** A mesh geom in MJCF collides
as its convex hull whether or not you want it to, which is why stage 9 runs CoACD
at all. So the honest target is not "use the raw mesh" — no engine here can — but
"use the same convex pieces MuJoCo will", and make those hug the mesh. Since the
CoACD preprocessing fix they do, at 1.00x of the visual mesh's extent.

**`fcl.Convex`, not a triangle-mesh query.** FCL's BVH mesh-mesh path returns a
contact per intersecting *triangle pair*, and the depth on those is the overlap of
two triangles rather than the distance the objects must move to separate — two
boxes 0.1 apart reported 1.0. `fcl.Convex` runs GJK/EPA and returns the minimum
translation, which is the quantity the residual wants. Measured exact on boxes at
several offsets, and 5000 queries take 8 ms, so this is not the expensive part of
a residual evaluation.

**Built per round, exact under translation, approximate under scale.** Object scale
is baked into the geometry, because FCL cannot rescale a collision object at query
time; position is applied per query and is exact. Scale is a stage-6 variable, so
within a Gauss-Newton pass the pieces are slightly the wrong size — by however
much that pass moves a scale, which is a few percent — and the round loop rebuilds
them. That is the same bargain the physics feedback already makes, and the
alternative is rebuilding the hulls inside the residual, which is the one thing
that would make this slow.
"""

import logging
from itertools import combinations

import numpy as np
import trimesh

from app.config import Settings
from app.export.gltf import geometry_for
from app.geometry import quat_to_matrix
from app.schemas import SceneGraph, SceneObject

__all__ = ["Penetrations", "build"]

log = logging.getLogger(__name__)


def _convex_object(mesh: trimesh.Trimesh):
    """One FCL convex collision object from a piece of the decomposition.

    Hulled again on the way in: CoACD's pieces are convex by construction, but
    `fcl.Convex` trusts the face list it is handed and a stray non-convex triangle
    makes EPA return nonsense rather than raise.
    """
    import fcl

    hull = mesh.convex_hull
    faces = np.hstack(
        [np.full((len(hull.faces), 1), 3, np.int64), np.asarray(hull.faces, np.int64)]
    ).ravel()
    geometry = fcl.Convex(np.asarray(hull.vertices, dtype=float), len(hull.faces), faces)
    return fcl.CollisionObject(geometry, fcl.Transform())


def _pieces(obj: SceneObject) -> list[trimesh.Trimesh]:
    """Collision pieces in a frame carrying orientation and scale but not position.

    Position is left out because it is applied per query, which is what makes the
    lookup exact under the variable the solver moves most.
    """
    out: list[trimesh.Trimesh] = []
    rotation = np.eye(4)
    rotation[:3, :3] = quat_to_matrix(obj.orientation)

    for part in obj.parts:
        offset = np.eye(4)
        offset[:3, 3] = np.asarray(part.origin_m, dtype=float)
        scale = np.eye(4)
        scale[:3, :3] *= obj.scale
        transform = rotation @ offset @ scale

        sources = []
        if part.collision_mesh_paths:
            for path in part.collision_mesh_paths:
                try:
                    loaded = trimesh.load(path, force="mesh")
                except Exception as exc:
                    log.warning("penetration: cannot read %s (%s)", path, exc)
                    continue
                if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
                    sources.append(loaded)
        else:
            # No decomposition yet — the visual mesh, or the OBB box behind it. The
            # box is the old behaviour, which is the right thing to degrade to.
            sources.append(geometry_for(part))

        for source in sources:
            piece = source.copy()
            piece.apply_transform(transform)
            out.append(piece)
    return out


class Penetrations:
    """Pairwise penetration depth, queried with the objects' current positions."""

    def __init__(self, objects: dict[str, list]):
        self._objects = objects

    def between(self, first: SceneObject, second: SceneObject) -> float:
        """Deepest overlap between any piece of one and any piece of the other.

        The deepest rather than the sum: this feeds a residual that has to reach
        zero, and the depth of the worst contact is what separating them costs.
        Summing would also double-count a single overlap that happens to straddle
        two pieces of the same decomposition.
        """
        import fcl

        left = self._objects.get(first.object_id)
        right = self._objects.get(second.object_id)
        if not left or not right:
            return 0.0

        a_pos = np.asarray(first.position_m, dtype=float)
        b_pos = np.asarray(second.position_m, dtype=float)
        worst = 0.0
        for a in left:
            a.setTranslation(a_pos)
            for b in right:
                b.setTranslation(b_pos)
                result = fcl.CollisionResult()
                fcl.collide(a, b, fcl.CollisionRequest(enable_contact=True), result)
                for contact in result.contacts:
                    worst = max(worst, float(contact.penetration_depth))
        return worst


def build(graph: SceneGraph, settings: Settings) -> Penetrations:
    """FCL objects for every object in the graph, at its current scale."""
    objects: dict[str, list] = {}
    for obj in graph.objects:
        pieces = [_convex_object(piece) for piece in _pieces(obj)]
        if pieces:
            objects[obj.object_id] = pieces
    log.info("penetration: %d objects prepared", len(objects))
    return Penetrations(objects)


def pairs(graph: SceneGraph) -> list[tuple[str, str]]:
    """Object pairs worth testing: everything except a declared support contact.

    A support contact is wanted, not a violation — the support term is already
    driving those two surfaces together, and penalising the same contact here would
    have the solver fighting itself.
    """
    supports = {(o.object_id, o.supported_by) for o in graph.objects if o.supported_by}
    return [
        (a.object_id, b.object_id)
        for a, b in combinations(graph.objects, 2)
        if (a.object_id, b.object_id) not in supports and (b.object_id, a.object_id) not in supports
    ]
