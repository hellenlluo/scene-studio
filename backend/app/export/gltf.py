"""glTF emission for the viewer.

Separate from MJCF on purpose. This carries the **visual** tier; MJCF carries the
collision proxies. Conflating them would mean either shipping decomposed hulls to
the browser or certifying against render meshes.

Three decisions worth knowing:

**Node names come from `mjcf.body_name`.** The viewer highlights a failing object
by looking up the node the certificate names, so if the two exports disagreed
about naming the red channel would silently point at nothing. One source of truth,
imported rather than reimplemented.

**Z-up geometry under a Y-up root.** glTF's convention is +Y up and this project is
+Z up. Rather than pick one and lose the other, everything hangs off a
`scene_root` node carrying the conversion: the file opens upright in any standard
viewer, while every child's *local* transform stays in scene coordinates. That
matters because the frontend positions gizmos and reads back drags using
`SceneSpec` numbers, and a second coordinate system would mean converting on every
interaction.

**No baked materials and no floor.** Objects are coloured client-side by
certificate status — red for a certified failure, amber for uncertainty, grey for
stale — so a baked colour would fight the thing the viewer exists to show. The
floor is omitted because MJCF's is an infinite plane and a finite box would be a
lie about its extent; the frontend draws its own grid at `floor_height_m`.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

from app.export.mjcf import body_name
from app.geometry import quat_to_matrix
from app.schemas import PartGeometry, SceneGraph, SceneObject

__all__ = ["ROOT_NODE", "build_scene", "write_gltf"]

log = logging.getLogger(__name__)

ROOT_NODE = "scene_root"

# -90 degrees about X takes +Z up to +Y up.
_Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2.0, [1.0, 0.0, 0.0])


def _node_transform(obj: SceneObject, part: PartGeometry) -> np.ndarray:
    """Part-local geometry to scene coordinates.

    Right to left: scale the part, move it to its origin within the object (also
    scaled), then apply the object's own rotation and position.
    """
    scale = np.eye(4)
    scale[:3, :3] *= obj.scale

    to_part = np.eye(4)
    to_part[:3, 3] = np.asarray(part.origin_m, dtype=float) * obj.scale

    to_world = np.eye(4)
    to_world[:3, :3] = quat_to_matrix(obj.orientation)
    to_world[:3, 3] = np.asarray(obj.position_m, dtype=float)

    return to_world @ to_part @ scale


def _geometry_for(part: PartGeometry) -> trimesh.Trimesh:
    """The part's visual mesh, or a box from its OBB extent.

    A missing file falls back rather than raising: an export that dies because one
    asset went astray is worse than one that ships a box in its place, and the
    proxy tier already records that the geometry is approximate.
    """
    if part.visual_mesh_path:
        path = Path(part.visual_mesh_path)
        if path.exists():
            loaded = trimesh.load(path, force="mesh")
            if isinstance(loaded, trimesh.Trimesh):
                return loaded
        log.warning("visual mesh %s missing for part %s; exporting its box", path, part.part_id)
    return trimesh.creation.box(extents=np.asarray(part.dims_m, dtype=float))


def build_scene(graph: SceneGraph) -> trimesh.Scene:
    """The scene as a trimesh graph. Pure — no filesystem, so tests are cheap."""
    scene = trimesh.Scene()
    scene.graph.update(frame_from="world", frame_to=ROOT_NODE, matrix=_Z_UP_TO_Y_UP)

    for obj in graph.objects:
        for part in obj.parts:
            scene.add_geometry(
                _geometry_for(part),
                node_name=body_name(obj.object_id, part.part_id),
                parent_node_name=ROOT_NODE,
                transform=_node_transform(obj, part),
            )
    return scene


def write_gltf(graph: SceneGraph, out_dir: Path) -> Path:
    """Write `scene.glb`.

    Binary glTF rather than `.gltf` + `.bin`: one file, no external buffer
    references to keep valid across a move, and it is what a browser wants anyway.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scene.glb"
    path.write_bytes(build_scene(graph).export(file_type="glb"))
    return path
