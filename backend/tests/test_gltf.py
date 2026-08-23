"""glTF emission for the viewer.

The load-bearing test is `test_bounds_agree_with_the_geometry_module`: it checks
trimesh's transform stack against the independent corner math in `app.geometry`.
Two implementations of the same thing agreeing is much stronger evidence than
either matching a number I typed in.
"""

import json
import math
import struct

import numpy as np
import pytest
import trimesh

from app.export import gltf, mjcf
from app.geometry import world_aabb
from tests.fixtures import scenes


def _gltf_json(blob: bytes) -> dict:
    """The JSON chunk of a GLB: 12-byte header, then a chunk header, then JSON."""
    length = struct.unpack("<I", blob[12:16])[0]
    return json.loads(blob[20 : 20 + length])


def _scene_space_bounds(scene, node: str):
    """Node bounds back in scene coordinates, undoing the root transform.

    The root carries the Y-up conversion *and* the horizontal recentring, so both have
    to come off. Taking the root's own matrix rather than a hard-coded constant keeps
    this honest if more presentation ends up on that node.
    """
    root = scene.graph.get(frame_to=gltf.ROOT_NODE, frame_from="world")[0]
    transform, geometry_name = scene.graph[node]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(np.linalg.inv(root) @ transform)
    return mesh.bounds


# --- structure ----------------------------------------------------------------


def test_one_node_per_part():
    scene = gltf.build_scene(scenes.kitchen())
    assert sorted(scene.graph.nodes_geometry) == ["cabinet/body", "mug/body", "table/top"]


def test_node_names_match_the_physics_export():
    """The viewer highlights a failing object by looking up the node the
    certificate names. If the two exports disagreed the red channel would point at
    nothing."""
    graph = scenes.kitchen()
    scene = gltf.build_scene(graph)
    for obj in graph.objects:
        for part in obj.parts:
            assert mjcf.body_name(obj.object_id, part.part_id) in scene.graph.nodes_geometry


def test_the_floor_is_not_exported():
    """MJCF's floor is an infinite plane; a finite box would misrepresent its
    extent. The frontend draws its own grid at floor_height_m."""
    scene = gltf.build_scene(scenes.kitchen())
    assert not any("floor" in n for n in scene.graph.nodes_geometry)


# --- placement ----------------------------------------------------------------


def test_bounds_agree_with_the_geometry_module():
    """trimesh's transform stack against the corner math in app.geometry. These
    are independent implementations and both are load-bearing — one places the
    viewer's meshes, the other decides whether a scene certifies."""
    graph = scenes.kitchen()
    scene = gltf.build_scene(graph)
    for obj in graph.objects:
        expected_low, expected_high = world_aabb(obj)
        low, high = _scene_space_bounds(scene, mjcf.body_name(obj.object_id, obj.root_part.part_id))
        assert low == pytest.approx(expected_low, abs=1e-9)
        assert high == pytest.approx(expected_high, abs=1e-9)


def test_object_scale_is_applied():
    graph = scenes.kitchen()
    graph.get("mug").scale = 2.0
    scene = gltf.build_scene(graph)
    low, high = _scene_space_bounds(scene, "mug/body")
    assert tuple(high - low) == pytest.approx((0.18, 0.18, 0.20))


def test_object_rotation_is_applied():
    graph = scenes.kitchen()
    # 90 degrees about x swaps the mug's y and z extents.
    graph.get("mug").orientation = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
    scene = gltf.build_scene(graph)
    low, high = _scene_space_bounds(scene, "mug/body")
    assert tuple(high - low) == pytest.approx((0.09, 0.10, 0.09), abs=1e-9)


# --- the axis convention ------------------------------------------------------


def test_the_root_node_carries_the_y_up_conversion():
    """glTF is +Y up, this project is +Z up. The file has to be spec-correct or it
    opens on its side in every standard viewer."""
    blob = gltf.build_scene(scenes.kitchen()).export(file_type="glb")
    document = _gltf_json(blob)

    roots = document["scenes"][0]["nodes"]
    assert len(roots) == 1
    root = document["nodes"][roots[0]]
    assert root["name"] == gltf.ROOT_NODE

    matrix = np.asarray(root["matrix"]).reshape(4, 4).T
    # Rotation only: the root also carries the horizontal recentring translation.
    assert matrix[:3, :3] == pytest.approx(gltf._Z_UP_TO_Y_UP[:3, :3], abs=1e-9)


def test_export_does_not_move_objects():
    """The exporter is a change of representation, not of coordinates: the numbers in
    the GLB have to be the numbers in SceneSpec, or a gizmo reading a world position
    off the rendered object writes back a value shifted by the difference."""
    graph = scenes.kitchen()
    scene = gltf.build_scene(graph)
    names = [mjcf.body_name(o.object_id, o.root_part.part_id) for o in graph.objects]
    offsets = [
        _scene_space_bounds(scene, name)[0] - world_aabb(o)[0]
        for name, o in zip(names, graph.objects, strict=True)
    ]
    for offset in offsets:
        assert offset == pytest.approx(np.zeros(3), abs=1e-9)


def test_child_transforms_stay_in_scene_coordinates():
    """The point of putting the conversion on a root node rather than baking it:
    the frontend positions gizmos and reads back drags using SceneSpec numbers, so
    a second coordinate system in the file would mean converting on every
    interaction."""
    graph = scenes.kitchen()
    blob = gltf.build_scene(graph).export(file_type="glb")
    document = _gltf_json(blob)

    by_name = {n.get("name"): n for n in document["nodes"]}
    matrix = np.asarray(by_name["mug/body"]["matrix"]).reshape(4, 4).T
    assert tuple(matrix[:3, 3]) == pytest.approx(graph.get("mug").position_m)


def test_the_exported_file_is_upright():
    """Composed through the root, the tabletop should be 0.75 m along +Y."""
    scene = gltf.build_scene(scenes.kitchen())
    transform, geometry_name = scene.graph["table/top"]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(transform)
    assert mesh.bounds[1][1] == pytest.approx(0.75)


# --- files --------------------------------------------------------------------


def test_write_gltf_produces_a_loadable_glb(tmp_path):
    path = gltf.write_gltf(scenes.kitchen(), tmp_path)
    assert path.name == "scene.glb"
    assert path.stat().st_size > 0

    reloaded = trimesh.load(path)
    assert len(reloaded.geometry) == 3


def test_a_missing_visual_mesh_falls_back_to_its_box(tmp_path):
    """An export that dies because one asset went astray is worse than one that
    ships a box in its place — and the proxy tier already records that the
    geometry is approximate."""
    graph = scenes.kitchen()
    graph.get("mug").root_part.visual_mesh_path = str(tmp_path / "does-not-exist.obj")

    scene = gltf.build_scene(graph)
    low, high = _scene_space_bounds(scene, "mug/body")
    assert tuple(high - low) == pytest.approx((0.09, 0.09, 0.10))
