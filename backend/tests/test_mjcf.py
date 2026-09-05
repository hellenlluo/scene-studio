"""MJCF emission.

The two that matter most are `test_parent_child_contacts_are_enabled` and
`test_box_size_is_a_half_extent`. Both guard MuJoCo defaults that are wrong for
this project and wrong *silently* — nothing raises, the scene just quietly stops
meaning what it says.
"""

from pathlib import Path

import mujoco
import pytest
import trimesh

from app.export import mjcf
from app.schemas import (
    AssetFrame,
    DimensionPrior,
    ObjectLabel,
    PartGeometry,
    SceneGraph,
    SceneObject,
)
from tests.fixtures import scenes

ALL_SCENES = ["kitchen", "mug_floating_above_table", "mug_sunk_into_table"]


@pytest.fixture(params=ALL_SCENES)
def graph(request):
    return getattr(scenes, request.param)()


def test_every_fixture_compiles(graph):
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    assert model.nbody > 1
    assert model.ngeom >= len(graph.objects)


def test_parent_child_contacts_are_enabled(graph):
    """MuJoCo excludes contacts between a parent body and its child by default,
    which would hide overlap between the rigidly-attached pieces of one object."""
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT


def test_box_size_is_a_half_extent():
    """MuJoCo's box `size` is a half-extent. Treating dims_m as the size directly
    doubles every object, and nothing complains — things just stop fitting."""
    graph = scenes.kitchen()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    table = graph.get("table").root_part
    size = model.geom("table/top/obb").size
    assert tuple(size * 2) == pytest.approx(table.dims_m)


def test_object_scale_multiplies_part_extents():
    graph = scenes.kitchen()
    graph.get("table").scale = 2.0
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    assert tuple(model.geom("table/top/obb").size * 2) == pytest.approx((2.4, 1.5, 1.5))


def test_root_part_origin_is_not_dropped():
    """The root body carries the object frame, so the root *part's* own offset has
    to land on its geometry — otherwise the whole object translates by it.

    Built inline rather than from a fixture: every fixture object happens to have
    a root at the origin, so this would pass vacuously against any of them.
    """
    root = PartGeometry(
        part_id="base", name="base", dims_m=(0.4, 0.4, 0.1), origin_m=(0.0, 0.0, -0.45)
    )
    child = PartGeometry(
        part_id="shade",
        name="shade",
        parent_part_id="base",
        dims_m=(0.3,) * 3,
        origin_m=(0, 0, 0.2),
    )
    obj = SceneObject(
        object_id="lamp",
        label=ObjectLabel(
            object_id="lamp",
            category="lamp",
            prior=DimensionPrior(dims_m=(0.4, 0.4, 1.0), sigma_m=(0.1,) * 3),
        ),
        frame=AssetFrame(source="test"),
        parts=[child, root],
        position_m=(1.0, 2.0, 0.5),
        scale=2.0,
    )
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(SceneGraph(objects=[obj])))

    # The root body sits at the object position; its geometry carries the offset,
    # scaled.
    assert tuple(model.body("lamp/base").pos) == pytest.approx((1.0, 2.0, 0.5))
    assert tuple(model.geom("lamp/base/obb").pos) == pytest.approx((0.0, 0.0, -0.9))
    # Children are placed relative to the root part's origin, also scaled.
    assert tuple(model.body("lamp/shade").pos) == pytest.approx((0.0, 0.0, 1.3))


def test_names_come_from_the_helpers_not_string_surgery():
    """certify maps MuJoCo indices back to schema objects through these, so they
    are contract, not convenience."""
    graph = scenes.kitchen()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    for obj in graph.objects:
        for part in obj.parts:
            assert model.body(mjcf.body_name(obj.object_id, part.part_id)) is not None
        assert model.joint(mjcf.free_joint_name(obj.object_id)) is not None


def test_write_mjcf_matches_build_xml(tmp_path):
    """Export and certification must read the same bytes, or a scene could pass
    certification and ship as something that behaves differently."""
    graph = scenes.kitchen()
    path = mjcf.write_mjcf(graph, tmp_path)
    assert path.read_text() == mjcf.build_xml(graph)


# --- one MJCF, two consumers ---------------------------------------------------


def _proxy_object(tmp_path, object_id="obj"):
    """An object whose collision geometry is two real files on disk."""
    paths = []
    for index in range(2):
        mesh = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
        mesh.apply_translation((index * 0.3, 0.0, 0.0))
        path = tmp_path / f"{object_id}_body_{index:02d}.collision.obj"
        path.write_text(mesh.export(file_type="obj"))
        paths.append(str(path))
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(object_id=object_id, category="other"),
        frame=AssetFrame(source="test"),
        parts=[
            PartGeometry(
                part_id="body", name="body", dims_m=(0.5, 0.2, 0.2), collision_mesh_paths=paths
            )
        ],
        position_m=(0.0, 0.0, 0.1),
    )


def test_mesh_assets_are_named_by_bare_filename(tmp_path):
    """Not by path. A browser cannot open an absolute path, and the frontend runs
    the same engine over the same MJCF — that is what makes it authoritative rather
    than an approximation of the server."""
    graph = SceneGraph(objects=[_proxy_object(tmp_path)])
    xml = mjcf.build_xml(graph)

    assert 'file="obj_body_00.collision.obj"' in xml
    assert str(tmp_path) not in xml, "no absolute path may survive into the MJCF"


def test_mesh_files_lists_what_the_xml_names(tmp_path):
    graph = SceneGraph(objects=[_proxy_object(tmp_path)])
    files = mjcf.mesh_files(graph)

    assert set(files) == {"obj_body_00.collision.obj", "obj_body_01.collision.obj"}
    for name, path in files.items():
        assert Path(path).name == name
        assert Path(path).exists()


def test_load_model_compiles_with_the_meshes_supplied(tmp_path):
    graph = SceneGraph(objects=[_proxy_object(tmp_path)])
    model = mjcf.load_model(graph)

    assert model.nmesh == 2
    assert model.ngeom >= 3  # two pieces plus the floor


def test_the_xml_no_longer_depends_on_where_the_files_are(tmp_path, monkeypatch):
    """The portability this buys, asserted rather than assumed.

    Compiling the string directly is what a consumer without the virtual filesystem
    would do, and it now fails — which is the point: the bytes travel with the
    model instead of being looked up on a filesystem the consumer may not share.
    """
    graph = SceneGraph(objects=[_proxy_object(tmp_path)])
    xml = mjcf.build_xml(graph)

    monkeypatch.chdir(tmp_path.parent)
    with pytest.raises(ValueError):
        mujoco.MjModel.from_xml_string(xml)
    assert mjcf.load_model(graph).nmesh == 2, "and it compiles when they are supplied"


def test_a_missing_proxy_costs_its_geom_not_the_scene(tmp_path):
    """Degrade, do not raise. An asset that went astray is a worse scene, not no
    scene — the same rule stage 4 and the glTF exporter both follow."""
    obj = _proxy_object(tmp_path)
    Path(obj.parts[0].collision_mesh_paths[0]).unlink()

    model = mjcf.load_model(SceneGraph(objects=[obj]))
    assert model.nmesh == 1, "the surviving piece is still there"
