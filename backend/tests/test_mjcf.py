"""MJCF emission.

The two that matter most are `test_parent_child_contacts_are_enabled` and
`test_box_size_is_a_half_extent`. Both guard MuJoCo defaults that are wrong for
this project and wrong *silently* — nothing raises, the scene just quietly stops
meaning what it says.
"""

import mujoco
import pytest

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
