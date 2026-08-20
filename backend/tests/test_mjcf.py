"""MJCF emission.

The two tests that matter most here are `test_parent_child_contacts_are_enabled`
and `test_box_size_is_a_half_extent`. Both guard defaults that are wrong for this
project and wrong *silently* — nothing raises, the scene just quietly stops
meaning what it says.
"""

import mujoco
import pytest

from app.export import mjcf
from app.schemas import JointType
from tests.fixtures import scenes

ALL_SCENES = ["kitchen", "cabinet_with_overlong_drawer", "cabinet_door_into_table"]


@pytest.fixture(params=ALL_SCENES)
def graph(request):
    return getattr(scenes, request.param)()


def test_every_fixture_compiles(graph):
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    assert model.nbody > 1
    assert model.ngeom >= len(graph.objects)


def test_parent_child_contacts_are_enabled(graph):
    """MuJoCo filters parent-child contacts by default, and with that default a
    drawer driven clean through the back of its cabinet reports zero contacts —
    so the kinematic axis would pass every articulated object it is ever given."""
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    assert model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT


def test_angles_are_radians():
    """MJCF defaults to degrees. JointLimits are radians, so a 1.57 rad door would
    compile as 1.57 degrees and every sweep would pass without opening anything."""
    graph = scenes.cabinet_door_into_table()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    joint = graph.objects[0].joints[0]
    lower, upper = model.jnt_range[model.joint("cabinet/door_hinge").id]
    assert upper == pytest.approx(joint.limits.upper, abs=1e-6)
    assert lower == pytest.approx(joint.limits.lower, abs=1e-6)


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
    to land on its geometry — otherwise the whole object translates by it."""
    graph = scenes.kitchen()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    root = graph.get("cabinet").root_part
    assert tuple(model.geom("cabinet/bottom/obb").pos) == pytest.approx(root.origin_m)


def test_names_come_from_the_helpers_not_string_surgery():
    """certify maps MuJoCo indices back to schema objects through these, so they
    are contract, not convenience."""
    graph = scenes.kitchen()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    for obj in graph.objects:
        for part in obj.parts:
            assert model.body(mjcf.body_name(obj.object_id, part.part_id)) is not None
        for joint in obj.joints:
            assert model.joint(mjcf.joint_name(obj.object_id, joint.joint_id)) is not None
        assert model.joint(mjcf.free_joint_name(obj.object_id)) is not None


def test_joint_types_map_to_mujoco_kinds():
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(scenes.kitchen()))
    assert model.joint("cabinet/drawer_top_slide").type == mujoco.mjtJoint.mjJNT_SLIDE
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(scenes.cabinet_door_into_table()))
    assert model.joint("cabinet/door_hinge").type == mujoco.mjtJoint.mjJNT_HINGE


def test_fixed_joints_emit_no_mujoco_joint():
    graph = scenes.kitchen()
    cabinet = graph.get("cabinet")
    cabinet.joints[0].type = JointType.FIXED
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    with pytest.raises(KeyError):
        model.joint("cabinet/drawer_top_slide")


def test_write_mjcf_matches_build_xml(tmp_path):
    """Export and certification must read the same bytes, or a scene could pass
    certification and ship as something that behaves differently."""
    graph = scenes.kitchen()
    path = mjcf.write_mjcf(graph, tmp_path)
    assert path.read_text() == mjcf.build_xml(graph)
