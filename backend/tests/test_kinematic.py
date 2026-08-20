"""The kinematic certification axis.

`test_deep_penetration_is_detected` is the load-bearing one. MuJoCo's contact
array does not report box-box overlap once one box is engulfed by the other, so a
sweep built on `data.contact` passes the very worst objects. That test pins the
behaviour that replaced it.
"""

import mujoco
import pytest

from app.certify import kinematic
from app.export import mjcf
from app.schemas import JointType
from tests.fixtures import scenes

TOL = 0.002


def _by_joint(checks):
    return {c.joint_id: c for c in checks}


# --- the happy path -----------------------------------------------------------


def test_a_sound_scene_sweeps_clean():
    checks = kinematic.run(scenes.kitchen(), TOL)
    assert len(checks) == 2
    assert all(c.passed for c in checks)
    assert all(c.max_parent_penetration_m == 0.0 for c in checks)
    assert all(not c.sibling_contacts and not c.world_contact for c in checks)


def test_a_clean_joint_reports_no_feasible_sub_range():
    """It already has its full range. Handing a repair a "feasible" range it
    already possesses invites it to trim something that was never wrong."""
    for check in kinematic.run(scenes.kitchen(), TOL):
        assert check.feasible_limits is None
        assert check.blocked_at_q is None


def test_every_step_is_recorded():
    check = kinematic.run(scenes.kitchen(), TOL, steps=17)[0]
    assert check.steps == 17


# --- failures -----------------------------------------------------------------


def test_deep_penetration_is_detected():
    """A drawer longer than its carcass overlaps the back panel by 160 mm when
    fully closed. MuJoCo's contact array reports *nothing* for that — it only sees
    the shallower overlap later in the sweep — so a validator reading `ncon` would
    call this object fine. It is not fine."""
    graph = scenes.cabinet_with_overlong_drawer()

    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, "fixture no longer reproduces the engulfed-box blind spot"

    check = _by_joint(kinematic.run(graph, TOL))["drawer_top_slide"]
    assert not check.passed
    assert check.blocked_at_q == pytest.approx(0.0), "it is broken from fully closed"
    assert "back" in check.sibling_contacts


def test_a_blocked_joint_keeps_the_range_that_works():
    """The gap between feasible and fitted limits is range the object genuinely
    has, and a repair that trims more than this is discarding it."""
    check = _by_joint(kinematic.run(scenes.cabinet_door_into_table(), TOL))["door_hinge"]
    assert not check.passed
    assert check.world_contact
    assert check.feasible_limits is not None
    assert check.feasible_limits.lower == pytest.approx(0.0)
    # Swings freely for part of its travel, then meets the table.
    assert 0.3 < check.feasible_limits.upper < 1.5708
    assert check.blocked_at_q > check.feasible_limits.upper


def test_collision_with_another_object_is_world_not_sibling():
    check = _by_joint(kinematic.run(scenes.cabinet_door_into_table(), TOL))["door_hinge"]
    assert check.world_contact
    assert check.sibling_contacts == []


# --- what does not get swept --------------------------------------------------


def test_jointless_objects_produce_no_checks():
    """They belong in Certificate.jointless_objects. A cabinet welded shut is
    trivially collision-free, and crediting it would make the rate a lie."""
    graph = scenes.kitchen()
    graph.objects = [o for o in graph.objects if not o.joints]
    assert kinematic.run(graph, TOL) == []


def test_fixed_joints_are_not_swept():
    graph = scenes.kitchen()
    graph.get("cabinet").joints[0].type = JointType.FIXED
    checks = kinematic.run(graph, TOL)
    assert [c.joint_id for c in checks] == ["drawer_bottom_slide"]


# --- tolerance ----------------------------------------------------------------


def test_tolerance_is_the_threshold_not_a_suggestion():
    """delta is non-zero because mesh discretisation puts sub-millimetre contacts
    on surfaces flush by design. Widening it far enough must forgive a real
    overlap, which is what makes the sensitivity of results to it measurable."""
    graph = scenes.cabinet_with_overlong_drawer()
    assert not _by_joint(kinematic.run(graph, 0.002))["drawer_top_slide"].passed
    assert _by_joint(kinematic.run(graph, 0.5))["drawer_top_slide"].passed


# --- rigid-only scope ---------------------------------------------------------


def test_a_rigid_scene_produces_no_sweeps_at_all():
    """With articulation off every object is jointless, so the kinematic axis has
    nothing to test. That is NOT_APPLICABLE, not a wall of failures — the
    distinction the certificate keeps so a rigid scene is not reported as broken."""
    assert kinematic.run(scenes.rigid_kitchen(), TOL) == []
