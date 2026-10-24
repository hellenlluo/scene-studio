"""The stability certification axis.

`test_settling_destroys_the_evidence_of_interpenetration` is the load-bearing one.
It demonstrates the reason the overlap is sampled at t=0 rather than asserting it
in a comment.
"""

import mujoco
import numpy as np
import pytest

from app.certify import stability
from app.config import get_settings
from app.export import mjcf
from tests.fixtures import scenes


@pytest.fixture
def settings():
    return get_settings()


def _by_object(checks):
    return {c.object_id: c for c in checks}


# --- the happy path -----------------------------------------------------------


def test_a_sound_scene_settles(settings):
    checks = stability.run(scenes.kitchen(), settings)
    assert len(checks) == 3
    assert all(c.passed for c in checks)
    assert all(c.settled for c in checks)
    assert all(c.quiet_duration_s >= settings.settle_quiet_seconds for c in checks)


def test_resting_objects_barely_move(settings):
    """Sub-millimetre, from MuJoCo's contact softness under load. If this starts
    creeping toward the 1 cm threshold, the masses or the contact parameters
    changed and the axis is about to become meaningless."""
    for check in stability.run(scenes.kitchen(), settings):
        assert check.com_displacement_m < 0.001
        assert check.orientation_drift_deg < 0.1
        assert check.initial_penetration_m == 0.0


# --- the two reconstruction errors this axis exists to catch -------------------


def test_a_floating_object_fails_on_displacement_not_penetration(settings):
    """Over-estimated depth. The two signals are independent, and a scene can be
    perfectly non-overlapping and still fall over."""
    check = _by_object(stability.run(scenes.mug_floating_above_table(), settings))["mug"]
    assert not check.passed
    assert check.com_displacement_m == pytest.approx(0.25, abs=0.01)
    assert check.initial_penetration_m == 0.0


def test_a_buried_object_fails_on_penetration(settings):
    """Under-estimated depth, or two objects whose scales disagree."""
    checks = _by_object(stability.run(scenes.mug_sunk_into_table(), settings))
    assert checks["mug"].initial_penetration_m == pytest.approx(0.05, abs=1e-3)
    assert not checks["mug"].passed
    # Symmetric by design: the table really is overlapping something.
    assert checks["table"].initial_penetration_m == pytest.approx(0.05, abs=1e-3)
    assert not checks["table"].passed


def test_an_untouched_object_is_unaffected_by_a_neighbour_failing(settings):
    checks = _by_object(stability.run(scenes.mug_sunk_into_table(), settings))
    assert checks["cabinet"].passed


def test_settling_destroys_the_evidence_of_interpenetration(settings):
    """Why penetration is sampled at t=0.

    The constraint solver's job is to push overlapping bodies apart, and it is
    good at it. Sample after settling and a 5 cm burial reports as clean — the
    scene "passes" because the engine papered over the defect rather than because
    the reconstruction was right.
    """
    graph = scenes.mug_sunk_into_table()
    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    data = mujoco.MjData(model)

    mug = model.geom("mug/body/obb").id
    table = model.geom("table/top/obb").id

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    at_start = -mujoco.mj_geomDistance(model, data, mug, table, 1.0, None)

    for _ in range(round(settings.settle_seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    after = -mujoco.mj_geomDistance(model, data, mug, table, 1.0, None)

    assert at_start == pytest.approx(0.05, abs=1e-3)
    assert after < 0.002, "the solver ejected it; measuring here would report a pass"

    # And the axis reports the honest number.
    check = _by_object(stability.run(graph, settings))["mug"]
    assert check.initial_penetration_m == pytest.approx(at_start, abs=1e-3)


# --- thresholds ---------------------------------------------------------------


def test_thresholds_come_from_settings(settings):
    """Both directions on the *same* object, which has to be one that moves.

    The tight case used to be asserted against a resting fixture table, on the
    assumption that anything settled still drifts by more than a micron. It does
    not: once `mjcf` stiffened the contact defaults, a correctly-placed rigid body
    settles to 6e-8 m, so a 1e-6 threshold passes it and the test failed for the
    right reason. The floating mug falls 25 cm, which is a displacement no contact
    model is going to argue with.
    """
    graph = scenes.mug_floating_above_table()

    loose = settings.model_copy(update={"max_com_displacement_m": 1.0})
    assert _by_object(stability.run(graph, loose))["mug"].passed

    tight = settings.model_copy(update={"max_com_displacement_m": 1e-6})
    assert not _by_object(stability.run(graph, tight))["mug"].passed


def test_settle_duration_comes_from_settings(settings):
    """Barely any time to fall, so pose alone looks safe but motion fails it."""
    brief = settings.model_copy(
        update={
            "settle_seconds": 0.01,
            "max_settle_seconds": 0.01,
            "settle_quiet_seconds": 0.005,
        }
    )
    check = _by_object(stability.run(scenes.mug_floating_above_table(), brief))["mug"]
    assert check.com_displacement_m < 0.01
    assert check.terminal_linear_speed_m_s > brief.max_linear_speed_m_s
    assert not check.settled
    assert not check.passed


def test_a_settled_check_records_terminal_velocity_evidence(settings):
    check = _by_object(stability.run(scenes.kitchen(), settings))["mug"]
    assert check.terminal_linear_speed_m_s <= settings.max_linear_speed_m_s
    assert check.terminal_angular_speed_deg_s <= settings.max_angular_speed_deg_s
    assert check.simulation_time_s == pytest.approx(settings.settle_seconds, abs=0.01)


def test_alternating_contact_velocity_is_filtered_when_pose_stays_bounded(
    settings, monkeypatch
):
    """Raw qvel is deliberately large on every step but alternates direction.

    This models contact chatter: the instantaneous solver velocity exceeds the
    rate limit by 4x while the pose only vibrates over a tiny bounded interval.
    The old per-step gate failed it forever; the filtered pose rate must pass.
    """
    direction = 1.0

    def chatter_step(mujoco_module, model, data, _guard):
        nonlocal direction
        mujoco_module.mj_step(model, data)
        joint = model.joint(mjcf.free_joint_name("mug"))
        dof = int(model.jnt_dofadr[joint.id])
        data.qvel[dof] = direction * 0.02  # 20 mm/s, alternating every 2 ms
        direction *= -1.0

    monkeypatch.setattr(stability, "checked_steps", chatter_step)
    brief = settings.model_copy(
        update={
            "settle_seconds": 0.1,
            "max_settle_seconds": 0.2,
            "settle_quiet_seconds": 0.02,
        }
    )
    check = _by_object(stability.run(scenes.kitchen(), brief))["mug"]

    assert check.com_displacement_m < 0.001
    assert check.terminal_linear_speed_m_s < brief.max_linear_speed_m_s
    assert check.settled
    assert check.passed


# --- gravity-relative tilt ----------------------------------------------------


def test_opposite_quaternions_have_the_same_tilt():
    """q and -q describe the same orientation."""
    q = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    assert stability._tilt_angle_deg(q, -q) == pytest.approx(0.0, abs=1e-6)
    assert stability._tilt_angle_deg(q, q) == pytest.approx(0.0, abs=1e-6)


def test_a_slightly_denormalised_quaternion_does_not_invent_tilt():
    sloppy = np.array([0.7071, 0.0, 0.0, 0.7071])  # norm 0.99999
    assert stability._tilt_angle_deg(sloppy, sloppy) == pytest.approx(0.0, abs=1e-6)


def test_world_yaw_is_not_tipping():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    yaw_quarter_turn = np.array([0.7071068, 0.0, 0.0, 0.7071068])
    assert stability._tilt_angle_deg(identity, yaw_quarter_turn) == pytest.approx(
        0.0, abs=1e-6
    )


def test_a_right_angle_tip_reads_as_ninety_degrees():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    x_quarter_turn = np.array([0.7071068, 0.7071068, 0.0, 0.0])
    assert stability._tilt_angle_deg(identity, x_quarter_turn) == pytest.approx(
        90.0, abs=1e-3
    )
