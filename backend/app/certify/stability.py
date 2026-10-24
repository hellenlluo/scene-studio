"""Stability certification: release the scene under gravity and require it to rest.

An object passes when its centre of mass barely moved, its orientation barely
drifted, it was not overlapping anything to begin with, and its linear and
filtered COM and tilt rates remain low for a continuous quiet interval.  Rates
come from pose change over a half-second window, not MuJoCo's instantaneous
velocity.  Contact impulses can report centimetres per second for one 2 ms step
while a visibly resting body vibrates by a fraction of a millimetre; treating each
impulse as renewed motion falsely failed the `room` rug and side table.

Orientation means **tilt relative to gravity**, not arbitrary quaternion change.
A mug may rotate a few degrees around world-up without becoming less stable.  The
old total-angle measurement called that yaw a tip even though its gravity-relative
tilt changed by only 0.21 degrees.

The third of those is the one with a trap in it. **Interpenetration is measured
at t=0, before a single step.** The constraint solver's whole job is to push
overlapping bodies apart, so it does — and within a few milliseconds the overlap
that revealed a reconstruction error is gone. Measure after settling and a scene
that started 4 cm inside the floor reports a clean 0.0, having "passed" by virtue
of the engine papering over the defect. The displacement caused by that shove
still shows up in `com_displacement_m`, so the object usually fails anyway, but it
fails for the wrong reason and the diagnostic is lost.

Overlap is read from `mj_geomDistance` rather than `data.contact`, because
MuJoCo's box-box collider stops reporting a contact once one box is engulfed by
the other. Measured: a box overlapping another by 160 mm reports `ncon=0`, while
a far shallower 38 mm overlap between the same pair is detected. An object buried
inside another is exactly the reconstruction failure this axis exists to catch, so
reading the contact array would go quiet on the worst cases.
"""

import numpy as np

from app.config import Settings
from app.export import mjcf
from app.schemas import SceneGraph, StabilityCheck
from app.simulation import SimulationGuard, checked_steps

__all__ = ["run"]


def _gravity_in_body(quaternion: np.ndarray) -> np.ndarray:
    """World-up expressed in body coordinates for a MuJoCo wxyz quaternion."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    # Third row of the body-to-world rotation matrix, equivalently R.T @ world_up.
    return np.asarray(
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]
    )


def _tilt_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Gravity-relative tilt between two poses, deliberately ignoring world yaw.

    Expressing gravity in body coordinates makes a world-up rotation disappear:
    ``(Rz @ R).T @ z == R.T @ z``.  Normalising before the dot product also avoids
    inventing tilt from rounded quaternions.
    """
    dot = float(np.dot(_gravity_in_body(a), _gravity_in_body(b)))
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def _object_geoms(model, graph: SceneGraph) -> dict[str, list[int]]:
    geoms: dict[str, list[int]] = {}
    for obj in graph.objects:
        bodies = {model.body(mjcf.body_name(obj.object_id, part.part_id)).id for part in obj.parts}
        geoms[obj.object_id] = [
            g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies
        ]
    return geoms


def _initial_penetration(
    mujoco, model, data, own: list[int], foreign: list[int], distmax: float
) -> tuple[float, str | None]:
    """Deepest overlap between this object and anything that is not part of it,
    and what that overlap is with.

    Contacts *within* an object are excluded: the rigidly-attached pieces of one
    reconstruction touch by construction, and counting that would fail every
    well-modelled object. What matters is an object buried in the floor, or two
    objects occupying the same space because their depths disagreed — and those
    two are different errors, so the counterpart is returned alongside the depth.
    """
    worst = 0.0
    against: int | None = None
    for a in own:
        for b in foreign:
            depth = -mujoco.mj_geomDistance(model, data, a, b, distmax, None)
            if depth > worst:
                worst, against = depth, b
    if worst <= 0.0 or against is None:
        return 0.0, None
    # Geom names are `{object_id}/{part_id}/{index}`; the floor geom is named
    # "floor" and belongs to no object.
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, against) or "floor"
    return worst, name.split("/")[0]


def run(
    graph: SceneGraph,
    settings: Settings,
    *,
    scene_id: str = "unknown",
    stage: str = "certify.stability",
) -> list[StabilityCheck]:
    import mujoco

    model = mjcf.load_model(graph)
    data = mujoco.MjData(model)
    guard = SimulationGuard.for_model(model, graph, scene_id=scene_id, stage=stage)

    roots = {
        obj.object_id: model.body(mjcf.body_name(obj.object_id, obj.root_part.part_id)).id
        for obj in graph.objects
    }
    geoms = _object_geoms(model, graph)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Everything at t=0, captured before the solver has had a chance to move
    # anything or to hide the overlap it is about to resolve.
    penetration: dict[str, tuple[float, str | None]] = {}
    start_com: dict[str, np.ndarray] = {}
    start_quat: dict[str, np.ndarray] = {}
    max_displacement = {obj.object_id: 0.0 for obj in graph.objects}
    max_drift = {obj.object_id: 0.0 for obj in graph.objects}
    # A trailing pose window filters the alternating collision impulses which made
    # raw qvel a poor definition of rest. Each list retains the most recent sample
    # at or before the window boundary plus everything newer.
    pose_history: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    last_above_speed_limit = {obj.object_id: 0.0 for obj in graph.objects}
    linear_speed = {obj.object_id: 0.0 for obj in graph.objects}
    angular_speed = {obj.object_id: 0.0 for obj in graph.objects}
    for obj in graph.objects:
        own = geoms[obj.object_id]
        # Everything else in the scene: other objects, and the floor.
        foreign = [g for g in range(model.ngeom) if g not in own]
        penetration[obj.object_id] = _initial_penetration(
            mujoco, model, data, own, foreign, settings.max_penetration_m
        )
        body = roots[obj.object_id]
        start_com[obj.object_id] = data.subtree_com[body].copy()
        start_quat[obj.object_id] = data.xquat[body].copy()
        pose_history[obj.object_id] = [
            (0.0, start_com[obj.object_id].copy(), start_quat[obj.object_id].copy())
        ]

    # `settle_seconds` is a minimum observation period.  A scene which is already
    # quiet stops there; a scene which is still moving receives time to settle, but
    # no longer than the configured timeout.  Taking the maximum also makes odd but
    # valid test/dev configurations (quiet interval longer than the minimum) behave
    # predictably.
    minimum_seconds = max(settings.settle_seconds, settings.settle_quiet_seconds)
    timeout_seconds = max(minimum_seconds, settings.max_settle_seconds)
    steps = round(timeout_seconds / model.opt.timestep)
    for _ in range(steps):
        checked_steps(mujoco, model, data, guard)
        # The certificate asks whether anything jumps at any point after release,
        # not merely whether it happens to return near its starting pose at the end
        # of the window.  Retaining the trajectory maximum also makes a transient
        # tip visible instead of certifying it because the final frame looks calm.
        for obj in graph.objects:
            body = roots[obj.object_id]
            max_displacement[obj.object_id] = max(
                max_displacement[obj.object_id],
                float(np.linalg.norm(data.subtree_com[body] - start_com[obj.object_id])),
            )
            max_drift[obj.object_id] = max(
                max_drift[obj.object_id],
                _tilt_angle_deg(start_quat[obj.object_id], data.xquat[body]),
            )
            history = pose_history[obj.object_id]
            history.append(
                (float(data.time), data.subtree_com[body].copy(), data.xquat[body].copy())
            )
            cutoff = float(data.time) - settings.settle_quiet_seconds
            while len(history) > 2 and history[1][0] <= cutoff:
                history.pop(0)
            old_time, old_com, old_quat = history[0]
            elapsed = float(data.time) - old_time
            window_ready = elapsed >= settings.settle_quiet_seconds - model.opt.timestep / 2
            if window_ready:
                linear_speed[obj.object_id] = float(
                    np.linalg.norm(data.subtree_com[body] - old_com) / elapsed
                )
                angular_speed[obj.object_id] = (
                    _tilt_angle_deg(old_quat, data.xquat[body]) / elapsed
                )
            if not window_ready or (
                linear_speed[obj.object_id] > settings.max_linear_speed_m_s
                or angular_speed[obj.object_id] > settings.max_angular_speed_deg_s
            ):
                last_above_speed_limit[obj.object_id] = float(data.time)

        if float(data.time) >= minimum_seconds and all(
            float(data.time) - last_above_speed_limit[obj.object_id]
            >= settings.settle_quiet_seconds
            for obj in graph.objects
        ):
            break
    # mj_step leaves the derived quantities one state behind for our purposes.
    mujoco.mj_forward(model, data)

    checks: list[StabilityCheck] = []
    for obj in graph.objects:
        displacement = max_displacement[obj.object_id]
        drift = max_drift[obj.object_id]
        overlap, against = penetration[obj.object_id]
        quiet_duration = max(
            0.0, float(data.time) - last_above_speed_limit[obj.object_id]
        )
        settled = quiet_duration >= settings.settle_quiet_seconds

        checks.append(
            StabilityCheck(
                object_id=obj.object_id,
                com_displacement_m=displacement,
                orientation_drift_deg=drift,
                initial_penetration_m=overlap,
                penetration_against=against,
                terminal_linear_speed_m_s=linear_speed[obj.object_id],
                terminal_angular_speed_deg_s=angular_speed[obj.object_id],
                quiet_duration_s=quiet_duration,
                simulation_time_s=float(data.time),
                settled=settled,
                passed=(
                    displacement <= settings.max_com_displacement_m
                    and drift <= settings.max_orientation_drift_deg
                    and overlap <= settings.max_penetration_m
                    and settled
                ),
            )
        )
    return checks
