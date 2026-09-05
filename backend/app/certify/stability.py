"""Stability certification: settle the scene under gravity and see what moved.

An object passes when, after `settle_seconds` of simulated time, its centre of
mass has barely moved, its orientation has barely drifted, and it was not
overlapping anything to begin with.

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

__all__ = ["run"]


def _quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation taking `a` to `b`.

    Two subtleties, both of which inflate the reported drift of an object that
    never moved:

    * `abs` on the dot product, because q and -q are the same rotation. Without it
      a body sitting still can report ~360 degrees.
    * Normalise first. `2*arccos(dot)` has an infinite derivative at dot = 1,
      which is exactly where a resting object sits, so a quaternion off unit
      length by 1e-5 reads as 0.7 degrees of drift — a third of the 2 degree
      budget, from rounding alone.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = min(1.0, abs(float(np.dot(a, b))))
    return float(np.degrees(2.0 * np.arccos(dot)))


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


def run(graph: SceneGraph, settings: Settings) -> list[StabilityCheck]:
    import mujoco

    model = mjcf.load_model(graph)
    data = mujoco.MjData(model)

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

    steps = round(settings.settle_seconds / model.opt.timestep)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    # mj_step leaves the derived quantities one state behind for our purposes.
    mujoco.mj_forward(model, data)

    checks: list[StabilityCheck] = []
    for obj in graph.objects:
        body = roots[obj.object_id]
        displacement = float(np.linalg.norm(data.subtree_com[body] - start_com[obj.object_id]))
        drift = _quat_angle_deg(start_quat[obj.object_id], data.xquat[body])
        overlap, against = penetration[obj.object_id]

        checks.append(
            StabilityCheck(
                object_id=obj.object_id,
                com_displacement_m=displacement,
                orientation_drift_deg=drift,
                initial_penetration_m=overlap,
                penetration_against=against,
                passed=(
                    displacement <= settings.max_com_displacement_m
                    and drift <= settings.max_orientation_drift_deg
                    and overlap <= settings.max_penetration_m
                ),
            )
        )
    return checks
