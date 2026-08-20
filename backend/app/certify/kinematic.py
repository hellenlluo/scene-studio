"""Kinematic certification: drive every joint across its range and look for interference.

The axis no prior work has, and the reason it matters is that parameter accuracy
and functional success are different quantities. Three degrees of axis error is
negligible on a 20 cm drawer and nine centimetres of swept displacement at the
far edge of a 1.8 m wardrobe door. AxisErr cannot tell those apart. This can.

The sweep is *kinematic*, not dynamic: set the joint coordinate, run
`mj_kinematics`, measure. Nothing is ever stepped, so the free joints holding each
object in place keep it exactly where the graph put it, and one MJCF serves both
this axis and the stability one.

Two MuJoCo behaviours had to be worked around, and both fail *open* — they make
broken objects look fine, which is the worst possible direction for a validator:

1. **`filterparent`.** MuJoCo excludes contacts between a parent body and its
   child by default, so a drawer driven clean through the back of its cabinet
   reports zero contacts. `app.export.mjcf` disables it. Measured: with the flag
   at its default a 20 cm interpenetration gives `ncon=0`; disabled, four contacts
   at `dist=-0.35`.

2. **`data.contact` misses deep penetration.** MuJoCo's box-box collider does not
   report a contact once one box is entirely engulfed by the other. Measured on
   `cabinet_with_overlong_drawer`: at the closed position the back panel sits
   wholly inside the drawer, overlapping by 160 mm, and `ncon` is **0** — while
   the far shallower 38 mm overlap later in the same sweep is detected. Reading
   `data.contact` would therefore miss exactly the failures that matter most, the
   "60 cm drawer in a 40 cm cabinet" case this project uses as its own example.

So the sweep does not read `data.contact` at all. It queries `mj_geomDistance`
over the pairs it cares about, which returns a true signed distance through deep
penetration (-0.160 where the contact array said nothing) and clamps only on the
positive side, so `distmax` can stay tight for speed. Roughly 0.5 us per query.
"""

import numpy as np

from app.export import mjcf
from app.schemas import JointLimits, JointType, SceneGraph, SweepCheck

__all__ = ["SWEEP_STEPS", "run", "sweep_object"]

SWEEP_STEPS = 50


def _subtree(model, root_body_id: int) -> set[int]:
    """Every body at or below `root_body_id`.

    A joint moves the whole subtree hanging off its child part, not just the one
    body, so a nested handle or shelf has to be treated as part of what is moving.
    """
    moving = {root_body_id}
    # body_parentid is monotonic — a child always has a higher index than its
    # parent — so one forward pass suffices.
    for body_id in range(root_body_id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in moving:
            moving.add(body_id)
    return moving


def _geoms_of(model, bodies: set[int]) -> list[int]:
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def _longest_clean_run(clean: list[bool], values: np.ndarray) -> JointLimits | None:
    """Largest contiguous stretch of joint values that swept without violation.

    The gap between this and the fitted limits is range the object genuinely has,
    and reporting it is what stops a repair from quietly trimming away more than
    it needed to.
    """
    best_start = best_len = current_start = current_len = 0
    for index, ok in enumerate(clean):
        if ok:
            if current_len == 0:
                current_start = index
            current_len += 1
            if current_len > best_len:
                best_start, best_len = current_start, current_len
        else:
            current_len = 0

    if best_len < 2:
        # A single clean sample is a numerical coincidence, not a usable range.
        return None
    return JointLimits(
        lower=float(values[best_start]), upper=float(values[best_start + best_len - 1])
    )


def sweep_object(graph: SceneGraph, tolerance_m: float, steps: int = SWEEP_STEPS):
    """Compile the graph once and sweep every non-fixed joint in it."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    data = mujoco.MjData(model)

    checks: list[SweepCheck] = []
    for obj in graph.objects:
        # part_id -> body id, so contacts map back to schema objects by name
        # rather than by taking strings apart.
        part_bodies = {
            part.part_id: model.body(mjcf.body_name(obj.object_id, part.part_id)).id
            for part in obj.parts
        }
        object_bodies = set(part_bodies.values())

        for joint in obj.joints:
            if joint.type is JointType.FIXED:
                continue
            checks.append(
                _sweep_joint(
                    mujoco,
                    model,
                    data,
                    obj,
                    joint,
                    part_bodies,
                    object_bodies,
                    tolerance_m,
                    steps,
                )
            )
    return checks


def _sweep_joint(
    mujoco,
    model,
    data,
    obj,
    joint,
    part_bodies: dict[str, int],
    object_bodies: set[int],
    tolerance_m: float,
    steps: int,
) -> SweepCheck:
    joint_id = model.joint(mjcf.joint_name(obj.object_id, joint.joint_id)).id
    qpos_adr = int(model.jnt_qposadr[joint_id])

    moving_bodies = _subtree(model, part_bodies[joint.child_part_id])
    parent_body = part_bodies.get(joint.parent_part_id)
    body_to_part = {body_id: part_id for part_id, body_id in part_bodies.items()}

    moving_geoms = _geoms_of(model, moving_bodies)

    # Classify everything the moving subtree could hit, once, before sweeping.
    # "Parent" is the joint's own parent part; "sibling" is any other part of the
    # same object; everything else — the floor, other objects — is the world.
    targets: list[tuple[int, str, str]] = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body in moving_bodies:
            continue  # the subtree against itself says nothing about this joint
        if body == parent_body:
            targets.append((geom, "parent", joint.parent_part_id))
        elif body in object_bodies:
            targets.append((geom, "sibling", body_to_part.get(body, str(body))))
        else:
            targets.append((geom, "world", ""))

    values = np.linspace(joint.limits.lower, joint.limits.upper, steps)
    max_parent_penetration = 0.0
    siblings: set[str] = set()
    world_contact = False
    blocked_at: float | None = None
    clean: list[bool] = []

    for q in values:
        # Reset every step so the other joints sit at rest and the previous
        # iteration cannot leak in. qpos0 also restores the free joint, which is
        # what keeps the object where the graph placed it.
        mujoco.mj_resetData(model, data)
        data.qpos[qpos_adr] = q
        # Kinematics only. Nothing is stepped, and the contact array is not read,
        # so the rest of the dynamics pipeline would be wasted work.
        mujoco.mj_kinematics(model, data)

        step_ok = True
        for moving_geom in moving_geoms:
            for geom, kind, part_id in targets:
                distance = mujoco.mj_geomDistance(model, data, moving_geom, geom, tolerance_m, None)
                penetration = -distance
                if penetration <= tolerance_m:
                    continue
                if kind == "parent":
                    max_parent_penetration = max(max_parent_penetration, penetration)
                elif kind == "sibling":
                    siblings.add(part_id)
                else:
                    world_contact = True
                step_ok = False

        clean.append(step_ok)
        if not step_ok and blocked_at is None:
            blocked_at = float(q)

    passed = all(clean)
    return SweepCheck(
        joint_id=joint.joint_id,
        object_id=obj.object_id,
        steps=steps,
        max_parent_penetration_m=max_parent_penetration,
        sibling_contacts=sorted(siblings),
        world_contact=world_contact,
        blocked_at_q=blocked_at,
        # A joint that swept clean already has its full range; reporting it again
        # as "feasible" would invite a repair to re-derive what it already has.
        feasible_limits=None if passed else _longest_clean_run(clean, values),
        passed=passed,
    )


def run(graph: SceneGraph, tolerance_m: float, steps: int = SWEEP_STEPS) -> list[SweepCheck]:
    """Sweep every joint in the scene.

    Objects with zero fitted joints produce no `SweepCheck` at all — they belong
    in `Certificate.jointless_objects`. A cabinet welded shut is trivially
    collision-free, and crediting it with a pass would make the rate a lie.

    This is also the entry point for the PartNet-Mobility audit: point it at a
    graph built from an asset's own ground-truth articulation and it counts how
    often the annotations everyone trains against self-collide.
    """
    return sweep_object(graph, tolerance_m, steps)
