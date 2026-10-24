"""Checked MuJoCo stepping with scene-level diagnostics.

MuJoCo reports numerical divergence through ``MjData.warning`` and also writes a
process-wide ``MUJOCO_LOG.TXT``.  The latter has no scene or object identity and is
append-only, so it is useful evidence but a poor application contract.  Every
backend simulation goes through this module instead: a new bad-position,
bad-velocity or bad-acceleration warning becomes a typed exception that names the
scene, pipeline stage, degree of freedom and owning object.
"""

import logging
from dataclasses import dataclass

from app.export import mjcf
from app.schemas import SceneGraph

__all__ = ["SimulationDiverged", "SimulationGuard", "checked_steps"]

log = logging.getLogger(__name__)


class SimulationDiverged(RuntimeError):
    """MuJoCo rejected a state as numerically unstable."""

    def __init__(
        self,
        *,
        scene_id: str,
        stage: str,
        warning: str,
        dof: int,
        object_id: str | None,
        time_s: float,
    ) -> None:
        self.scene_id = scene_id
        self.stage = stage
        self.warning = warning
        self.dof = dof
        self.object_id = object_id
        self.time_s = time_s
        owner = object_id or "unknown object"
        super().__init__(
            f"scene {scene_id}, stage {stage}: MuJoCo {warning} at DOF {dof} "
            f"({owner}) after {time_s:.4f} s"
        )


@dataclass(frozen=True)
class SimulationGuard:
    """Stable context and DOF ownership for one compiled model."""

    scene_id: str
    stage: str
    dof_objects: tuple[str | None, ...]

    @classmethod
    def for_model(
        cls,
        model,
        graph: SceneGraph,
        *,
        scene_id: str,
        stage: str,
    ) -> "SimulationGuard":
        # Map bodies through the same naming function that emitted the MJCF.  A
        # free joint belongs to the root body today, but mapping every part keeps
        # the diagnostic correct if articulated objects gain child joints later.
        body_objects: dict[int, str] = {}
        for obj in graph.objects:
            for part in obj.parts:
                try:
                    body_id = int(model.body(mjcf.body_name(obj.object_id, part.part_id)).id)
                except KeyError:
                    continue
                body_objects[body_id] = obj.object_id

        return cls(
            scene_id=scene_id,
            stage=stage,
            dof_objects=tuple(
                body_objects.get(int(model.dof_bodyid[dof])) for dof in range(model.nv)
            ),
        )


def checked_steps(mujoco, model, data, guard: SimulationGuard, steps: int = 1) -> None:
    """Advance ``steps`` times and raise on a newly recorded fatal warning.

    Warning counters are sampled around the batch rather than inferred from the
    state after MuJoCo has reset it.  ``lastinfo`` is the offending DOF for the
    three numerical warnings watched here.
    """

    watched = (
        (mujoco.mjtWarning.mjWARN_BADQPOS, "BADQPOS"),
        (mujoco.mjtWarning.mjWARN_BADQVEL, "BADQVEL"),
        (mujoco.mjtWarning.mjWARN_BADQACC, "BADQACC"),
    )
    before = {int(kind): int(data.warning[int(kind)].number) for kind, _ in watched}
    time_before = float(data.time)

    for _ in range(steps):
        mujoco.mj_step(model, data)

    for kind, name in watched:
        stat = data.warning[int(kind)]
        if int(stat.number) <= before[int(kind)]:
            continue
        dof = int(stat.lastinfo)
        object_id = guard.dof_objects[dof] if 0 <= dof < len(guard.dof_objects) else None
        error = SimulationDiverged(
            scene_id=guard.scene_id,
            stage=guard.stage,
            warning=name,
            dof=dof,
            object_id=object_id,
            # MuJoCo may reset time when it rejects the state.  Retain the last
            # credible time instead of reporting the reset as a failure at t=0.
            time_s=max(time_before, float(data.time)),
        )
        log.error("%s", error)
        raise error
