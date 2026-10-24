import mujoco
import pytest

from app.export import mjcf
from app.simulation import SimulationDiverged, SimulationGuard, checked_steps
from tests.fixtures import scenes


class _Warning:
    number = 0
    lastinfo = 0


class _Data:
    def __init__(self):
        self.warning = [_Warning() for _ in range(7)]
        self.time = 0.0


class _Model:
    nv = 6


class _DivergingMuJoCo:
    mjtWarning = mujoco.mjtWarning

    @staticmethod
    def mj_step(_model, data):
        data.time += 0.002
        warning = data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)]
        warning.number += 1
        warning.lastinfo = 4


def test_checked_step_names_the_scene_stage_dof_and_object():
    guard = SimulationGuard(
        scene_id="room2",
        stage="certify.stability",
        dof_objects=("table", "table", "table", "side_table", "side_table", "side_table"),
    )

    with pytest.raises(SimulationDiverged) as raised:
        checked_steps(_DivergingMuJoCo, _Model(), _Data(), guard)

    error = raised.value
    assert error.scene_id == "room2"
    assert error.stage == "certify.stability"
    assert error.warning == "BADQACC"
    assert error.dof == 4
    assert error.object_id == "side_table"
    assert str(error) == (
        "scene room2, stage certify.stability: MuJoCo BADQACC at DOF 4 (side_table) after 0.0020 s"
    )


def test_guard_maps_compiled_dofs_back_to_graph_objects():
    graph = scenes.kitchen()
    model = mjcf.load_model(graph)

    guard = SimulationGuard.for_model(model, graph, scene_id="kitchen", stage="certify.stability")

    assert len(guard.dof_objects) == model.nv
    assert guard.dof_objects[:6] == ("table",) * 6
    assert guard.dof_objects[6:12] == ("mug",) * 6
    assert guard.dof_objects[12:18] == ("cabinet",) * 6
