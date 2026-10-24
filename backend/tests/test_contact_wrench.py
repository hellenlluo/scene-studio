"""General 3-D equilibrium cases the horizontal footprint model cannot express."""

import numpy as np

from app.certify import balance, contact_wrench
from app.pipeline.support import Contact
from app.schemas import (
    AssetFrame,
    InertialProperties,
    ObjectLabel,
    PartGeometry,
    SceneGraph,
    SceneObject,
)


def _body(com_x: float = 0.5) -> SceneObject:
    return SceneObject(
        object_id="book",
        label=ObjectLabel(object_id="book", category="book"),
        frame=AssetFrame(source="3d-wrench-test"),
        parts=[
            PartGeometry(
                part_id="body",
                name="body",
                dims_m=(1.0, 0.1, 1.0),
                inertial=InertialProperties(
                    density_kg_m3=1.0,
                    volume_m3=1.0,
                    mass_kg=1.0,
                    com_m=(com_x, 0.0, 0.0),
                    inertia_diag=(1.0, 1.0, 1.0),
                    watertight=True,
                ),
            )
        ],
        position_m=(0.0, 0.0, 0.5),
    )


def _contact(
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    tangent_1: tuple[float, float, float],
    tangent_2: tuple[float, float, float],
    other: str | None,
) -> contact_wrench.ContactWrench:
    return contact_wrench.ContactWrench(
        position_m=position,
        normal_m=normal,
        tangent_1_m=tangent_1,
        tangent_2_m=tangent_2,
        friction=(1.0, 1.0, 0.005, 0.0001, 0.0001),
        dimensions=6,
        other_object_id=other,
        distance_m=0.0,
    )


FLOOR = _contact(
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    None,
)
WALL = _contact(
    (1.0, 0.0, 1.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    "basket",
)


class _NarrowBottom:
    """A bottom patch whose XY hull does not contain the authored COM."""

    points = np.array(
        [(-0.1, -0.05, 0.0), (-0.1, 0.05, 0.0), (0.1, 0.05, 0.0), (0.1, -0.05, 0.0)]
    )

    def contact(self, *_args) -> Contact:
        return Contact(
            gap=0.0,
            resting_on="basket",
            contacts_xy=self.points[:, :2],
            contacts_m=self.points,
        )

    def base_of(self, _obj: SceneObject) -> float:
        return 0.0


def test_wall_contact_supplies_wrench_that_bottom_contact_cannot():
    obj = _body()
    centre = np.array([0.5, 0.0, 0.5])

    assert not contact_wrench.is_feasible(obj, [FLOOR], centre)
    assert contact_wrench.is_feasible(obj, [FLOOR, WALL], centre)
    assert contact_wrench.is_non_horizontal(WALL)
    assert not contact_wrench.is_non_horizontal(FLOOR)


def test_contact_evidence_uses_the_same_small_gap_tolerance_as_support_checks():
    obj = _body(com_x=0.0)
    obj.position_m = (0.0, 0.0, 0.5005)  # box bottom is 0.5 mm above the floor
    graph = SceneGraph(objects=[obj])

    assert contact_wrench.contacts_by_object(graph)["book"] == []

    nearby = contact_wrench.contacts_by_object(graph, contact_tolerance_m=0.001)["book"]

    assert any(
        contact.other_object_id is None and 0.0 < contact.distance_m <= 0.001
        for contact in nearby
    )


def test_balance_uses_3d_wrench_instead_of_failing_the_xy_hull(monkeypatch):
    obj = _body()
    graph = SceneGraph(objects=[obj])
    monkeypatch.setattr(
        contact_wrench,
        "contacts_by_object",
        lambda _graph, _tolerance: {"book": [FLOOR, WALL]},
    )

    check = balance.run(graph, _NarrowBottom(), 0.005)[0]

    assert check.support_margin_m < 0.0  # the old horizontal verdict would fail
    assert check.passed
    assert check.contact_force_feasible
    assert check.failure_kind == "none"
    assert check.equilibrium_model == "contact_wrench_3d"
    assert check.non_horizontal_contact_count == 1
    assert check.wrench_contacts[1].other_object_id == "basket"


def test_child_contact_does_not_claim_to_hold_its_parent_up(monkeypatch):
    basket = _body(com_x=0.0)
    basket.object_id = "basket"
    basket.label = basket.label.model_copy(update={"object_id": "basket", "category": "basket"})
    book = _body()
    book.supported_by = "basket"
    graph = SceneGraph(objects=[basket, book])
    reaction_on_basket = _contact(
        WALL.position_m,
        tuple(-value for value in WALL.normal_m),
        WALL.tangent_1_m,
        WALL.tangent_2_m,
        "book",
    )
    monkeypatch.setattr(
        contact_wrench,
        "contacts_by_object",
        lambda _graph, _tolerance: {
            "basket": [reaction_on_basket],
            "book": [FLOOR, WALL],
        },
    )

    checks = {check.object_id: check for check in balance.run(graph, _NarrowBottom(), 0.005)}

    assert checks["book"].equilibrium_model == "contact_wrench_3d"
    assert checks["book"].passed
    assert checks["basket"].equilibrium_model == "horizontal_support_2d"
    assert checks["basket"].wrench_contacts == []
    assert checks["basket"].passed


def test_descendant_contact_does_not_claim_to_hold_container_up(monkeypatch):
    basket = _body(com_x=0.0)
    basket.object_id = "basket"
    basket.label = basket.label.model_copy(update={"object_id": "basket", "category": "basket"})
    lower = _body(com_x=0.0)
    lower.object_id = "lower"
    lower.label = lower.label.model_copy(update={"object_id": "lower"})
    lower.supported_by = "basket"
    upper = _body()
    upper.object_id = "upper"
    upper.label = upper.label.model_copy(update={"object_id": "upper"})
    upper.supported_by = "lower"
    graph = SceneGraph(objects=[basket, lower, upper])
    descendant_reaction = _contact(
        WALL.position_m,
        tuple(-value for value in WALL.normal_m),
        WALL.tangent_1_m,
        WALL.tangent_2_m,
        "upper",
    )
    monkeypatch.setattr(
        contact_wrench,
        "contacts_by_object",
        lambda _graph, _tolerance: {
            "basket": [descendant_reaction],
            "lower": [FLOOR],
            "upper": [FLOOR, WALL],
        },
    )

    checks = {check.object_id: check for check in balance.run(graph, _NarrowBottom(), 0.005)}

    assert checks["basket"].equilibrium_model == "horizontal_support_2d"
    assert checks["basket"].wrench_contacts == []
    assert checks["basket"].passed
