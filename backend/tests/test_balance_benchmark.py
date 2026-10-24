"""Controlled benchmark for the static-balance certificate.

Unlike the reconstructed-scene tests, every contact and centre of mass here is
authored, so the expected margin and verdict are known before the implementation
runs.  This isolates the equilibrium calculation from perception and contact
sampling; ``test_balance.py`` covers their integration.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from app.certify import balance
from app.pipeline.support import Contact
from app.schemas import (
    AssetFrame,
    InertialProperties,
    ObjectLabel,
    PartGeometry,
    SceneGraph,
    SceneObject,
)


@dataclass(frozen=True)
class BalanceCase:
    name: str
    contacts_xy: tuple[tuple[float, float], ...]
    com_xy: tuple[float, float]
    expected_pass: bool
    expected_margin_m: float
    expected_hull_vertices: int


CASES = (
    BalanceCase(
        "centred_square",
        ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)),
        (0.0, 0.0),
        True,
        0.5,
        4,
    ),
    BalanceCase(
        "near_edge_inside",
        ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)),
        (0.49, 0.0),
        True,
        0.01,
        4,
    ),
    BalanceCase(
        "exactly_on_edge",
        ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)),
        (0.5, 0.0),
        True,
        0.0,
        4,
    ),
    BalanceCase(
        "outside_edge",
        ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5)),
        (0.6, 0.0),
        False,
        -0.1,
        4,
    ),
    BalanceCase(
        "inside_triangle",
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (0.25, 0.25),
        True,
        0.25,
        3,
    ),
    BalanceCase("one_point", ((0.0, 0.0),), (0.0, 0.0), False, 0.0, 1),
    BalanceCase(
        "two_points",
        ((-0.5, 0.0), (0.5, 0.0)),
        (0.0, 0.0),
        False,
        0.0,
        2,
    ),
    BalanceCase(
        "three_collinear_points",
        ((-0.5, 0.0), (0.0, 0.0), (0.5, 0.0)),
        (0.0, 0.0),
        False,
        0.0,
        3,
    ),
    BalanceCase(
        "four_spread_feet",
        ((-0.8, -0.4), (-0.8, 0.4), (0.8, 0.4), (0.8, -0.4)),
        (0.0, 0.0),
        True,
        0.4,
        4,
    ),
    BalanceCase(
        "one_of_four_feet_missing",
        ((-1.0, -0.5), (-1.0, 0.5), (1.0, -0.5)),
        (0.2, 0.1),
        False,
        -0.4 / np.sqrt(5.0),
        3,
    ),
)


class ControlledSupports:
    """Supply exact contacts while preserving ``balance.run``'s public boundary."""

    def __init__(self, contacts_xy: tuple[tuple[float, float], ...]):
        xy = np.asarray(contacts_xy, dtype=float).reshape(-1, 2)
        self.points = np.column_stack((xy, np.zeros(len(xy))))

    def contact(self, *_args) -> Contact:
        return Contact(
            gap=0.0,
            resting_on=None,
            contacts_xy=self.points[:, :2],
            contacts_m=self.points,
        )

    def base_of(self, _obj: SceneObject) -> float:
        # A known base suppresses balance's OBB fallback for deliberately
        # degenerate one- and two-point cases.
        return 0.0


def controlled_graph(com_xy: tuple[float, float]) -> SceneGraph:
    inertial = InertialProperties(
        density_kg_m3=1.0,
        volume_m3=1.0,
        mass_kg=1.0,
        com_m=(com_xy[0], com_xy[1], 0.0),
        inertia_diag=(1.0, 1.0, 1.0),
        watertight=True,
    )
    obj = SceneObject(
        object_id="body",
        label=ObjectLabel(object_id="body", category="other"),
        frame=AssetFrame(source="controlled-balance-benchmark"),
        parts=[
            PartGeometry(
                part_id="body",
                name="body",
                dims_m=(1.0, 1.0, 1.0),
                inertial=inertial,
            )
        ],
        position_m=(0.0, 0.0, 0.5),
    )
    return SceneGraph(objects=[obj])


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_controlled_balance_benchmark(case: BalanceCase):
    check = balance.run(
        controlled_graph(case.com_xy),
        ControlledSupports(case.contacts_xy),
        contact_tolerance_m=0.005,
        use_3d_contacts=False,
    )[0]

    assert check.passed is case.expected_pass
    assert check.contact_force_feasible is case.expected_pass
    assert check.support_margin_m == pytest.approx(case.expected_margin_m, abs=1e-9)
    assert check.contact_count == len(case.contacts_xy)
    assert len(check.support_polygon_xy_m) == case.expected_hull_vertices
    assert check.evidence_source == "collision_proxy"


@pytest.mark.parametrize("com_x", np.linspace(-0.6, 0.6, 13))
def test_square_margin_sweep_matches_analytic_distance(com_x: float):
    """For a one-metre square, the exact X margin is 0.5 - abs(COM.x)."""
    contacts = ((-0.5, -0.5), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5))
    check = balance.run(
        controlled_graph((float(com_x), 0.0)),
        ControlledSupports(contacts),
        contact_tolerance_m=0.005,
        use_3d_contacts=False,
    )[0]

    expected_margin = 0.5 - abs(float(com_x))
    assert check.support_margin_m == pytest.approx(expected_margin, abs=1e-9)
    assert check.passed is (expected_margin >= -1e-9)


def test_balance_is_invariant_to_contact_order():
    contacts = ((-0.8, -0.4), (-0.8, 0.4), (0.8, 0.4), (0.8, -0.4))
    graph = controlled_graph((0.2, -0.1))
    forward = balance.run(
        graph, ControlledSupports(contacts), 0.005, use_3d_contacts=False
    )[0]
    reverse = balance.run(
        graph,
        ControlledSupports(tuple(reversed(contacts))),
        0.005,
        use_3d_contacts=False,
    )[0]

    assert reverse.passed is forward.passed
    assert reverse.support_margin_m == pytest.approx(forward.support_margin_m, abs=1e-9)
    assert set(reverse.support_polygon_xy_m) == set(forward.support_polygon_xy_m)
