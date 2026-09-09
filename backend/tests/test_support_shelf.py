"""Picking a surface inside a parent that has geometry above the contact.

The height grid used to keep only the topmost surface per column, which answers
"how high is this object" when the question is "what is this child standing on".
For anything with a lid, a shelf or a rim above the contact those differ by the
whole height of the parent, and both cases are in `room2.png`: books on a middle
shelf of a bookcase, and magazines standing in a wicker basket.

Neither is a containment *exemption*. A magazine in a basket rests on the basket's
inside floor and its mesh should touch that floor, not pass through it — the
overlap the certifier used to report was the consequence of aiming the contact at
the rim, not evidence that overlap is acceptable inside a container.
"""

import numpy as np
import pytest
import trimesh

from app.config import get_settings
from app.geometry import world_aabb
from app.pipeline import support
from app.schemas import AssetFrame, ObjectLabel, PartGeometry, SceneGraph, SceneObject


@pytest.fixture
def settings():
    return get_settings()


def _mesh(tmp_path, mesh, name) -> str:
    path = tmp_path / f"{name}.obj"
    path.write_text(mesh.export(file_type="obj"))
    return str(path)


def _object(object_id, collision, dims, position, supported_by=None) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(object_id=object_id, category=object_id, support_parent=supported_by),
        frame=AssetFrame(source="test"),
        parts=[
            PartGeometry(part_id="body", name="body", dims_m=dims, collision_mesh_paths=collision)
        ],
        position_m=position,
        supported_by=supported_by,
    )


def _shelf_unit(tmp_path):
    """Two horizontal boards: one at z=0, one at z=0.4, centred on the origin."""
    lower = trimesh.creation.box(extents=(0.8, 0.3, 0.02))
    upper = trimesh.creation.box(extents=(0.8, 0.3, 0.02))
    upper.apply_translation((0.0, 0.0, 0.4))
    return _mesh(tmp_path, trimesh.util.concatenate([lower, upper]), "unit")


def test_a_book_rests_on_its_own_shelf_not_the_top_of_the_case(tmp_path, settings):
    """The failure this exists for. Every surface above the book is irrelevant to it,
    and the topmost one is 400 mm out."""
    unit = _shelf_unit(tmp_path)
    book = _mesh(tmp_path, trimesh.creation.box(extents=(0.2, 0.15, 0.06)), "book")
    graph = SceneGraph(
        objects=[
            _object("case", [unit], (0.8, 0.3, 0.42), (0.0, 0.0, 0.0)),
            # Sitting on the lower board: its top is z=0.01, so the book centres at 0.04.
            _object("book", [book], (0.2, 0.15, 0.06), (0.0, 0.0, 0.04), supported_by="case"),
        ]
    )
    heights = support.build(graph, settings)
    low, high = world_aabb(graph.get("book"))
    surface = heights.under(
        graph.get("case"), low, high,
        base=heights.base_of(graph.get("book")),
        slack=heights.burial_slack_m,
    )
    assert surface == pytest.approx(0.01, abs=2e-3), "the lower board, not the upper"
    assert abs(heights.base_of(graph.get("book")) - surface) < 0.005, "and it is resting on it"


def test_the_top_surface_is_still_the_answer_without_a_child(tmp_path, settings):
    """`base=None` keeps the old query, which is what a caller with nothing resting
    on the parent still wants."""
    unit = _shelf_unit(tmp_path)
    graph = SceneGraph(objects=[_object("case", [unit], (0.8, 0.3, 0.42), (0.0, 0.0, 0.0))])
    heights = support.build(graph, settings, candidates={"case"})
    low = np.array([-0.1, -0.1, 0.0])
    high = np.array([0.1, 0.1, 0.1])
    assert heights.under(graph.get("case"), low, high) == pytest.approx(0.41, abs=2e-3)


def test_an_object_buried_in_a_shelf_stays_on_that_shelf(tmp_path, settings):
    """Reconstruction buries things by centimetres. A strict cutoff would skip the
    shelf the book is buried in and answer with the one below, turning a 20 mm error
    into a whole shelf of error."""
    unit = _shelf_unit(tmp_path)
    book = _mesh(tmp_path, trimesh.creation.box(extents=(0.2, 0.15, 0.06)), "book")
    graph = SceneGraph(
        objects=[
            _object("case", [unit], (0.8, 0.3, 0.42), (0.0, 0.0, 0.0)),
            # 20 mm lower than resting: its base is below the upper board's top.
            _object("book", [book], (0.2, 0.15, 0.06), (0.0, 0.0, 0.39), supported_by="case"),
        ]
    )
    heights = support.build(graph, settings)
    low, high = world_aabb(graph.get("book"))
    surface = heights.under(
        graph.get("case"), low, high,
        base=heights.base_of(graph.get("book")),
        slack=heights.burial_slack_m,
    )
    assert surface == pytest.approx(0.41, abs=2e-3), "the upper board it is buried in"


def test_a_magazine_stands_on_a_basket_floor_not_its_rim(tmp_path, settings):
    """A container is a shelf turned inside out. The rim is the topmost surface in
    the column and the contact is the inside floor, 240 mm below it."""
    floor = trimesh.creation.box(extents=(0.3, 0.3, 0.02))
    wall = trimesh.creation.box(extents=(0.3, 0.02, 0.25))
    wall.apply_translation((0.0, 0.14, 0.125))
    basket = _mesh(tmp_path, trimesh.util.concatenate([floor, wall]), "basket")
    mag = _mesh(tmp_path, trimesh.creation.box(extents=(0.2, 0.02, 0.3)), "mag")
    graph = SceneGraph(
        objects=[
            _object("basket", [basket], (0.3, 0.3, 0.26), (0.0, 0.0, 0.0)),
            # Standing on the inside floor (top z=0.01), sticking out above the rim.
            _object("mag", [mag], (0.2, 0.02, 0.3), (0.0, 0.0, 0.16), supported_by="basket"),
        ]
    )
    heights = support.build(graph, settings)
    low, high = world_aabb(graph.get("mag"))
    surface = heights.under(
        graph.get("basket"), low, high,
        base=heights.base_of(graph.get("mag")),
        slack=heights.burial_slack_m,
    )
    assert surface == pytest.approx(0.01, abs=2e-3), "the inside floor, not the rim"


def _pedestal(tmp_path):
    """A wide round top on a narrow stem — a table whose footprint is almost all air."""
    top = trimesh.creation.cylinder(radius=0.28, height=0.02)
    top.apply_translation((0.0, 0.0, 0.59))
    stem = trimesh.creation.cylinder(radius=0.03, height=0.58)
    stem.apply_translation((0.0, 0.0, 0.29))
    foot = trimesh.creation.cylinder(radius=0.09, height=0.02)
    foot.apply_translation((0.0, 0.0, 0.01))
    return _mesh(tmp_path, trimesh.util.concatenate([top, stem, foot]), "pedestal")


def test_the_gap_is_measured_where_the_object_actually_touches(tmp_path, settings):
    """The two halves used to be read in different places.

    `base_of` is the lowest point anywhere on the child; the old surface query was
    the highest surface anywhere under the child's *bounding box*. For a solid box
    those coincide. For a pedestal they need not — measured on `room2.png`, the side
    table's contact patch is 0.107 x 0.037 m inside a 0.56 x 0.58 m footprint, so
    99% of what was probed is the empty air under a round top.

    Here the rug reaches only the near half of the table's footprint and stops well
    short of its foot. Probing the bounding box finds the rug and calls it contact;
    probing the foot correctly finds nothing under it.
    """
    # Clear of the whole table, top included: the round top has radius 0.28, so a rug
    # spanning y[-1.0, -0.6] is under no part of it. A rug that merely missed the foot
    # while still reaching under the overhanging top would correctly report the
    # top-to-rug clearance, which is a real nearest approach and not a contact.
    rug = trimesh.creation.box(extents=(1.0, 0.4, 0.02))
    rug.apply_translation((0.0, -0.8, 0.01))
    graph = SceneGraph(
        objects=[
            _object("rug", [_mesh(tmp_path, rug, "rug")], (1.0, 0.4, 0.02), (0.0, 0.0, 0.0)),
            _object("table", [_pedestal(tmp_path)], (0.56, 0.56, 0.6), (0.0, 0.0, 0.0), "rug"),
        ]
    )
    heights = support.build(graph, settings)
    assert heights.gap_to(graph.get("rug"), graph.get("table"), 0.0) is None, (
        "nothing is under the foot, so there is no contact to report"
    )


def test_a_pedestal_resting_on_a_wide_rug_reads_zero(tmp_path, settings):
    """The other half: when the rug *is* under the foot, the gap is the foot's."""
    rug = trimesh.creation.box(extents=(2.0, 2.0, 0.02))
    rug.apply_translation((0.0, 0.0, 0.01))
    graph = SceneGraph(
        objects=[
            _object("rug", [_mesh(tmp_path, rug, "rug")], (2.0, 2.0, 0.02), (0.0, 0.0, 0.0)),
            _object("table", [_pedestal(tmp_path)], (0.56, 0.56, 0.6), (0.0, 0.0, 0.02), "rug"),
        ]
    )
    heights = support.build(graph, settings)
    gap = heights.gap_to(graph.get("rug"), graph.get("table"), 0.0)
    assert gap == pytest.approx(0.0, abs=3e-3), f"resting on the rug, got {gap}"
