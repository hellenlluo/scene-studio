"""Pairwise penetration depth against the collision decomposition.

The load-bearing test is `test_a_flat_object_under_an_open_one_does_not_collide`.
It is the sofa-and-rug case: a bounding box is a solid brick from an object's feet
to its highest point, so anything lying underneath reads as buried inside it. On
`room.png` that was a phantom 14.5 mm the solver worked to resolve every iteration
while both the decomposition and MuJoCo agreed the pair was not touching.

`test_depth_is_the_separating_distance` guards the other trap: FCL's triangle-mesh
path returns a contact per intersecting triangle pair, and the depth on those is
the overlap of two triangles rather than how far the objects must move apart — two
boxes 0.1 apart reported 1.0. Convex geometry and EPA are what make the number
mean what the residual assumes.
"""

import numpy as np
import pytest
import trimesh

from app.config import get_settings
from app.pipeline import penetration
from app.schemas import (
    AssetFrame,
    ObjectLabel,
    PartGeometry,
    SceneGraph,
    SceneObject,
)


@pytest.fixture
def settings():
    return get_settings()


def _write(tmp_path, mesh, name) -> str:
    path = tmp_path / f"{name}.obj"
    path.write_text(mesh.export(file_type="obj"))
    return str(path)


def _object(
    object_id,
    collision,
    dims,
    position=(0.0, 0.0, 0.0),
    scale=1.0,
    supported_by=None,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(object_id=object_id, category="other"),
        frame=AssetFrame(source="test"),
        parts=[
            PartGeometry(part_id="body", name="body", dims_m=dims, collision_mesh_paths=collision)
        ],
        position_m=position,
        scale=scale,
        supported_by=supported_by,
    )


def _box(tmp_path, extents, translation, name) -> str:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    return _write(tmp_path, mesh, name)


def _table_pieces(tmp_path) -> list[str]:
    """A tabletop on four legs: open underneath, which is the whole point.

    Its bounding box is solid from the floor to the top. Its decomposition is not,
    and everything below turns on that difference.
    """
    top = _box(tmp_path, (1.0, 1.0, 0.05), (0.0, 0.0, 0.475), "top")
    legs = [
        _box(tmp_path, (0.06, 0.06, 0.45), (x, y, 0.225), f"leg{i}")
        for i, (x, y) in enumerate(((0.45, 0.45), (0.45, -0.45), (-0.45, 0.45), (-0.45, -0.45)))
    ]
    return [top, *legs]


# --- the bug ------------------------------------------------------------------


def _rug(tmp_path) -> str:
    """A thin mat small enough to lie *between* the table's legs.

    That is the real geometry of the case: a rug under a sofa passes between its
    feet, so nothing about the two ever touches — while their bounding boxes
    overlap by the rug's whole thickness.
    """
    return _box(tmp_path, (0.5, 0.5, 0.02), (0.0, 0.0, 0.01), "rug")


def test_a_flat_object_under_an_open_one_does_not_collide(tmp_path, settings):
    """The sofa-and-rug case, which the bounding box got wrong by 14.5 mm."""
    table = _object("table", _table_pieces(tmp_path), (1.0, 1.0, 0.5))
    rug = _object("rug", [_rug(tmp_path)], (0.5, 0.5, 0.02))
    graph = SceneGraph(objects=[table, rug])

    depths = penetration.build(graph, settings)
    assert depths.between(table, rug) == pytest.approx(0.0, abs=1e-6)


def test_a_leg_standing_in_it_still_collides(tmp_path, settings):
    """The other direction, so the test above is not just "always returns zero".

    Widen the mat until it reaches the legs and it genuinely intersects them; the
    depth is how far up the leg the mat comes.
    """
    table = _object("table", _table_pieces(tmp_path), (1.0, 1.0, 0.5))
    wide = _object(
        "rug", [_box(tmp_path, (2.0, 2.0, 0.02), (0.0, 0.0, 0.01), "wide")], (2.0, 2.0, 0.02)
    )
    depths = penetration.build(SceneGraph(objects=[table, wide]), settings)

    assert depths.between(table, wide) == pytest.approx(0.02, abs=1e-3)


def test_the_bounding_boxes_of_that_pair_do_overlap(tmp_path, settings):
    """Establishes that the case above is a real disagreement, not a trivial pass.

    Without this, the test above would still pass if the geometry happened not to
    be arranged the way the bug requires.
    """
    from app.geometry import world_aabb

    table = _object("table", _table_pieces(tmp_path), (1.0, 1.0, 0.5))
    rug = _object("rug", [_rug(tmp_path)], (0.5, 0.5, 0.02))
    tl, th = world_aabb(table)
    rl, rh = world_aabb(rug)
    overlap = np.minimum(th, rh) - np.maximum(tl, rl)

    assert overlap.min() > 0.0, "the boxes must overlap for the real test to mean anything"


# --- the measurement itself ---------------------------------------------------


def test_depth_is_the_separating_distance(tmp_path, settings):
    """Not the overlap of some triangle pair — how far they must move to part."""
    piece = _box(tmp_path, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "unit")
    for offset, expected in ((0.9, 0.1), (0.5, 0.5), (1.1, 0.0)):
        a = _object("a", [piece], (1.0, 1.0, 1.0))
        b = _object("b", [piece], (1.0, 1.0, 1.0), position=(offset, 0.0, 0.0))
        depths = penetration.build(SceneGraph(objects=[a, b]), settings)
        assert depths.between(a, b) == pytest.approx(expected, abs=1e-3)


def test_touching_is_not_overlapping(tmp_path, settings):
    """One-sided by design: the residual must not push apart objects in contact,
    which is every object resting on another one."""
    piece = _box(tmp_path, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "unit")
    a = _object("a", [piece], (1.0, 1.0, 1.0))
    b = _object("b", [piece], (1.0, 1.0, 1.0), position=(1.0, 0.0, 0.0))
    depths = penetration.build(SceneGraph(objects=[a, b]), settings)

    assert depths.between(a, b) == pytest.approx(0.0, abs=1e-6)


def test_position_is_applied_per_query(tmp_path, settings):
    """Position is the variable the solver moves most, so it is not baked in.

    The same built objects, queried at a pose they were not built at.
    """
    piece = _box(tmp_path, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "unit")
    a = _object("a", [piece], (1.0, 1.0, 1.0))
    b = _object("b", [piece], (1.0, 1.0, 1.0), position=(5.0, 0.0, 0.0))
    depths = penetration.build(SceneGraph(objects=[a, b]), settings)

    assert depths.between(a, b) == pytest.approx(0.0, abs=1e-6)
    moved = b.model_copy(update={"position_m": (0.4, 0.0, 0.0)})
    assert depths.between(a, moved) == pytest.approx(0.6, abs=1e-3)


def test_scale_is_baked_in_at_build_time(tmp_path, settings):
    """Documented approximation, asserted so it cannot change silently.

    FCL cannot rescale a collision object, so scale is baked and the round loop
    rebuilds. A scale changed after the build is therefore *not* reflected — which
    is fine at the few percent a Gauss-Newton pass moves one, and would be a
    surprise to anyone assuming otherwise.
    """
    piece = _box(tmp_path, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "unit")
    a = _object("a", [piece], (1.0, 1.0, 1.0))
    b = _object("b", [piece], (1.0, 1.0, 1.0), position=(1.2, 0.0, 0.0))
    depths = penetration.build(SceneGraph(objects=[a, b]), settings)

    assert depths.between(a, b) == pytest.approx(0.0, abs=1e-6)
    grown = b.model_copy(update={"scale": 2.0})
    assert depths.between(a, grown) == pytest.approx(0.0, abs=1e-6), "stale until rebuilt"

    rebuilt = penetration.build(SceneGraph(objects=[a, grown]), settings)
    assert rebuilt.between(a, grown) > 0.0, "and correct once it is"


# --- which pairs get tested ---------------------------------------------------


def test_a_declared_support_contact_is_not_a_pair(tmp_path, settings):
    """The support term is already driving those surfaces together; penalising the
    same contact here would have the solver fighting itself."""
    piece = _box(tmp_path, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), "unit")
    graph = SceneGraph(
        objects=[
            _object("table", [piece], (1.0, 1.0, 1.0)),
            _object("mug", [piece], (1.0, 1.0, 1.0), supported_by="table"),
            _object("lamp", [piece], (1.0, 1.0, 1.0)),
        ]
    )
    pairs = penetration.pairs(graph)

    assert ("table", "mug") not in pairs and ("mug", "table") not in pairs
    assert ("table", "lamp") in pairs
    assert ("mug", "lamp") in pairs


def test_falls_back_to_the_box_without_a_decomposition(settings):
    """The fixtures carry no collision paths, and a graph that has not reached the
    inertia stage carries none either. Degrades to the old behaviour, not to zero."""
    a = _object("a", [], (1.0, 1.0, 1.0))
    b = _object("b", [], (1.0, 1.0, 1.0), position=(0.5, 0.0, 0.0))
    depths = penetration.build(SceneGraph(objects=[a, b]), settings)

    assert depths.between(a, b) == pytest.approx(0.5, abs=1e-3)
