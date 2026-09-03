"""Support-surface heights from real geometry.

The load-bearing test is `test_finds_the_seat_not_the_backrest`. Everything else
checks a mechanical property; that one is the bug this module exists for — a
bounding box is topped by a sofa's backrest, every cushion placed on that sofa was
seeded 23 cm too high, and the only visible symptom was the stability axis
reporting a 165 degree tumble two stages later.

`test_is_exact_under_scale` and `test_is_exact_under_translation` are the reason
the grid is stored in the parent's oriented unit-scale frame rather than in world
space. Both are variables the solver moves, and it moves them *inside* a
Gauss-Newton pass, so a grid that went stale under either would be wrong for most
of the evaluations that read it.
"""

import numpy as np
import pytest
import trimesh

from app.config import get_settings
from app.pipeline import support
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


def _write(tmp_path, mesh: trimesh.Trimesh, name: str) -> str:
    path = tmp_path / f"{name}.obj"
    path.write_text(mesh.export(file_type="obj"))
    return str(path)


def _object(
    object_id: str,
    collision: list[str],
    dims: tuple[float, float, float],
    position=(0.0, 0.0, 0.0),
    scale: float = 1.0,
    supported_by: str | None = None,
    orientation=(1.0, 0.0, 0.0, 0.0),
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(object_id=object_id, category="other"),
        frame=AssetFrame(source="test"),
        parts=[
            PartGeometry(
                part_id="body",
                name="body",
                dims_m=dims,
                collision_mesh_paths=collision,
            )
        ],
        position_m=position,
        orientation=orientation,
        scale=scale,
        supported_by=supported_by,
    )


def _slab(tmp_path, extents, translation, name) -> str:
    """A box written as a collision piece, positioned in the part's own frame."""
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(translation)
    return _write(tmp_path, mesh, name)


def _sofa_pieces(tmp_path) -> list[str]:
    """A seat at z in [0, 0.4] and a backrest behind it reaching 0.7.

    Two convex pieces, which is what CoACD produces and what MuJoCo collides
    against. The point of the shape is that its highest point is nowhere near the
    surface anything rests on.
    """
    seat = _slab(tmp_path, (1.0, 0.5, 0.4), (0.0, 0.0, 0.2), "seat")
    back = _slab(tmp_path, (1.0, 0.1, 0.3), (0.0, 0.2, 0.55), "back")
    return [seat, back]


# --- the bug ------------------------------------------------------------------


def test_finds_the_seat_not_the_backrest(tmp_path, settings):
    """A cushion on the seat rests at 0.4, not at the backrest's 0.7."""
    sofa = _object("sofa", _sofa_pieces(tmp_path), dims=(1.0, 0.5, 0.7))
    cushion = _object(
        "cushion", [], dims=(0.3, 0.3, 0.1), position=(0.0, -0.1, 0.45), supported_by="sofa"
    )
    graph = SceneGraph(objects=[sofa, cushion])

    heights = support.build(graph, settings)
    low = np.array([-0.15, -0.25, 0.0])
    high = np.array([0.15, 0.05, 0.0])

    assert heights.under(sofa, low, high) == pytest.approx(0.4, abs=1e-3)


def test_the_backrest_is_still_found_when_that_is_what_is_underneath(tmp_path, settings):
    """Not a blanket preference for lower surfaces — it reads what is actually there."""
    sofa = _object("sofa", _sofa_pieces(tmp_path), dims=(1.0, 0.5, 0.7))
    graph = SceneGraph(objects=[sofa, _object("x", [], (0.1, 0.1, 0.1), supported_by="sofa")])
    heights = support.build(graph, settings)

    over_back = heights.under(sofa, np.array([-0.05, 0.18, 0.0]), np.array([0.05, 0.22, 0.0]))
    assert over_back == pytest.approx(0.7, abs=1e-3)


def test_measures_the_mesh_rather_than_dims_m(tmp_path, settings):
    """`world_aabb` trusts the stored box; this does not.

    The second source of truth that caused the other half of the real failure: a
    reconstructed rug's `dims_m` box disagreed with its collision mesh by 8.5 mm at
    the top, so the table driven onto it fell that far. Here `dims_m` is a
    deliberate lie and the measurement has to ignore it.
    """
    plate = _slab(tmp_path, (1.0, 1.0, 0.2), (0.0, 0.0, 0.1), "plate")
    lying = _object("plate", [plate], dims=(1.0, 1.0, 5.0))  # dims_m says 5 m tall
    graph = SceneGraph(objects=[lying, _object("x", [], (0.1,) * 3, supported_by="plate")])

    heights = support.build(graph, settings)
    got = heights.under(lying, np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.0]))
    assert got == pytest.approx(0.2, abs=1e-3), "read the mesh, not the 5 m dims_m"


# --- exactness under what the solver moves ------------------------------------


def test_is_exact_under_translation(tmp_path, settings):
    plate = _slab(tmp_path, (1.0, 1.0, 0.2), (0.0, 0.0, 0.1), "plate")
    graph_at = lambda pos: SceneGraph(  # noqa: E731 - one expression, read once
        objects=[
            _object("plate", [plate], (1.0, 1.0, 0.2), position=pos),
            _object("x", [], (0.1,) * 3, supported_by="plate"),
        ]
    )

    base = graph_at((0.0, 0.0, 0.0))
    heights = support.build(base, settings)
    at_origin = heights.under(
        base.get("plate"), np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.0])
    )

    # The same grid, queried against a parent that has since moved. Exact, because
    # the grid holds oriented unit-scale heights and the lookup divides the pose out.
    moved = graph_at((3.0, -2.0, 0.5)).get("plate")
    at_moved = heights.under(moved, np.array([2.9, -2.1, 0.0]), np.array([3.1, -1.9, 0.0]))

    assert at_origin == pytest.approx(0.2, abs=1e-3)
    assert at_moved == pytest.approx(0.7, abs=1e-3), "0.5 m of parent z on top of 0.2"


def test_is_exact_under_scale(tmp_path, settings):
    """Height goes as scale, and the footprint lookup has to scale with it.

    Scale is a stage-6 variable, so this is read at a scale the grid was not built
    at on almost every residual evaluation.
    """
    plate = _slab(tmp_path, (1.0, 1.0, 0.2), (0.0, 0.0, 0.1), "plate")
    graph = SceneGraph(
        objects=[
            _object("plate", [plate], (1.0, 1.0, 0.2)),
            _object("x", [], (0.1,) * 3, supported_by="plate"),
        ]
    )
    heights = support.build(graph, settings)

    doubled = graph.get("plate").model_copy(update={"scale": 2.0})
    got = heights.under(doubled, np.array([-0.2, -0.2, 0.0]), np.array([0.2, 0.2, 0.0]))
    assert got == pytest.approx(0.4, abs=2e-3)


# --- edges --------------------------------------------------------------------


def test_a_child_beyond_the_edge_reports_nothing(tmp_path, settings):
    """None, not a number, so the caller can fall back rather than be handed a lie.

    An object placed off the side of its support has no surface under it, and
    inventing one — the nearest edge, the box top — would be the solver optimising
    toward a contact that cannot exist.
    """
    plate = _slab(tmp_path, (0.4, 0.4, 0.2), (0.0, 0.0, 0.1), "plate")
    graph = SceneGraph(
        objects=[
            _object("plate", [plate], (0.4, 0.4, 0.2)),
            _object("x", [], (0.1,) * 3, supported_by="plate"),
        ]
    )
    heights = support.build(graph, settings)

    assert (
        heights.under(graph.get("plate"), np.array([5.0, 5.0, 0.0]), np.array([5.1, 5.1, 0.0]))
        is None
    )


def test_a_gap_under_the_centre_still_finds_the_surface(tmp_path, settings):
    """Why the probes spread over the footprint instead of sampling the centre.

    A convex decomposition is a pile of hulls with gaps between them, so the point
    directly under an object's middle is quite often empty air. A centre-only probe
    reads that as "no surface" and falls back to the bounding box — the very number
    this module exists to stop using. The corner probes find the hulls either side.
    """
    left = _slab(tmp_path, (0.4, 1.0, 0.2), (-0.3, 0.0, 0.1), "left")
    right = _slab(tmp_path, (0.4, 1.0, 0.2), (0.3, 0.0, 0.1), "right")
    graph = SceneGraph(
        objects=[
            _object("split", [left, right], (1.0, 1.0, 0.2)),
            _object("x", [], (0.1,) * 3, supported_by="split"),
        ]
    )
    heights = support.build(graph, settings)

    # Centred on the 0.2 m gap, wide enough that its corners land on both hulls.
    got = heights.under(
        graph.get("split"), np.array([-0.35, -0.2, 0.0]), np.array([0.35, 0.2, 0.0])
    )
    assert got == pytest.approx(0.2, abs=1e-3)


def test_falls_back_to_the_visual_box_without_collision_geometry(settings):
    """The hand-authored fixtures carry no collision paths, and neither does a graph
    that has not been through the inertia stage yet."""
    parent = _object("parent", [], dims=(1.0, 1.0, 0.4))
    graph = SceneGraph(objects=[parent, _object("x", [], (0.1,) * 3, supported_by="parent")])
    heights = support.build(graph, settings)

    got = heights.under(parent, np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.0]))
    assert got == pytest.approx(0.2, abs=1e-3), "half of the 0.4 m box, centred on its origin"


def test_only_builds_grids_for_things_that_support_something(tmp_path, settings):
    plate = _slab(tmp_path, (1.0, 1.0, 0.2), (0.0, 0.0, 0.1), "plate")
    graph = SceneGraph(
        objects=[
            _object("holds", [plate], (1.0, 1.0, 0.2)),
            _object("alone", [plate], (1.0, 1.0, 0.2), position=(5.0, 0.0, 0.0)),
            _object("x", [], (0.1,) * 3, supported_by="holds"),
        ]
    )
    heights = support.build(graph, settings)

    assert heights.under(graph.get("holds"), np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.0]))
    assert (
        heights.under(graph.get("alone"), np.array([4.9, -0.1, 0.0]), np.array([5.1, 0.1, 0.0]))
        is None
    )
