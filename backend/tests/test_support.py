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


# --- what cannot be holding it up ---------------------------------------------


def _overhang(tmp_path, settings, intermediate: bool):
    """A wide board on a narrow table, with something standing on the overhanging end.

    Under the overhang there is nothing but floor 500 mm down, so the nearest surface
    to those underside points is the *bottom* of the object standing on the board —
    60 mm up, well inside the 100 mm `burial_slack_m` window. `contact` takes the
    minimum gap over the board's underside, so that one point outvotes the points
    genuinely resting on the table.

    `intermediate` puts a tray between the two, which is the same error one hop
    further away.
    """
    table = _object(
        "table",
        [_slab(tmp_path, (0.2, 0.2, 0.5), (0.0, 0.0, 0.0), "t")],
        dims=(0.2, 0.2, 0.5),
        position=(0.0, 0.0, 0.25),
    )
    board = _object(
        "board",
        [_slab(tmp_path, (0.6, 0.2, 0.06), (0.0, 0.0, 0.0), "b")],
        dims=(0.6, 0.2, 0.06),
        position=(0.2, 0.0, 0.53),
        supported_by="table",
    )
    objects = [table, board]

    standing_on, base = "board", 0.56
    if intermediate:
        objects.append(
            _object(
                "tray",
                [_slab(tmp_path, (0.1, 0.1, 0.02), (0.0, 0.0, 0.0), "y")],
                dims=(0.1, 0.1, 0.02),
                position=(0.0, 0.0, 0.57),
                supported_by="board",
            )
        )
        standing_on, base = "tray", 0.58

    objects.append(
        _object(
            "book",
            [_slab(tmp_path, (0.2, 0.2, 0.06), (0.0, 0.0, 0.0), "u")],
            dims=(0.2, 0.2, 0.06),
            position=(0.35, 0.0, base + 0.03),
            supported_by=standing_on,
        )
    )
    # Something on the book, because `build` only rasterises an object that supports
    # something — and in `room2.png` the book in question had a book on it.
    objects.append(
        _object(
            "crumb",
            [_slab(tmp_path, (0.02, 0.02, 0.01), (0.0, 0.0, 0.0), "c")],
            dims=(0.02, 0.02, 0.01),
            position=(0.35, 0.0, base + 0.065),
            supported_by="book",
        )
    )

    graph = SceneGraph(objects=objects)
    heights = support.build(graph, settings)
    return heights.contact(board, graph.objects, graph.floor_height_m, 0.005)


def test_an_object_resting_on_this_one_is_not_a_candidate_support(tmp_path, settings):
    """The failure this exclusion exists for. `contact` is deliberately parent-free —
    an object rests on whatever is beneath it — but the one set of objects that
    cannot be beneath it is the set resting on it, and the `burial_slack_m` window is
    wide enough to let one in.

    Measured on `room2.png`: a book was reported resting 98.87 mm *inside* the book
    standing on it, one shave under the 100 mm slack, which is the signature of a
    bound doing the choosing. Its real clearance to the side table it rests on was
    4.66 mm. It cost `certify.scale` a false failure and `certify.repair` a 98.9 mm
    snap proposed and rejected in every one of its five rounds.
    """
    contact = _overhang(tmp_path, settings, intermediate=False)
    assert contact.resting_on == "table"
    assert contact.gap == pytest.approx(0.0, abs=2e-3)


def test_the_exclusion_is_transitive(tmp_path, settings):
    """A mug on a tray on a board is two hops up and still cannot be under the board.

    `reconcile.resolve_supports` reads `resting_on` straight back into
    `supported_by`, so adopting one of these would close a cycle in the support
    graph — and a support graph with a cycle has no floor to settle against.
    """
    contact = _overhang(tmp_path, settings, intermediate=True)
    assert contact.resting_on == "table"
    assert contact.gap == pytest.approx(0.0, abs=2e-3)


def test_a_neighbour_overlapping_only_in_plan_view_is_not_a_support(tmp_path, settings):
    """A plant standing beside a vase on the same shelf, its foliage reaching over the
    bit of the vase that overhangs the shelf edge.

    The two do not touch — the foliage clears the vase's top — but they share columns
    in plan view, and over the overhang there is no shelf, so the only surface
    anywhere near those points is the plant's, 50 mm *above* them.
    `grid.sample`'s "under the whole parent" fallback hands it back, it beats the
    floor half a metre down, and `contact` takes the minimum gap over the underside —
    so a few overhanging points outvote every point genuinely resting on the shelf.

    Measured on `room2.png`: a vase resting on its shelf at +2.55 mm over 1640 of its
    1658 underside points was reported 93.97 mm inside the plant beside it, on 51
    points near its rim. `certify.stability` measured the pair as not touching at all.

    The discriminator is not a tolerance. Burial *straddles* the point — a mug 50 mm
    into a table has the tabletop above its underside and the table's legs below —
    and a neighbour does not. Every one of those 51 columns was entirely above the
    point: lowest surface 1.9970 against a point at 1.9509.
    """
    shelf = _object(
        "shelf",
        [_slab(tmp_path, (1.0, 0.4, 0.04), (0.0, 0.0, 0.0), "s")],
        dims=(1.0, 0.4, 0.04),
        position=(0.0, 0.0, 0.48),  # top at 0.50, out to x = 0.5
    )
    vase = _object(
        "vase",
        [_slab(tmp_path, (0.2, 0.2, 0.04), (0.0, 0.0, 0.0), "v")],
        dims=(0.2, 0.2, 0.04),
        # Base 0.50 on the shelf, top 0.54, overhanging the edge from x = 0.5 to 0.55.
        position=(0.45, 0.0, 0.52),
        supported_by="shelf",
    )
    plant = _object(
        "plant",
        [
            _slab(tmp_path, (0.04, 0.04, 0.05), (0.0, 0.0, 0.025), "stem"),
            # Foliage from 0.55 to 0.59: clear of the vase's 0.54 top, so nothing
            # touches, and within `burial_slack_m` of the vase's 0.50 underside.
            _slab(tmp_path, (0.3, 0.3, 0.04), (0.0, 0.0, 0.07), "leaves"),
        ],
        dims=(0.3, 0.3, 0.09),
        position=(0.62, 0.0, 0.50),
        supported_by="shelf",
    )
    graph = SceneGraph(objects=[shelf, vase, plant])

    # Grids for every object, which is what `reconcile.resolve_supports` asks for —
    # the default candidate set is the declared parents, and a plant that supports
    # nothing never gets a grid to be wrongly chosen from.
    heights = support.build(graph, settings, candidates={o.object_id for o in graph.objects})
    contact = heights.contact(vase, graph.objects, graph.floor_height_m, 0.005)

    assert contact.resting_on == "shelf"
    assert contact.gap == pytest.approx(0.0, abs=2e-3)
