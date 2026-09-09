"""Geometric validation of the support relations the VLM guessed.

`ObjectLabel.support_parent` comes from stage 3, which looks at a flat photo two
stages before any 3D exists. `schemas.ObjectLabel` has always documented that
reconcile overrides it with measured contact; this is that override.

The load-bearing test is `test_a_claim_the_geometry_contradicts_is_replaced`. It
is the observed failure: on one run the labeller called a sofa rug-supported when
the sofa's footprint centre was 0.19 m clear of the rug, the scale axis failed
`base_inside_parent`, and the solver drove the sofa onto a surface it was not over.

`test_an_unverifiable_claim_is_left_alone` is the other half, and the reason this
validates rather than re-derives: at this stage nothing has been solved, so the
geometry is the raw reconstruction and is quite capable of preferring the wrong
parent. Refuting a claim is evidence; failing to confirm one is not.
"""

import pytest
import trimesh

from app.config import get_settings
from app.pipeline.reconcile import resolve_supports
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


def _slab(tmp_path, extents, name) -> str:
    path = tmp_path / f"{name}.obj"
    path.write_text(trimesh.creation.box(extents=extents).export(file_type="obj"))
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


def _supports(graph):
    return {o.object_id: o.supported_by for o in graph.objects}


def test_a_claim_the_geometry_contradicts_is_replaced(tmp_path, settings):
    """The sofa-and-rug case: claimed on the mat, but standing beside it."""
    mat = _slab(tmp_path, (1.0, 1.0, 0.02), "mat")
    block = _slab(tmp_path, (0.5, 0.5, 0.4), "block")
    graph = SceneGraph(
        objects=[
            _object("mat", [mat], (1.0, 1.0, 0.02), (0.0, 0.0, 0.01)),
            # Centred 1.2 m away — its box may reach the mat, its middle does not.
            _object("sofa", [block], (0.5, 0.5, 0.4), (1.2, 0.0, 0.2), supported_by="mat"),
        ]
    )
    assert _supports(resolve_supports(graph, settings))["sofa"] is None


def test_a_claim_the_geometry_confirms_is_kept(tmp_path, settings):
    mat = _slab(tmp_path, (1.0, 1.0, 0.02), "mat")
    block = _slab(tmp_path, (0.5, 0.5, 0.4), "block")
    graph = SceneGraph(
        objects=[
            _object("mat", [mat], (1.0, 1.0, 0.02), (0.0, 0.0, 0.01)),
            _object("sofa", [block], (0.5, 0.5, 0.4), (0.0, 0.0, 0.22), supported_by="mat"),
        ]
    )
    assert _supports(resolve_supports(graph, settings))["sofa"] == "mat"


def test_an_unverifiable_claim_is_left_alone(tmp_path, settings):
    """Refuting a claim is evidence; failing to confirm one is not.

    The mug is a metre above the table it says it is on — too far for any contact
    to be measured, and too far from the floor to call it floor-supported. The
    reconstruction is what is wrong here, not the label, and the solver can still
    close a gap it has been told about.
    """
    table = _slab(tmp_path, (1.0, 1.0, 0.05), "table")
    mug = _slab(tmp_path, (0.1, 0.1, 0.1), "mug")
    graph = SceneGraph(
        objects=[
            _object("table", [table], (1.0, 1.0, 0.05), (0.0, 0.0, 0.5)),
            _object("mug", [mug], (0.1, 0.1, 0.1), (0.0, 0.0, 1.6), supported_by="table"),
        ]
    )
    assert _supports(resolve_supports(graph, settings))["mug"] == "table"


def test_the_nearest_surface_wins(tmp_path, settings):
    """Claimed on the lower of two stacked surfaces; resolved onto the one it rests on."""
    low = _slab(tmp_path, (2.0, 2.0, 0.1), "low")
    high = _slab(tmp_path, (1.0, 1.0, 0.1), "high")
    mug = _slab(tmp_path, (0.1, 0.1, 0.1), "mug")
    graph = SceneGraph(
        objects=[
            _object("low", [low], (2.0, 2.0, 0.1), (0.0, 0.0, 0.05)),
            _object("high", [high], (1.0, 1.0, 0.1), (0.0, 0.0, 0.25)),
            _object("mug", [mug], (0.1, 0.1, 0.1), (0.0, 0.0, 0.35), supported_by="low"),
        ]
    )
    assert _supports(resolve_supports(graph, settings))["mug"] == "high"


def test_the_result_is_acyclic(tmp_path, settings):
    """Only a lower base can hold something up, which makes a cycle unconstructible.

    A support cycle has no floor to settle against, so MuJoCo cannot resolve it and
    `SceneObject` would reject the graph outright.
    """
    piece = _slab(tmp_path, (1.0, 1.0, 0.2), "piece")
    graph = SceneGraph(
        objects=[
            _object("a", [piece], (1.0, 1.0, 0.2), (0.0, 0.0, 0.1), supported_by="b"),
            _object("b", [piece], (1.0, 1.0, 0.2), (0.0, 0.0, 0.3), supported_by="a"),
        ]
    )
    resolved = _supports(resolve_supports(graph, settings))

    seen, node = set(), "b"
    while node is not None:
        assert node not in seen, "cycle in the resolved support graph"
        seen.add(node)
        node = resolved[node]


def test_an_empty_graph_is_returned_unchanged(settings):
    graph = SceneGraph(objects=[])
    assert resolve_supports(graph, settings).objects == []


def test_a_near_tie_keeps_the_label(tmp_path, settings):
    """Nearest-surface alone is wrong when two surfaces are nearly the same height.

    The observed failure on `room2.png`: the labeller correctly put a book on the
    side table, and the nearest surface under the book's centre was the *saucer*
    beside it, 10 mm higher. A book resting on a saucer is not a thing, and nothing
    downstream can tell — the gap closes either way, so the error is invisible until
    someone reads the graph.

    Geometry has to beat the label by `support_claim_margin_m` before it overrules
    it. Ten millimetres is disagreement between two surfaces at the same contact,
    not evidence of a different one.
    """
    table = _slab(tmp_path, (1.0, 1.0, 0.05), "table")
    saucer = _slab(tmp_path, (0.15, 0.15, 0.01), "saucer")
    book = _slab(tmp_path, (0.3, 0.2, 0.03), "book")
    graph = SceneGraph(
        objects=[
            _object("table", [table], (1.0, 1.0, 0.05), (0.0, 0.0, 0.5)),
            _object("saucer", [saucer], (0.15, 0.15, 0.01), (0.0, 0.0, 0.530)),
            # Base at 0.540: 15 mm over the tabletop, 5 mm over the saucer.
            _object("book", [book], (0.3, 0.2, 0.03), (0.0, 0.0, 0.555), supported_by="table"),
        ]
    )
    assert _supports(resolve_supports(graph, settings))["book"] == "table"


def test_an_object_on_bare_floor_still_reads_as_floor(tmp_path, settings):
    """The demotion must not invent a support. With nothing else underneath, the
    floor is the only viable candidate and still wins."""
    block = _slab(tmp_path, (0.5, 0.5, 0.4), "block")
    graph = SceneGraph(
        objects=[_object("chair", [block], (0.5, 0.5, 0.4), (0.0, 0.0, 0.2), supported_by=None)]
    )
    assert _supports(resolve_supports(graph, settings))["chair"] is None


# --- run twice, because the pipeline now does ---------------------------------


def test_resolving_twice_changes_nothing_the_second_time(tmp_path, settings):
    """The orchestrator runs this twice: once inside `reconcile.run`, and again after
    stage 9, which is where `collision_mesh_paths` first exists. Before that every
    part falls back to its visual mesh or its OBB box, so the first pass decides the
    support relation from geometry no later stage uses — measured on `room2.png`,
    0 collision meshes at stage 5 against 263 after stage 9, and re-running this
    unchanged on the post-stage-9 graph moves an armchair off the floor and onto the
    rug it measurably rests on.

    A second pass is only safe if it is a fixed point on geometry that has not
    changed. It has to be: the rule reads the measurement, and the measurement does
    not depend on what the relation currently says.
    """
    low = _slab(tmp_path, (2.0, 2.0, 0.1), "low")
    high = _slab(tmp_path, (1.0, 1.0, 0.1), "high")
    mug = _slab(tmp_path, (0.1, 0.1, 0.1), "mug")
    graph = SceneGraph(
        objects=[
            _object("low", [low], (2.0, 2.0, 0.1), (0.0, 0.0, 0.05)),
            _object("high", [high], (1.0, 1.0, 0.1), (0.0, 0.0, 0.25)),
            _object("mug", [mug], (0.1, 0.1, 0.1), (0.0, 0.0, 0.35), supported_by="low"),
        ]
    )
    once = resolve_supports(graph, settings)
    twice = resolve_supports(once, settings)
    assert _supports(once) == {"low": None, "high": "low", "mug": "high"}
    assert _supports(twice) == _supports(once)


# --- the user-edit path -------------------------------------------------------
#
# `reconsider` names objects the user has just dragged. Two things change for
# them: only they are rewritten, and their recorded parent stops being a claim to
# defend — a drag carries a new position and no new label, so the stored edge
# describes where the object *was*.


def test_a_dragged_object_adopts_the_surface_it_was_dropped_on(tmp_path, settings):
    """The reported bug, at the unit level.

    Measured end to end on `room2` before this existed: a book recorded as
    floor-supported and dropped onto the side table came back from solve at
    z=0.015, underneath it, because the support term was still closing the gap to
    the floor. Nothing was wrong with the solver — it was told the floor.
    """
    table = _slab(tmp_path, (1.0, 1.0, 0.05), "table")
    book = _slab(tmp_path, (0.2, 0.2, 0.04), "book")
    graph = SceneGraph(
        objects=[
            _object("table", [table], (1.0, 1.0, 0.05), (0.0, 0.0, 0.5)),
            # Resting on the tabletop, but still recorded as floor-supported.
            _object("book", [book], (0.2, 0.2, 0.04), (0.0, 0.0, 0.545)),
        ]
    )
    assert _supports(graph)["book"] is None
    assert _supports(resolve_supports(graph, settings, {"book"}))["book"] == "table"


def test_reconsidering_leaves_every_other_object_alone(tmp_path, settings):
    """A drag must not quietly reparent the far side of the room.

    The whole reason repair is scoped on this path too — see
    `api.scenes.edit_scene`. An unbounded pass here would undo that at the first
    step.
    """
    mat = _slab(tmp_path, (1.0, 1.0, 0.02), "mat")
    block = _slab(tmp_path, (0.5, 0.5, 0.4), "block")
    book = _slab(tmp_path, (0.2, 0.2, 0.04), "book")
    graph = SceneGraph(
        objects=[
            _object("mat", [mat], (1.0, 1.0, 0.02), (0.0, 0.0, 0.01)),
            # Contradicted by the geometry — an unscoped pass would demote it.
            _object("sofa", [block], (0.5, 0.5, 0.4), (1.2, 0.0, 0.2), supported_by="mat"),
            _object("book", [book], (0.2, 0.2, 0.04), (0.0, 0.0, 0.04)),
        ]
    )
    resolved = resolve_supports(graph, settings, {"book"})
    assert _supports(resolved)["sofa"] == "mat", "an untouched object was reparented"
    assert _supports(resolve_supports(graph, settings))["sofa"] is None, (
        "the unscoped pass should still demote it, or this test proves nothing"
    )


def test_a_stale_edge_cannot_defend_itself(tmp_path, settings):
    """The margin that protects a *claim* must not protect a *memory*.

    Without clearing it, `support_claim_margin_m` keeps the old parent whenever it
    is within 50 mm of the best candidate — so nudging a mug from one coaster to
    an adjacent one of the same height would leave it recorded on the first.
    """
    left = _slab(tmp_path, (0.3, 0.3, 0.05), "left")
    right = _slab(tmp_path, (0.3, 0.3, 0.05), "right")
    mug = _slab(tmp_path, (0.1, 0.1, 0.1), "mug")
    graph = SceneGraph(
        objects=[
            _object("left", [left], (0.3, 0.3, 0.05), (-0.4, 0.0, 0.025)),
            _object("right", [right], (0.3, 0.3, 0.05), (0.4, 0.0, 0.025)),
            # Sitting on `right`, still recorded on `left`: the drag that moved it.
            _object("mug", [mug], (0.1, 0.1, 0.1), (0.4, 0.0, 0.1), supported_by="left"),
        ]
    )
    assert _supports(resolve_supports(graph, settings, {"mug"}))["mug"] == "right"


def test_a_drop_into_open_space_keeps_its_edge(tmp_path, settings):
    """The fallback is the same as the pipeline's, and that took a wrong turn first.

    An earlier version dropped a stale edge to the floor here, reasoning that a
    drop into open space has no parent. It reads well and it destroys the useful
    information: measured on `room2`, the plate 138 mm off the side table lost its
    recorded parent, and with `supported_by=None` every strategy that could put it
    back declined — the floor is the one support an object is always over, so
    `repair._slide_onto_support` skips it. The edge says where the object belongs;
    the geometry says where it is, and keeping the edge is what lets the rest of
    the system close the gap.
    """
    table = _slab(tmp_path, (1.0, 1.0, 0.05), "table")
    mug = _slab(tmp_path, (0.1, 0.1, 0.1), "mug")
    graph = SceneGraph(
        objects=[
            _object("table", [table], (1.0, 1.0, 0.05), (0.0, 0.0, 0.5)),
            # A metre up: nothing measurable under it, and too far from the floor.
            _object("mug", [mug], (0.1, 0.1, 0.1), (0.0, 0.0, 1.6), supported_by="table"),
        ]
    )
    assert _supports(resolve_supports(graph, settings, {"mug"}))["mug"] == "table"
