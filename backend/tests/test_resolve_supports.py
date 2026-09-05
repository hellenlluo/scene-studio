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
