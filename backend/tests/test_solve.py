"""Stage 6: the coupled solve over scale and pose.

The stage the whole project rests on, and until now the only one with no direct
tests — every change to it was verified by a one-off script against a single
reconstructed scene, which checks that it works on that scene and nothing about
why.

**Depth is injected, not read.** `solve.run` gets its measurements by calling
`reconcile.observe`, which loads a depth map and mask PNGs off disk. Driving that
from a test would mean authoring a depth map whose backprojection happens to say
what the test means, and the assertion would then be about the fixture rather than
the solver. Patching `observe` states the measurement directly: *depth says this
object is here, this big*. That is exactly the contract stage 6 consumes.

`test_depth_alone_does_not_fix_scale` and `test_an_anchor_still_works_without_depth`
are the two halves of row 1 of the ablation in the design doc: which term actually
carries absolute size. Written as tests so the claim stays true rather than being
measured once and quoted forever. The prior-only row cannot be written yet —
`E_prior` has no measured priors table behind it, which is the same reason
`DimensionPrior` is still None on every object.

The multi-piece fixtures centre their geometry on the part origin deliberately.
`dims_m` is an extent *about* that origin and `world_aabb` believes it, so a mesh
spanning 0 to 0.7 whose box claims -0.35 to 0.35 is a fixture lying to the solver —
which duly lifted the whole sofa 0.35 m trying to satisfy both.
"""

import numpy as np
import pytest
import trimesh

from app.pipeline import reconcile, solve
from app.pipeline.base import PipelineContext
from app.schemas import (
    AssetFrame,
    DepthResult,
    Intrinsics,
    ObjectLabel,
    PartGeometry,
    ScaleAnchor,
    SceneGraph,
    SceneObject,
    SegmentResult,
    SolveWeights,
)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.write_bytes(b"nothing reads this; observe is patched")
    context = PipelineContext.create("solve-test", image)
    monkeypatch.setattr(context.settings, "storage_dir", tmp_path)
    return context


@pytest.fixture
def inputs():
    """The two arguments `observe` would have consumed, and does not once patched."""
    depth = DepthResult(
        depth_path="unused.npy",
        intrinsics=Intrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=400.0, width=1000, height=800),
        is_metric=True,
    )
    return depth, SegmentResult(masks=[], image_size_px=(1000, 800))


def observed(monkeypatch, **by_id):
    """Patch `observe` to report exactly these (centre, extent) per object.

    `footprint_xy` is unused by stage 6 — it exists for reconcile's yaw fit — so it
    is left empty rather than invented.
    """
    table = {
        object_id: reconcile.Observation(
            centre=np.asarray(centre, dtype=float),
            extent=np.asarray(extent, dtype=float),
            footprint_xy=np.empty((0, 2)),
        )
        for object_id, (centre, extent) in by_id.items()
    }
    monkeypatch.setattr(reconcile, "observe", lambda *_: (table, np.eye(3), 0.0))


def box_part(tmp_path, dims, name) -> PartGeometry:
    """A part whose collision geometry is a real file, so support and penetration
    measure a mesh rather than falling back to the OBB."""
    mesh = trimesh.creation.box(extents=dims)
    path = tmp_path / f"{name}.obj"
    path.write_text(mesh.export(file_type="obj"))
    return PartGeometry(part_id="body", name="body", dims_m=dims, collision_mesh_paths=[str(path)])


def obj(object_id, part, position, scale=1.0, supported_by=None) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(object_id=object_id, category=object_id),
        frame=AssetFrame(source="test"),
        parts=[part],
        position_m=position,
        scale=scale,
        supported_by=supported_by,
    )


def support_gap(graph, child_id) -> float:
    """Signed gap at the contact, measured the way the solver and certifier both do."""
    from app.config import get_settings
    from app.geometry import world_aabb
    from app.pipeline import support

    child = graph.get(child_id)
    parent = graph.get(child.supported_by)
    heights = support.build(graph, get_settings())
    low, high = world_aabb(child)
    surface = heights.under(parent, low, high)
    base = heights.base_of(child)
    return base - surface


# --- the contact it exists to close -------------------------------------------


def test_a_sunk_object_rises_onto_its_support(ctx, inputs, tmp_path, monkeypatch):
    """A mug seeded 5 cm inside a tabletop ends up resting on it."""
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug",
        box_part(tmp_path, (0.1, 0.1, 0.1), "mug"),
        (0.0, 0.0, 0.50),  # base at 0.45 — 50 mm into a tabletop whose surface is 0.50
        supported_by="table",
    )
    graph = SceneGraph(objects=[table, mug])
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        mug=((0.0, 0.0, 0.55), (0.1, 0.1, 0.1)),
    )

    assert support_gap(graph, "mug") == pytest.approx(-0.05, abs=2e-3), "starts sunk"
    result = solve.run(ctx, graph, *inputs, SolveWeights())
    assert abs(support_gap(result.graph, "mug")) < 0.005, "ends resting"


def test_a_floating_object_drops_onto_its_support(ctx, inputs, tmp_path, monkeypatch):
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug", box_part(tmp_path, (0.1, 0.1, 0.1), "mug"), (0.0, 0.0, 0.70), supported_by="table"
    )
    graph = SceneGraph(objects=[table, mug])
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        mug=((0.0, 0.0, 0.55), (0.1, 0.1, 0.1)),
    )

    assert support_gap(graph, "mug") == pytest.approx(0.15, abs=2e-3), "starts floating"
    result = solve.run(ctx, graph, *inputs, SolveWeights())
    assert abs(support_gap(result.graph, "mug")) < 0.005


def test_contact_outweighs_depth_on_the_vertical_axis(ctx, inputs, tmp_path, monkeypatch):
    """Depth insists the mug floats; the contact wins, and that ordering is chosen.

    Depth contributes six residuals per object against one for support, so equal
    weights are a six-to-one loss for contact — the reason `SolveWeights.support`
    is 8. A reconstructed table sat 17.8 mm clear of its rug at the old value.
    """
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug", box_part(tmp_path, (0.1, 0.1, 0.1), "mug"), (0.0, 0.0, 0.55), supported_by="table"
    )
    graph = SceneGraph(objects=[table, mug])
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        mug=((0.0, 0.0, 0.65), (0.1, 0.1, 0.1)),  # depth is 100 mm too high
    )

    result = solve.run(ctx, graph, *inputs, SolveWeights())
    assert abs(support_gap(result.graph, "mug")) < 0.02, "contact held against depth"


# --- anchors ------------------------------------------------------------------


def test_an_anchor_fixes_the_dimension_it_names(ctx, inputs, tmp_path, monkeypatch):
    """`E_prior` with sigma -> 0. Depth says 1.0 m wide; the user says 1.5 m."""
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "table")
    graph = SceneGraph(objects=[obj("table", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)))

    result = solve.run(
        ctx, graph, *inputs, SolveWeights(), [ScaleAnchor(object_id="table", axis=0, value_m=1.5)]
    )
    assert result.graph.get("table").dims_m[0] == pytest.approx(1.5, abs=0.02)


def test_an_anchor_propagates_through_the_support_graph(ctx, inputs, tmp_path, monkeypatch):
    """The claim the whole coupled formulation is for.

    Anchoring the table alone rescales the mug on it, because scale is one global
    degree of freedom and the support contact carries it. Nothing tells the mug its
    own size: its depth observation still describes the unanchored scene.
    """
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug", box_part(tmp_path, (0.1, 0.1, 0.1), "mug"), (0.0, 0.0, 0.55), supported_by="table"
    )
    graph = SceneGraph(objects=[table, mug])
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        mug=((0.0, 0.0, 0.55), (0.1, 0.1, 0.1)),
    )

    anchored = solve.run(
        ctx, graph, *inputs, SolveWeights(), [ScaleAnchor(object_id="table", axis=2, value_m=0.8)]
    ).graph
    # The tabletop rose from 0.50 to 0.80, so the mug has to follow it up or the
    # contact it is party to opens by 300 mm.
    assert abs(support_gap(anchored, "mug")) < 0.01, "the mug stayed on the table"
    assert anchored.get("mug").position_m[2] > 0.70, "and moved with it"


# --- the ablation -------------------------------------------------------------


def test_depth_alone_does_not_fix_scale(ctx, inputs, tmp_path, monkeypatch):
    """Ablation row 1: with no depth term, nothing pins absolute size.

    `E_prior` has no measured priors table behind it yet, so with depth switched off
    the only scale information left is the anchors — and with none supplied, an
    object is free to be any size the bounds allow. That is the honest statement of
    where absolute scale currently comes from.
    """
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "table")
    graph = SceneGraph(objects=[obj("table", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, table=((0.0, 0.0, 0.25), (2.0, 2.0, 1.0)))  # depth says twice as big

    with_depth = solve.run(ctx, graph, *inputs, SolveWeights()).graph
    without = solve.run(ctx, graph, *inputs, SolveWeights(depth=0.0)).graph

    assert with_depth.get("table").scale == pytest.approx(2.0, abs=0.1), "depth sets the size"
    assert without.get("table").scale == pytest.approx(1.0, abs=1e-6), "nothing else does"


def test_an_anchor_still_works_without_depth(ctx, inputs, tmp_path, monkeypatch):
    """The other half of row 1: a user dimension is scale information depth is not.

    This is why one typed number is worth so much — it is the only term in the
    objective that carries absolute size on its own.
    """
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "table")
    graph = SceneGraph(objects=[obj("table", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)))

    result = solve.run(
        ctx,
        graph,
        *inputs,
        SolveWeights(depth=0.0),
        [ScaleAnchor(object_id="table", axis=0, value_m=1.6)],
    )
    assert result.graph.get("table").dims_m[0] == pytest.approx(1.6, abs=0.02)


# --- what the diagnostics mean ------------------------------------------------


def test_a_settled_scene_reports_both_flags(ctx, inputs, tmp_path, monkeypatch):
    """One object resting on the floor, already where it belongs: nothing to do."""
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "block")
    graph = SceneGraph(objects=[obj("block", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, block=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)))

    diagnostics = solve.run(ctx, graph, *inputs, SolveWeights()).diagnostics
    assert diagnostics.converged and diagnostics.settled
    assert diagnostics.max_settle_drift_m < 0.002


def test_an_unstable_object_blocks_settled_but_not_converged(ctx, inputs, tmp_path, monkeypatch):
    """The two flags are separate because one object can fail only the second.

    A tall thin block on a small base topples however well it is placed. The
    optimiser is not at fault and should not report as though it were — which is
    what a single flag did, on a real scene, for four rounds.
    """
    part = box_part(tmp_path, (0.05, 0.05, 2.0), "pole")
    graph = SceneGraph(objects=[obj("pole", part, (0.0, 0.0, 1.0))])
    observed(monkeypatch, pole=((0.0, 0.0, 1.0), (0.05, 0.05, 2.0)))
    # Tipped a little off vertical, so gravity has a lever to work with.
    graph.objects[0].orientation = (0.9999, 0.0, 0.0144, 0.0)

    diagnostics = solve.run(ctx, graph, *inputs, SolveWeights()).diagnostics
    assert not diagnostics.settled
    assert diagnostics.max_settle_drift_m > 0.01


def test_an_empty_graph_is_returned_untouched(ctx, inputs):
    """Both flags true: nothing to move, and nothing to settle."""
    result = solve.run(ctx, SceneGraph(objects=[]), *inputs, SolveWeights())
    assert result.graph.objects == []
    assert result.diagnostics.converged and result.diagnostics.settled


# --- integration with the geometry modules ------------------------------------


def test_the_support_target_is_the_seat_not_the_bounding_box(ctx, inputs, tmp_path, monkeypatch):
    """End to end through `support`: a cushion lands on the seat, not the backrest.

    `test_support.py` proves the height query returns the seat. This proves stage 6
    actually solves against it — the two were disconnected for as long as the
    residual read a box top, and that gap is what put every cushion 23 cm high.
    """
    # Centred on the part origin, because `dims_m` is an extent about that origin and
    # `world_aabb` believes it. A mesh sitting 0 to 0.7 while its box claims -0.35 to
    # 0.35 is a fixture that lies to the solver, and it lifted the whole sofa by
    # 0.35 m trying to reconcile the two.
    seat = trimesh.creation.box(extents=(1.0, 0.5, 0.4))
    seat.apply_translation((0.0, 0.0, -0.15))
    back = trimesh.creation.box(extents=(1.0, 0.1, 0.3))
    back.apply_translation((0.0, 0.2, 0.2))
    paths = []
    for piece, name in ((seat, "seat"), (back, "back")):
        path = tmp_path / f"{name}.obj"
        path.write_text(piece.export(file_type="obj"))
        paths.append(str(path))

    sofa = obj(
        "sofa",
        PartGeometry(
            part_id="body", name="body", dims_m=(1.0, 0.5, 0.7), collision_mesh_paths=paths
        ),
        (0.0, 0.0, 0.35),  # standing on the floor: geometry now spans 0.0 to 0.7
    )
    cushion = obj(
        "cushion",
        box_part(tmp_path, (0.3, 0.3, 0.1), "cushion"),
        (0.0, -0.1, 0.75),  # up at backrest height, which is where the box top is
        supported_by="sofa",
    )
    graph = SceneGraph(objects=[sofa, cushion])
    observed(
        monkeypatch,
        sofa=((0.0, 0.0, 0.35), (1.0, 0.5, 0.7)),
        cushion=((0.0, -0.1, 0.45), (0.3, 0.3, 0.1)),
    )

    result = solve.run(ctx, graph, *inputs, SolveWeights())
    base = result.graph.get("cushion").position_m[2] - 0.05
    assert base == pytest.approx(0.4, abs=0.02), "the seat at 0.40, not the backrest at 0.70"


def test_a_rug_under_a_table_is_not_pushed_away(ctx, inputs, tmp_path, monkeypatch):
    """End to end through `penetration`: overlapping boxes, no overlapping meshes.

    With the AABB term this pair read as a 20 mm collision and the solver spent
    every iteration separating them. Measured on a real scene at 14.5 mm.
    """
    # Centred on the part origin; see the sofa fixture above for why that matters.
    top = trimesh.creation.box(extents=(1.0, 1.0, 0.05))
    top.apply_translation((0.0, 0.0, 0.225))
    legs = []
    for x, y in ((0.45, 0.45), (0.45, -0.45), (-0.45, 0.45), (-0.45, -0.45)):
        leg = trimesh.creation.box(extents=(0.06, 0.06, 0.45))
        leg.apply_translation((x, y, -0.025))
        legs.append(leg)
    paths = []
    for piece, name in [(top, "top"), *((leg, f"leg{i}") for i, leg in enumerate(legs))]:
        path = tmp_path / f"{name}.obj"
        path.write_text(piece.export(file_type="obj"))
        paths.append(str(path))

    table = obj(
        "table",
        PartGeometry(
            part_id="body", name="body", dims_m=(1.0, 1.0, 0.5), collision_mesh_paths=paths
        ),
        (0.0, 0.0, 0.25),  # standing on the floor
    )
    rug = obj("rug", box_part(tmp_path, (0.5, 0.5, 0.02), "rug"), (0.0, 0.0, 0.01))
    graph = SceneGraph(objects=[table, rug])
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        rug=((0.0, 0.0, 0.01), (0.5, 0.5, 0.02)),
    )

    result = solve.run(ctx, graph, *inputs, SolveWeights())
    moved = np.linalg.norm(
        np.asarray(result.graph.get("rug").position_m) - np.asarray(rug.position_m)
    )
    assert moved < 0.01, "nothing to resolve, so nothing should have moved"


def test_an_object_with_no_observation_keeps_its_size(ctx, inputs, tmp_path, monkeypatch):
    """Nothing else in the objective carries absolute size.

    `E_prior` has no priors table yet, and support and penetration only care about
    surfaces meeting — so an unmeasured object can close a contact by shrinking as
    easily as by moving, and will. Measured on a hand-authored fixture: a mug
    re-solved to a quarter of its size with a support gap of 1e-10.
    """
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug", box_part(tmp_path, (0.1, 0.1, 0.1), "mug"), (0.0, 0.0, 1.40), supported_by="table"
    )
    # Only the table is measured; the mug has been dragged into mid-air by a user.
    observed(monkeypatch, table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)))

    result = solve.run(ctx, SceneGraph(objects=[table, mug]), *inputs, SolveWeights())
    solved = result.graph.get("mug")

    assert solved.scale == pytest.approx(1.0, abs=1e-6), "unmeasured size is not touched"
    assert solved.position_m[2] < 1.0, "but it is still brought back down to its support"


def test_a_measured_object_may_still_be_resized(ctx, inputs, tmp_path, monkeypatch):
    """The freeze is about absent evidence, not a general refusal to resize."""
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "table")
    graph = SceneGraph(objects=[obj("table", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, table=((0.0, 0.0, 0.5), (2.0, 2.0, 1.0)))

    result = solve.run(ctx, graph, *inputs, SolveWeights())
    assert result.graph.get("table").scale == pytest.approx(2.0, abs=0.1)


def test_the_measurement_is_carried_for_a_later_re_solve(ctx, inputs, tmp_path, monkeypatch):
    """Committing an edit re-solves without the depth map, so the six numbers stage
    6 reads have to travel on the graph."""
    part = box_part(tmp_path, (1.0, 1.0, 0.5), "table")
    graph = SceneGraph(objects=[obj("table", part, (0.0, 0.0, 0.25))])
    observed(monkeypatch, table=((0.0, 0.0, 0.3), (1.2, 1.0, 0.5)))

    solved = solve.run(ctx, graph, *inputs, SolveWeights()).graph
    carried = solved.get("table").observation
    assert carried is not None
    assert carried.centre_m == pytest.approx((0.0, 0.0, 0.3))
    assert carried.extent_m == pytest.approx((1.2, 1.0, 0.5))

    # And a re-solve with neither depth nor segments reaches the same answer. Not
    # bit-identical: it starts from the solved pose rather than the reconstructed
    # one, so the trust-region takes a different path to the same place. Measured at
    # 1e-5 relative, asserted at 1e-3.
    again = solve.run(ctx, solved, None, None, SolveWeights()).graph
    assert again.get("table").scale == pytest.approx(solved.get("table").scale, rel=1e-3)


# --- containment ---------------------------------------------------------------


def _offset_xy(graph, child_id) -> float:
    """How far the child's footprint centre sits outside its parent's, in metres.

    Measured the way `certify.scale` measures `base_inside_parent`, so a test here
    and a failing check there are talking about the same quantity.
    """
    import numpy as np

    from app.geometry import world_aabb

    child = graph.get(child_id)
    low, high = world_aabb(child)
    parent_low, parent_high = world_aabb(graph.get(child.supported_by))
    centre = (low[:2] + high[:2]) / 2.0
    outside = np.maximum(0.0, np.maximum(parent_low[:2] - centre, centre - parent_high[:2]))
    return float(np.hypot(*outside))


def test_depth_cannot_slide_an_object_off_its_support(ctx, inputs, tmp_path, monkeypatch):
    """The failure this term exists for.

    `E_supp` is signed but vertical: it closes the gap to whatever an object rests
    on and is indifferent to *where*. So a depth measurement half a metre to the
    side used to be satisfiable at zero cost to contact — the book keeps a perfect
    0 mm gap while hovering over the floor beside the table. Measured on room2,
    four objects left the side table that way, by up to 1225 mm.
    """
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    book = obj(
        "book",
        box_part(tmp_path, (0.2, 0.2, 0.05), "book"),
        (0.0, 0.0, 0.525),
        supported_by="table",
    )
    # Depth is wrong by 0.8 m laterally, and is the only thing pulling sideways.
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        book=((0.8, 0.0, 0.525), (0.2, 0.2, 0.05)),
    )

    # Explicit, because the term ships off — see `SolveWeights.containment`.
    weights = SolveWeights(containment=60.0)
    result = solve.run(ctx, SceneGraph(objects=[table, book]), *inputs, weights)

    assert _offset_xy(result.graph, "book") < 0.001, "the book stayed over the table"
    assert abs(support_gap(result.graph, "book")) < 0.005, "and is still resting on it"


def test_containment_costs_nothing_when_the_object_is_already_over_its_support(
    ctx, inputs, tmp_path, monkeypatch
):
    """One-sided, like penetration. A correctly placed object must not be dragged
    toward its parent's centre, or the term would quietly recentre every scene."""
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    # Well inside the tabletop, but off-centre — where depth says it is.
    book = obj(
        "book",
        box_part(tmp_path, (0.2, 0.2, 0.05), "book"),
        (0.3, 0.0, 0.525),
        supported_by="table",
    )
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        book=((0.3, 0.0, 0.525), (0.2, 0.2, 0.05)),
    )

    weights = SolveWeights(containment=60.0)
    result = solve.run(ctx, SceneGraph(objects=[table, book]), *inputs, weights)

    assert result.graph.get("book").position_m[0] == pytest.approx(0.3, abs=0.02)


def test_a_frozen_scale_survives_a_second_round(ctx, inputs, tmp_path, monkeypatch):
    """`_bounds` freezes an unobserved object's scale to a 1e-12-wide interval, and
    the warm start round-trips through `exp` then `log`, which is not exactly the
    identity. Measured: 1e-12 came back as 1.0000889e-12 and `least_squares` refused
    the round outright. Clipping the warm start is what keeps a solve from failing
    over one ulp."""
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    mug = obj(
        "mug", box_part(tmp_path, (0.1, 0.1, 0.1), "mug"), (0.0, 0.0, 1.40), supported_by="table"
    )
    observed(monkeypatch, table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)))

    result = solve.run(ctx, SceneGraph(objects=[table, mug]), *inputs, SolveWeights())

    assert result.graph.get("mug").scale == pytest.approx(1.0, abs=1e-6), "scale stayed frozen"


def test_an_object_buried_in_its_own_support_is_pushed_out(ctx, inputs, tmp_path, monkeypatch):
    """`E_supp` measures the vertical gap under the footprint, which is not the same
    quantity as mesh overlap — a shelf-shaped parent can report a satisfied gap while
    the child is centimetres inside it. `penetration.pairs` drops support contacts, so
    before this term nothing in the objective could see that at all.

    Measured on room2: a 0.77 kg vase sat 15.8 mm inside the bookshelf it rests on,
    MuJoCo ejected it at 15.9 m/s, and the scene diverged after 0.446 s.
    """
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    vase = obj(
        "vase",
        box_part(tmp_path, (0.1, 0.1, 0.2), "vase"),
        (0.0, 0.0, 0.44),  # 60 mm of its 200 mm height is inside the tabletop
        supported_by="table",
    )
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        vase=((0.0, 0.0, 0.44), (0.1, 0.1, 0.2)),  # depth agrees with the bad pose
    )

    result = solve.run(ctx, SceneGraph(objects=[table, vase]), *inputs, SolveWeights())

    assert support_gap(result.graph, "vase") > -0.005, "no longer buried in the tabletop"


def test_a_resting_contact_is_not_penalised_twice(ctx, inputs, tmp_path, monkeypatch):
    """One-sided past the tolerance, so a correct contact costs nothing. Scoring a
    resting contact here as well as in `E_supp` would have the solver fighting itself,
    which is why `penetration.pairs` excludes support pairs in the first place."""
    table = obj("table", box_part(tmp_path, (1.0, 1.0, 0.5), "table"), (0.0, 0.0, 0.25))
    vase = obj(
        "vase",
        box_part(tmp_path, (0.1, 0.1, 0.2), "vase"),
        (0.0, 0.0, 0.60),  # resting exactly on the tabletop
        supported_by="table",
    )
    observed(
        monkeypatch,
        table=((0.0, 0.0, 0.25), (1.0, 1.0, 0.5)),
        vase=((0.0, 0.0, 0.60), (0.1, 0.1, 0.2)),
    )

    result = solve.run(ctx, SceneGraph(objects=[table, vase]), *inputs, SolveWeights())

    assert abs(support_gap(result.graph, "vase")) < 0.005, "still resting, not pushed off"
    assert result.graph.get("vase").position_m[2] == pytest.approx(0.60, abs=0.01)
