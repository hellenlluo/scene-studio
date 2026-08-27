"""World-space geometry: bounds, support surfaces, candidate hosts.

Pure functions over the scene graph, so these are the cheapest tests in the repo —
no MuJoCo, no meshes, no files.
"""

import math

import numpy as np
import pytest

from app.geometry import (
    candidate_hosts,
    footprint_xy,
    matrix_to_quat,
    quat_to_matrix,
    recentre,
    support_surfaces,
    world_aabb,
)
from app.schemas import CameraPose, SceneGraph
from tests.fixtures import scenes


def _surface(graph, object_id):
    surfaces = support_surfaces(graph, object_id)
    assert surfaces, f"{object_id} has no support surface"
    return surfaces[0]


# --- bounds -------------------------------------------------------------------


def test_world_bounds_match_the_authored_extents():
    graph = scenes.kitchen()
    low, high = world_aabb(graph.get("table"))
    assert tuple(high - low) == pytest.approx((1.2, 0.75, 0.75))
    assert low[2] == pytest.approx(0.0)


def test_bounds_grow_under_rotation():
    """A box rotated 45 degrees about z has a larger axis-aligned footprint than
    its own dimensions. Scaling the extent directly would miss that."""
    graph = scenes.kitchen()
    table = graph.get("table")
    table.orientation = (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))
    width, depth = footprint_xy(table)
    assert width > 1.2
    assert depth > 0.75


def test_bounds_scale_with_the_object():
    graph = scenes.kitchen()
    graph.get("mug").scale = 2.0
    assert footprint_xy(graph.get("mug")) == pytest.approx((0.18, 0.18))


# --- support surfaces ---------------------------------------------------------


def test_a_box_offers_only_its_top_face():
    """Five of six faces point sideways or down. Offering any of them as a shelf
    would be nonsense, and the +Z filter is what stops it."""
    surfaces = support_surfaces(scenes.kitchen(), "cabinet")
    assert len(surfaces) == 1
    assert surfaces[0].normal == pytest.approx((0.0, 0.0, 1.0))


def test_surface_height_and_area_are_the_top_of_the_object():
    surface = _surface(scenes.kitchen(), "table")
    assert surface.height_m == pytest.approx(0.75)
    assert surface.area_m2 == pytest.approx(1.2 * 0.75)


def test_occupied_area_is_subtracted():
    """The mug rests on the table, so the table's free area is short by the mug's
    footprint. Without this a surface reads as empty however crowded it is."""
    surface = _surface(scenes.kitchen(), "table")
    assert surface.area_m2 == pytest.approx(0.9)
    assert surface.free_area_m2 == pytest.approx(0.9 - 0.09 * 0.09, abs=1e-6)


def test_an_empty_surface_is_fully_free():
    surface = _surface(scenes.kitchen(), "cabinet")
    assert surface.free_area_m2 == pytest.approx(surface.area_m2)


def test_a_tilted_surface_is_rejected():
    """Reconstruction rarely returns a perfectly level tabletop, so some tolerance
    is needed — but a surface tipped past it is not somewhere things rest."""
    graph = scenes.kitchen()
    table = graph.get("table")
    # 30 degrees about x, well past the 10 degree tolerance.
    table.orientation = (math.cos(math.pi / 12), math.sin(math.pi / 12), 0.0, 0.0)
    assert support_surfaces(graph, "table") == []


def test_a_slightly_tilted_surface_is_still_accepted():
    graph = scenes.kitchen()
    table = graph.get("table")
    half = math.radians(5.0) / 2
    table.orientation = (math.cos(half), math.sin(half), 0.0, 0.0)
    surfaces = support_surfaces(graph, "table")
    assert len(surfaces) == 1
    assert surfaces[0].normal[2] > math.cos(math.radians(10.0))


def test_a_surface_too_small_to_matter_is_dropped():
    graph = scenes.kitchen()
    graph.get("mug").scale = 0.2  # an 18 mm top face
    assert support_surfaces(graph, "mug") == []


def test_an_unknown_object_has_no_surfaces():
    assert support_surfaces(scenes.kitchen(), "no-such-object") == []


# --- candidate hosts ----------------------------------------------------------


def test_candidate_hosts_are_ordered_by_height():
    hosts = candidate_hosts(scenes.kitchen(), "mug")
    assert [h.object_id for h in hosts] == ["cabinet", "table"]
    assert hosts[0].height_m > hosts[1].height_m


def test_an_object_is_never_its_own_host():
    assert all(h.object_id != "mug" for h in candidate_hosts(scenes.kitchen(), "mug"))


def test_hosts_too_small_are_excluded():
    """Nothing in the kitchen can hold the table."""
    assert candidate_hosts(scenes.kitchen(), "table") == []


def test_a_long_thin_surface_is_rejected_despite_ample_area():
    """Total area is not enough — the surface has to be wide enough in both
    directions, or a narrow shelf accepts a wide object it cannot hold."""
    graph = scenes.kitchen()
    table = graph.get("table")
    table.parts[0].dims_m = (4.0, 0.05, 0.75)  # 0.2 m2, plenty by area alone
    graph.get("mug").supported_by = None

    hosts = {h.object_id for h in candidate_hosts(graph, "mug")}
    assert "table" not in hosts


def test_dependents_are_excluded_transitively():
    """With a mug on a tray on a table, putting the table on the mug is a cycle two
    hops away — and a support graph with a cycle has no floor to settle against."""
    graph = scenes.kitchen()
    # cabinet -> table -> mug, so the mug is the table's grandchild.
    graph.get("mug").supported_by = "cabinet"
    graph.get("cabinet").supported_by = "table"

    hosts = {h.object_id for h in candidate_hosts(graph, "table")}
    assert "cabinet" not in hosts
    assert "mug" not in hosts


def test_the_current_host_stays_a_candidate():
    """ "Leave it where it is" is a legitimate variation, so the surface an object
    already rests on is not filtered out."""
    hosts = {h.object_id for h in candidate_hosts(scenes.kitchen(), "mug")}
    assert "table" in hosts


def test_hosts_are_geometric_only():
    """Nothing here knows an apple does not belong on a toilet cistern. That is the
    VLM's job, and the point of this list is to bound what it is asked about."""
    graph = scenes.kitchen()
    hosts = candidate_hosts(graph, "mug")
    assert {h.object_id for h in hosts} == {"table", "cabinet"}
    assert all(h.free_area_m2 > 0 for h in hosts)


# --- polygons -----------------------------------------------------------------


def test_the_polygon_is_a_closed_ring_in_world_space():
    surface = _surface(scenes.kitchen(), "table")
    assert len(surface.polygon_xy) == 4
    points = np.asarray(surface.polygon_xy)
    assert tuple(np.ptp(points, axis=0)) == pytest.approx((1.2, 0.75))
    # Ring order, not bit order: consecutive corners share an edge, so no diagonal.
    edges = np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1)
    assert max(edges) == pytest.approx(1.2)


# --- recentring ---------------------------------------------------------------


def test_recentre_puts_the_horizontal_centre_on_the_origin():
    """One fixed viewer camera has to frame every scene, but reconstruction happens in
    camera coordinates — so where a room lands depends on where the photographer
    stood."""
    graph = recentre(scenes.kitchen())
    corners = np.vstack([np.vstack(world_aabb(obj)) for obj in graph.objects])
    centre = (corners.min(axis=0) + corners.max(axis=0)) / 2.0
    assert centre[:2] == pytest.approx([0.0, 0.0], abs=1e-9)


def test_recentre_leaves_the_floor_alone():
    """Z is fixed by the plane fit and shared with MJCF's ground plane and the viewer's
    grid, so shifting it would put objects above a floor that does not move."""
    before = scenes.kitchen()
    after = recentre(before)
    for a, b in zip(before.objects, after.objects, strict=True):
        assert a.position_m[2] == pytest.approx(b.position_m[2])


def test_recentre_preserves_relative_placement():
    """A change of frame, not a change of scene."""
    before = scenes.kitchen()
    after = recentre(before)
    shifts = {
        a.object_id: np.asarray(b.position_m) - np.asarray(a.position_m)
        for a, b in zip(before.objects, after.objects, strict=True)
    }
    for shift in shifts.values():
        assert shift == pytest.approx(next(iter(shifts.values())))


def test_recentre_records_the_offset_it_applied():
    """`solve` measures against depth observations that are still in camera
    coordinates, so it has to be able to undo this."""
    before = scenes.kitchen()
    after = recentre(before)
    moved = np.asarray(after.objects[0].position_m) - np.asarray(before.objects[0].position_m)
    assert np.asarray(after.world_offset_m) == pytest.approx(moved)


def test_recentre_is_idempotent():
    """It accumulates onto world_offset_m rather than assigning, so a second call is a
    no-op and not a second shift."""
    once = recentre(scenes.kitchen())
    twice = recentre(once)
    assert twice.world_offset_m == pytest.approx(once.world_offset_m)
    for a, b in zip(once.objects, twice.objects, strict=True):
        assert a.position_m == pytest.approx(b.position_m)


def test_recentre_carries_the_camera_with_the_scene():
    """The recovered pose has to keep pointing at what it pointed at."""
    before = scenes.kitchen().model_copy(update={"camera": CameraPose(position_m=(0.0, 0.0, 1.2))})
    after = recentre(before)
    moved = np.asarray(after.camera.position_m) - np.asarray(before.camera.position_m)
    assert moved == pytest.approx(after.world_offset_m)


def test_recentre_tolerates_an_empty_scene():
    assert recentre(SceneGraph(objects=[])).objects == []


# --- quaternions --------------------------------------------------------------


@pytest.mark.parametrize(
    "angle",
    [0.0, 0.3, math.pi / 2, math.pi - 1e-9, math.pi, 2.5],
)
@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, -2, 3]])
def test_matrix_to_quat_inverts_quat_to_matrix(angle, axis):
    """Round-trip through the matrix, not through the quaternion.

    Comparing quaternions directly would fail on a valid answer: q and -q are the
    same rotation. The matrix is unique, so that is what the assertion is on.

    The half-turn cases are the point of the parametrisation. `w` goes to zero
    there, and the naive `q = (w, ...)` form divides by it — which is why this uses
    the branch with the largest divisor instead.
    """
    unit = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    expected = quat_to_matrix(
        (
            math.cos(angle / 2),
            *(math.sin(angle / 2) * unit),
        )
    )
    assert quat_to_matrix(matrix_to_quat(expected)) == pytest.approx(expected, abs=1e-12)


def test_matrix_to_quat_is_unit_length():
    """MuJoCo normalises whatever it is given, so a drifting quaternion is a silent
    rescale of nothing — but an inertia frame that is not a rotation is not one."""
    rotation = quat_to_matrix((0.5, 0.5, -0.5, 0.5))
    assert np.linalg.norm(matrix_to_quat(rotation)) == pytest.approx(1.0)


def test_matrix_to_quat_takes_the_positive_hemisphere():
    """A convention, not a correctness property: q and -q are the same rotation."""
    for angle in (0.3, 2.0, math.pi):
        rotation = quat_to_matrix((math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)))
        assert matrix_to_quat(rotation)[0] >= 0.0
