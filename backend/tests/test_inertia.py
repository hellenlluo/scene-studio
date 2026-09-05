"""Stage 9: collision proxies and mass properties.

The load-bearing test is `test_convex_mesh_matches_mujoco`. Everything else here
checks a rule in isolation; that one checks the whole calculation against a second
implementation nobody in this repo wrote, on the one shape where the two are
obliged to agree exactly — mass, centre of mass, and the full inertia tensor
including the principal-axis quaternion. `iquat` in particular has no cheap
independent check: a transposed rotation is still a rotation, so a wrong
convention produces a plausible tensor that is simply the wrong one, and the only
symptom is an object tumbling oddly under actuation months later.

The comparison is only exact for a *convex* mesh, and that is not a limitation of
the test but the subject of `test_open_shape_volume_is_not_its_hull`: MuJoCo takes
a mesh geom's convex hull, so on an open shape it computes a mass that is wrong by
the ratio this module exists to avoid.
"""

import pathlib
import tempfile

import mujoco
import numpy as np
import pytest
import trimesh

from app.certify import inertial as inertial_axis
from app.export import mjcf
from app.export.gltf import node_transform
from app.geometry import quat_to_matrix
from app.pipeline import inertia
from app.pipeline.base import PipelineContext
from app.schemas import (
    AssetFrame,
    Material,
    ObjectLabel,
    PartGeometry,
    ProxyTier,
    SceneGraph,
    SceneObject,
)

WOOD = inertia.MATERIAL_DENSITY_KG_M3[Material.WOOD]


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A pipeline context whose workdir is a temp directory.

    `PipelineContext.create` reads the cached global settings, so the storage dir
    is redirected on the instance rather than by re-reading the environment.
    """
    image = tmp_path / "image.png"
    image.write_bytes(b"not really an image; nothing here opens it")
    context = PipelineContext.create("test-job", image)
    monkeypatch.setattr(context.settings, "storage_dir", tmp_path)
    return context


def _object(part: PartGeometry, scale: float = 1.0, material: Material = Material.WOOD):
    return SceneObject(
        object_id="obj",
        label=ObjectLabel(object_id="obj", category="thing", material=material),
        frame=AssetFrame(source="test"),
        parts=[part],
        scale=scale,
    )


def _mesh_part(tmp_path, mesh: trimesh.Trimesh, name: str = "mesh") -> PartGeometry:
    """A part backed by a real file, as stage 4 would leave it."""
    path = tmp_path / f"{name}.obj"
    path.write_text(mesh.export(file_type="obj"))
    return PartGeometry(
        part_id="body",
        name="body",
        visual_mesh_path=str(path),
        dims_m=tuple(float(v) for v in mesh.extents),
        source_volume_m3=float(mesh.volume) if mesh.is_watertight else None,
        source_watertight=bool(mesh.is_watertight),
    )


def _l_shape() -> trimesh.Trimesh:
    """A concave solid: a slab with an upright on one end.

    Boolean union rather than concatenation — concatenating leaves the interior
    faces in place, so the result is not watertight and its volume is whatever the
    self-intersecting surface integrates to.
    """
    slab = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    slab.apply_translation((0.2, 0.1, 0.05))
    upright = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    upright.apply_translation((0.05, 0.1, 0.15))
    return trimesh.boolean.union([slab, upright])


def _tensor(diag, quat) -> np.ndarray:
    """The full inertia tensor in the part frame, from the diagonal and its axes."""
    rotation = quat_to_matrix(quat)
    return rotation @ np.diag(np.asarray(diag, dtype=float)) @ rotation.T


# --- the OBB fallback ---------------------------------------------------------


def test_part_with_no_mesh_gets_its_box(ctx):
    """Reconstruction failed, so mass is the bounding-box number. It is too heavy,
    and that is recorded as the proxy tier rather than as untrustworthy volume."""
    part = PartGeometry(part_id="body", name="body", dims_m=(0.4, 0.2, 0.1))
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0]

    assert got.proxy_tier is ProxyTier.OBB
    assert got.collision_mesh_paths == []
    assert got.inertial.volume_m3 == pytest.approx(0.4 * 0.2 * 0.1)
    assert got.inertial.mass_kg == pytest.approx(0.4 * 0.2 * 0.1 * WOOD)
    assert got.inertial.watertight, "a box is a closed solid; its volume is exact"


def test_box_inertia_is_the_closed_form(ctx):
    part = PartGeometry(part_id="body", name="body", dims_m=(0.4, 0.2, 0.1))
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0].inertial

    x, y, z = 0.4, 0.2, 0.1
    k = got.mass_kg / 12.0
    assert got.inertia_diag == pytest.approx(
        (k * (y * y + z * z), k * (x * x + z * z), k * (x * x + y * y))
    )


def test_missing_mesh_file_degrades_rather_than_raising(ctx, tmp_path):
    """An asset that went astray costs fidelity, not the scene."""
    part = PartGeometry(
        part_id="body",
        name="body",
        visual_mesh_path=str(tmp_path / "never-written.obj"),
        dims_m=(0.4, 0.2, 0.1),
    )
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0]

    assert got.proxy_tier is ProxyTier.OBB
    assert got.inertial.mass_kg == pytest.approx(0.4 * 0.2 * 0.1 * WOOD)


# --- against MuJoCo -----------------------------------------------------------


def test_convex_mesh_matches_mujoco(ctx, tmp_path):
    """Mass, centre of mass and the full tensor, against MuJoCo's own compiler.

    A rotated, translated box: convex, so MuJoCo's hull of the mesh *is* the mesh
    and the two calculations have to land on the same numbers. Off-axis and
    off-centre on purpose — a shape at the origin aligned to the axes would pass
    with an identity `iquat` and a zero `com`, which is exactly the case that
    cannot distinguish a right convention from a wrong one.
    """
    mesh = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(0.7, [0, 1, 1]))
    mesh.apply_translation((0.3, -0.2, 0.05))
    part = _mesh_part(tmp_path, mesh)

    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    ours = result.objects[0].parts[0].inertial

    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="m" file="{part.visual_mesh_path}"/></asset>'
        '<worldbody><body name="b"><freejoint/>'
        f'<geom type="mesh" mesh="m" density="{WOOD}"/>'
        "</body></worldbody></mujoco>"
    )

    assert ours.mass_kg == pytest.approx(float(model.body_mass[1]), rel=1e-6)
    assert ours.com_m == pytest.approx(tuple(model.body_ipos[1]), abs=1e-8)
    assert _tensor(ours.inertia_diag, ours.principal_axes) == pytest.approx(
        _tensor(model.body_inertia[1], model.body_iquat[1]), abs=1e-8
    )


def test_open_shape_volume_is_not_its_hull(ctx, tmp_path):
    """The 15x claim in the module docstring, on a shape that has the property.

    An L is 88% of its own convex hull, which is the *mild* version — and it is
    still a mass MuJoCo would get wrong, because a mesh geom collides as its hull.
    Taking volume from the visual mesh is what avoids that.
    """
    mesh = _l_shape()
    part = _mesh_part(tmp_path, mesh)
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    ours = result.objects[0].parts[0].inertial

    assert ours.volume_m3 == pytest.approx(mesh.volume, rel=1e-6)
    assert ours.volume_m3 < mesh.convex_hull.volume
    assert ours.mass_kg == pytest.approx(mesh.volume * WOOD, rel=1e-6)


def test_mass_survives_the_round_trip_through_mjcf(ctx, tmp_path):
    """`_add_inertial` is what stops MuJoCo re-deriving mass from the proxy."""
    part = _mesh_part(tmp_path, _l_shape())
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    ours = result.objects[0].parts[0].inertial

    # `load_model`, not `from_xml_string`: the MJCF names meshes by bare filename
    # and the bytes are supplied through a virtual filesystem, so that the browser
    # can compile the identical string. Compiling it directly cannot find them.
    model = mjcf.load_model(result)
    assert float(model.body_mass[1]) == pytest.approx(ours.mass_kg, rel=1e-6)


# --- decomposition ------------------------------------------------------------


def test_concave_mesh_is_decomposed(ctx, tmp_path):
    part = _mesh_part(tmp_path, _l_shape())
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0]

    assert got.proxy_tier is ProxyTier.DECOMPOSED
    assert len(got.collision_mesh_paths) > 1
    for path in got.collision_mesh_paths:
        assert pathlib.Path(path).exists()


def test_decomposition_covers_the_shape_a_box_cannot(ctx, tmp_path):
    """The point of the tier: the pieces fill the solid, not its bounding box.

    An OBB around this L is 0.024 m3 against the solid's 0.012 — so the concavity
    the decomposition recovers is half the box. That gap is the shelf a book sits
    in, and at OBB tier there is no placement that satisfies both the support and
    the non-penetration constraint.
    """
    mesh = _l_shape()
    part = _mesh_part(tmp_path, mesh)
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    paths = result.objects[0].parts[0].collision_mesh_paths
    pieces = [trimesh.load(path, force="mesh") for path in paths]
    covered = sum(float(p.volume) for p in pieces)
    box = float(np.prod(mesh.extents))

    assert covered == pytest.approx(mesh.volume, rel=0.1)
    assert covered < box * 0.75


def test_convex_mesh_gets_one_hull(ctx, tmp_path):
    """Nothing to decompose, so the tier is honest about being a single hull."""
    part = _mesh_part(tmp_path, trimesh.creation.icosphere(subdivisions=2, radius=0.1))
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0]

    assert got.proxy_tier is ProxyTier.CONVEX_HULL
    assert len(got.collision_mesh_paths) == 1


def test_proxies_scale_with_the_object_in_mjcf(ctx, tmp_path):
    """Files stay at unit scale; `<mesh scale>` carries the object's.

    Without this the collision geometry would keep its authored size while the
    visual mesh and the box tier both grew, so an object would certify against a
    body a different size from the one on screen.
    """
    part = _mesh_part(tmp_path, _l_shape())
    result = inertia.run(ctx, SceneGraph(objects=[_object(part, scale=3.0)]))

    xml = mjcf.build_xml(result)
    assert 'scale="3 3 3"' in xml

    # Compiled extents against the files on disk, sorted twice over. MuJoCo
    # recentres each mesh on its centre of mass and rotates it onto its principal
    # axes — the geom carries the inverse, so the placement is unchanged, but the
    # stored axis order is not the file's. Sorting within each extent vector makes
    # the comparison invariant to that permutation; sorting across the pieces makes
    # it invariant to MuJoCo ordering its assets by name, which is not the order
    # `collision_mesh_paths` is in. What survives both is the size, which is the
    # claim.
    model = mjcf.load_model(result)
    mesh_geoms = model.geom_aabb[model.geom_type == mujoco.mjtGeom.mjGEOM_MESH]
    compiled = sorted(sorted(half * 2.0 for half in geom[3:]) for geom in mesh_geoms)
    stored = sorted(
        sorted(extent * 3.0 for extent in trimesh.load(path, force="mesh").extents)
        for path in result.objects[0].parts[0].collision_mesh_paths
    )
    # Loose because the compiled box is measured in the principal frame and the
    # stored one in the file's, and the two differ by however far the hull is from
    # axis-aligned; MuJoCo also keeps vertices in float32. Neither is close to the
    # factor of three a missing `scale` attribute would cost.
    assert np.ravel(compiled).tolist() == pytest.approx(np.ravel(stored).tolist(), rel=1e-3)


def test_a_shrinking_decomposition_keeps_its_older_pieces(ctx, tmp_path):
    """Files for an object still in the scene survive, even ones this run did not write.

    The artifact cache stores this stage's output keyed on its inputs, and that
    output names collision files by absolute path — so the files are part of the
    artifact. Deleting a higher-indexed piece because today's decomposition came
    out smaller leaves a cache *hit* pointing at geometry that is gone, which
    surfaces two stages later as MuJoCo failing to open a file. Observed on
    `obj_548_711_body_15.collision.obj`.
    """
    part = _mesh_part(tmp_path, _l_shape())
    graph = SceneGraph(objects=[_object(part)])
    inertia.run(ctx, graph)

    # A piece from an imagined earlier, larger decomposition of the same object.
    older = ctx.workdir() / f"obj_body_99{inertia.COLLISION_SUFFIX}"
    older.write_text("an earlier run's piece")
    inertia.run(ctx, graph)

    assert older.exists(), "the cache may still name this file"


def test_stale_proxies_are_pruned(ctx, tmp_path):
    part = _mesh_part(tmp_path, _l_shape())
    graph = SceneGraph(objects=[_object(part)])
    inertia.run(ctx, graph)

    orphan = ctx.workdir() / f"gone_body_00{inertia.COLLISION_SUFFIX}"
    orphan.write_text("stale")
    inertia.run(ctx, graph)

    assert not orphan.exists()


# --- what the certificate makes of it -----------------------------------------


def test_output_passes_the_inertial_axis(ctx, tmp_path):
    part = _mesh_part(tmp_path, _l_shape())
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    checks = inertial_axis.run(result)
    assert len(checks) == 1
    assert checks[0].passed


def test_an_unclosable_mesh_fails_rather_than_being_caveated(ctx, tmp_path):
    """A single open triangle fan: no volume anyone should believe.

    The mass that comes back is not nothing — trimesh integrates the open surface
    without complaining — which is precisely why this has to fail the axis rather
    than pass with a footnote.
    """
    open_mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    part = _mesh_part(tmp_path, open_mesh)
    assert not part.source_watertight

    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    assert not result.objects[0].parts[0].inertial.watertight
    assert not inertial_axis.run(result)[0].passed


def test_every_material_has_a_density():
    """A missing entry would be a KeyError inside a background job, on whichever
    photo first contains a stone worktop."""
    assert set(inertia.MATERIAL_DENSITY_KG_M3) == set(Material)


# --- closing what hole-filling cannot ------------------------------------------


def test_a_hole_too_complex_for_fill_holes_is_still_closed(ctx, tmp_path):
    """`trimesh.repair.fill_holes` spans simple planar loops and little else.

    The real case: a reconstructed rug with exactly one hexagonal hole — six
    boundary edges out of 9598 faces — which `fill_holes` left open, so its volume
    and therefore its mass were untrustworthy and the inertial axis failed it.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.2)
    # Punch out a patch big enough that the boundary is neither planar nor small.
    keep = np.ones(len(sphere.faces), dtype=bool)
    keep[np.argsort(sphere.triangles_center[:, 2])[-14:]] = False
    sphere.update_faces(keep)
    assert not sphere.is_watertight, "the fixture has to actually be open"

    part = _mesh_part(tmp_path, sphere, "holed")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    got = result.objects[0].parts[0].inertial

    assert got.watertight, "closed, so the volume is trustworthy"
    assert got.volume_m3 == pytest.approx(
        float(trimesh.creation.icosphere(subdivisions=3, radius=0.2).volume), rel=0.05
    )


def test_an_already_closed_mesh_is_left_alone(ctx, tmp_path):
    """Repair runs only when hole-filling did not already succeed, so an object
    that never needed it cannot be altered by it."""
    box = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    part = _mesh_part(tmp_path, box, "box")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    assert result.objects[0].parts[0].inertial.volume_m3 == pytest.approx(
        float(box.volume), rel=1e-9
    )


def test_an_unrepairable_surface_is_still_reported_untrusted(ctx, tmp_path):
    """Best effort, not a guarantee. Two triangles are not a solid, and saying so
    is the point — the inertial axis treats an untrustworthy mass as a failure."""
    flat = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.2, 0, 0], [0.2, 0.2, 0], [0, 0.2, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    part = _mesh_part(tmp_path, flat, "flat")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    assert not result.objects[0].parts[0].inertial.watertight
    assert not inertial_axis.run(result)[0].passed


# --- a base that can actually stand -------------------------------------------


def _wobbly_disc(tilt_mm: float = 1.6) -> trimesh.Trimesh:
    """A disc whose underside is a fraction of a millimetre out of flat.

    The reconstructed-lamp case in miniature: level to a hundredth of a degree,
    and still resting on one point because its lowest vertices do not share a
    plane.
    """
    disc = trimesh.creation.cylinder(radius=0.13, height=0.05, sections=48)
    bottom = disc.vertices[:, 2] < disc.vertices[:, 2].min() + 1e-9
    # Tip the underside only, leaving the rest of the shape alone.
    disc.vertices[bottom, 2] -= (disc.vertices[bottom, 0] / 0.13) * (tilt_mm / 2000.0)
    return disc


def _on_floor(mesh: trimesh.Trimesh) -> int:
    """How many vertices sit within 10 microns of the lowest one."""
    z = mesh.vertices[:, 2]
    return int((z < z.min() + 1e-5).sum())


def test_a_wobbly_base_is_brought_onto_one_plane(ctx, tmp_path):
    """One contact point cannot hold a body up; three non-collinear ones can.

    Measured on the reconstruction this comes from: a floor lamp with a support gap
    of 0.1 mm and a base level to 0.01 degrees still fell 331 mm, because exactly
    one vertex of its decomposition reached the floor.
    """
    part = _mesh_part(tmp_path, _wobbly_disc(), "disc")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    pieces = [
        trimesh.load(path, force="mesh") for path in result.objects[0].parts[0].collision_mesh_paths
    ]
    assert pieces
    assert _on_floor(trimesh.util.concatenate(pieces)) > 4, "the base rests on a plane, not a point"


def test_flattening_stays_within_the_penetration_tolerance(ctx, tmp_path):
    """The band is `max_penetration_m`, so nothing moves further than the amount
    the certificate already declares immaterial."""
    part = _mesh_part(tmp_path, _wobbly_disc(), "disc")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    original = trimesh.load(part.visual_mesh_path, force="mesh")
    pieces = trimesh.util.concatenate(
        [trimesh.load(p, force="mesh") for p in result.objects[0].parts[0].collision_mesh_paths]
    )
    dropped = original.vertices[:, 2].min() - pieces.vertices[:, 2].min()
    assert abs(dropped) <= ctx.settings.max_penetration_m + 1e-6


def test_a_flat_base_is_left_alone(ctx, tmp_path):
    """A box already rests on a plane; flattening must not invent a change."""
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.2))
    part = _mesh_part(tmp_path, box, "box")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))

    pieces = trimesh.util.concatenate(
        [trimesh.load(p, force="mesh") for p in result.objects[0].parts[0].collision_mesh_paths]
    )
    assert pieces.bounds[0][2] == pytest.approx(box.bounds[0][2], abs=1e-6)


def test_it_flattens_along_the_object_s_own_down_direction(ctx, tmp_path):
    """What has to be flat is the face that meets the floor, and which face that is
    depends on the object's orientation — not on the mesh's own -Z.

    So: wobble the disc's *top* face and turn the object upside down. The wobbled
    face is now the one on the ground. Code that flattened the mesh's lowest
    vertices would smooth the face pointing at the ceiling and leave this object
    balanced on a point exactly as before.
    """
    disc = trimesh.creation.cylinder(radius=0.13, height=0.05, sections=48)
    top = disc.vertices[:, 2] > disc.vertices[:, 2].max() - 1e-9
    disc.vertices[top, 2] += (disc.vertices[top, 0] / 0.13) * 0.0008
    part = _mesh_part(tmp_path, disc, "disc")

    # 180 degrees about X: the wobbled +Z face now points down in world.
    flipped = _object(part).model_copy(update={"orientation": (0.0, 1.0, 0.0, 0.0)})
    result = inertia.run(ctx, SceneGraph(objects=[flipped]))

    posed = result.objects[0]
    transform = node_transform(posed, posed.parts[0])
    pieces = []
    for path in posed.parts[0].collision_mesh_paths:
        piece = trimesh.load(path, force="mesh")
        piece.apply_transform(transform)
        pieces.append(piece)
    world = trimesh.util.concatenate(pieces)
    assert _on_floor(world) > 4, "the face against the floor is the one that got flattened"


def test_flattening_does_not_touch_mass(ctx, tmp_path):
    """Volume comes from the visual mesh, so collision geometry cannot move it."""
    part = _mesh_part(tmp_path, _wobbly_disc(), "disc")
    result = inertia.run(ctx, SceneGraph(objects=[_object(part)]))
    original = trimesh.load(part.visual_mesh_path, force="mesh")

    assert result.objects[0].parts[0].inertial.volume_m3 == pytest.approx(
        float(original.volume), rel=1e-6
    )


# --- recompute ----------------------------------------------------------------


def _scaled_graph(ctx, tmp_path, scale):
    part = _mesh_part(tmp_path, _l_shape())
    return inertia.run(ctx, SceneGraph(objects=[_object(part, scale=scale)]))


def test_recompute_scales_by_the_right_powers(ctx, tmp_path):
    """Volume as s^3, mass with it, the centre of mass as s, inertia as s^5.

    Inertia is mass times length squared, so s^3 * s^2. Getting it wrong is
    invisible at scale 1, which is where every fixture in this repo sits.
    """
    before = _scaled_graph(ctx, tmp_path, 1.0)
    scaled = before.model_copy(deep=True)
    scaled.objects[0].scale = 2.0
    after = inertia.recompute(scaled)

    first = before.objects[0].parts[0].inertial
    second = after.objects[0].parts[0].inertial

    assert second.volume_m3 == pytest.approx(first.volume_m3 * 8.0)
    assert second.mass_kg == pytest.approx(first.mass_kg * 8.0)
    assert second.com_m == pytest.approx(tuple(v * 2.0 for v in first.com_m))
    assert second.inertia_diag == pytest.approx(tuple(v * 32.0 for v in first.inertia_diag))
    assert second.density_kg_m3 == first.density_kg_m3, "density is what stays fixed"
    assert second.principal_axes == first.principal_axes, "isotropic scale cannot rotate it"


def test_recompute_agrees_with_running_the_stage_at_that_scale(ctx, tmp_path):
    """The two paths to the same scene have to produce the same numbers, or the
    solve loop drifts away from what a re-run of the pipeline would say."""
    direct = _scaled_graph(ctx, tmp_path, 2.5).objects[0].parts[0].inertial

    stepped = _scaled_graph(ctx, tmp_path, 1.0).model_copy(deep=True)
    stepped.objects[0].scale = 2.5
    stepped = inertia.recompute(stepped).objects[0].parts[0].inertial

    assert stepped.mass_kg == pytest.approx(direct.mass_kg, rel=1e-9)
    assert stepped.volume_m3 == pytest.approx(direct.volume_m3, rel=1e-9)
    assert stepped.com_m == pytest.approx(direct.com_m, rel=1e-9)
    assert stepped.inertia_diag == pytest.approx(direct.inertia_diag, rel=1e-9)


def test_recompute_does_not_drift_when_the_scale_has_not_moved(ctx, tmp_path):
    """It runs on every solver iteration, so a call that walks away from its own
    fixed point would compound over a solve rather than showing up once.

    Volume and mass are exact fixed points, because both are recomputed from
    `source_volume_m3` rather than stepped. Inertia and the centre of mass have no
    stored unit-scale form to go back to, so they step by a ratio that is 1 only to
    within rounding — a hundred calls at an unchanged scale is what a solve
    actually does, and the assertion below is what that costs.
    """
    graph = _scaled_graph(ctx, tmp_path, 1.7)
    start = graph.objects[0].parts[0].inertial

    for _ in range(100):
        graph = inertia.recompute(graph)
    end = graph.objects[0].parts[0].inertial

    assert end.volume_m3 == start.volume_m3
    assert end.mass_kg == start.mass_kg
    assert end.com_m == pytest.approx(start.com_m, rel=1e-12)
    assert end.inertia_diag == pytest.approx(start.inertia_diag, rel=1e-12)


def test_recompute_leaves_a_part_stage_9_never_saw_alone(ctx, tmp_path):
    """No inertial properties means the stage has not run for that part. Inventing
    some here would let a scene certify on numbers nothing measured."""
    part = PartGeometry(part_id="body", name="body", dims_m=(0.4, 0.2, 0.1))
    graph = SceneGraph(objects=[_object(part, scale=2.0)])

    assert inertia.recompute(graph).objects[0].parts[0].inertial is None


def test_recompute_is_pure(ctx, tmp_path):
    graph = _scaled_graph(ctx, tmp_path, 1.0)
    graph.objects[0].scale = 4.0
    inertia.recompute(graph)

    assert graph.objects[0].parts[0].inertial.volume_m3 == pytest.approx(
        graph.objects[0].parts[0].source_volume_m3
    ), "the input still describes scale 1"


def test_recompute_survives_a_temp_dir(tmp_path):
    """Pure means no filesystem: the meshes it was built from can be gone."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        image = scratch / "image.png"
        image.write_bytes(b"x")
        context = PipelineContext.create("purity-job", image)
        context.settings = context.settings.model_copy(update={"storage_dir": scratch})
        part = _mesh_part(scratch, _l_shape())
        graph = inertia.run(context, SceneGraph(objects=[_object(part)]))

    graph.objects[0].scale = 2.0
    assert inertia.recompute(graph).objects[0].parts[0].inertial.mass_kg > 0.0
