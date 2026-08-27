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

    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(result))
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
    model = mujoco.MjModel.from_xml_string(xml)
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
