"""Collision proxies and mass properties.

Mass is `density x volume`, and both halves have a trap in them.

**Volume comes from the visual mesh, never the collision proxy.** Measured on a
dining table modelled as a slab and four legs:

| volume source | volume | mass at wood 700 |
|---|---|---|
| bounding box | 1.013 m3 | 709 kg |
| convex hull | 0.962 m3 | 674 kg |
| **visual mesh** | **0.064 m3** | **45 kg** |

A real dining table weighs about 40 kg. The hull is 95% of the bounding box —
taking volume from the proxy is a 15x error on any open shape, and open shapes are
most of the furniture in a room. The two meshes exist for different purposes and
this is the one that means mass. MuJoCo would otherwise derive inertia from the
collision geoms, which is why `app.export.mjcf` emits an explicit `<inertial>`
block whenever these properties are populated.

**Density is keyed on material, not category.** Once volume is a real mesh volume,
plain material density lands within about 12% without calibration. That only holds
against mesh volume — against a bounding box it overestimates by an order of
magnitude, which is what makes the rule above load-bearing rather than tidy.

**Known limitation: containers come out heavy.** Single-view reconstruction
returns an outer shell, so a mug is a solid ceramic cylinder — 1.5 kg against a
real 0.35 kg, roughly 4x. The defining feature of a container is the hole, and the
hole is unobservable from a photo of it. Accepted for now rather than papered over
with a fudge factor: the error is systematic, it only affects container categories,
and nothing in the current certification depends on mass being right. It starts
mattering when actuation does.

Where a part has no mesh at all — reconstruction failed and it fell back to an OBB
— its mass is the bounding-box number and is too heavy for the same reason, only
worse. That is recorded as `degradation_reason` and `ProxyTier.OBB`, not as
`watertight=False`: a box is a closed solid and its stated volume is exactly the
one its mass was computed from, so the inertial axis, which asks whether the mass
properties are *self-consistent and physically possible*, has nothing to complain
about. Whether the box is a good likeness of the object is a geometry question and
the proxy tier is where it is answered.

**Scale is applied here and nowhere else in the mesh.** The files on disk stay at
unit object scale, so moving `SceneObject.scale` never means rewriting geometry —
`app.export.mjcf` puts the factor on the `<mesh>` asset instead. What the solver
does have to keep in step is these numbers: volume goes as `s^3`, mass with it,
inertia as `s^5`, and the centre of mass as `s`. `recompute` is that update, and
it is pure so the solve loop can call it between residual evaluations.
"""

import logging
from pathlib import Path

import numpy as np
import trimesh

from app.geometry import matrix_to_quat, quat_to_matrix
from app.pipeline.base import PipelineContext
from app.schemas import (
    InertialProperties,
    Material,
    PartGeometry,
    ProxyTier,
    SceneGraph,
    SceneObject,
    Vec3,
)

__all__ = ["MATERIAL_DENSITY_KG_M3", "SceneGraph", "recompute", "run"]

log = logging.getLogger(__name__)

# Solid-material densities. Correct against a *mesh* volume; see the module
# docstring for why they are badly wrong against a bounding box.
#
# `metal` and `wood` span a wide real range (aluminium 2700 to steel 7800; balsa
# 150 to oak 900). The values here are the common case for household objects, and
# a finer vocabulary would be asking the VLM to judge something it cannot see.
MATERIAL_DENSITY_KG_M3: dict[Material, float] = {
    Material.WOOD: 650.0,
    Material.METAL: 7800.0,
    Material.PLASTIC: 1100.0,
    Material.GLASS: 2500.0,
    Material.CERAMIC: 2400.0,
    Material.FABRIC: 300.0,
    Material.STONE: 2700.0,
    Material.PAPER: 800.0,
    # Water, which is also MuJoCo's own default for a geom with no density.
    Material.OTHER: 1000.0,
}

COLLISION_SUFFIX = ".collision.obj"

# A decomposition piece thinner than this in any direction is a sliver CoACD
# emitted to chase a crease. MuJoCo will happily collide with it, at the cost of a
# contact pair per step and a hull whose normals are numerically unreliable —
# below a millimetre it is under the penetration tolerance anyway, so it cannot
# express a contact the certificate would believe.
MIN_PIECE_EXTENT_M = 1e-3


# --- volume and inertia -------------------------------------------------------


def _load_visual(part: PartGeometry) -> trimesh.Trimesh | None:
    """The part's visual mesh, or None when there is no usable file.

    A missing file degrades to the OBB path rather than raising, for the same
    reason `app.export.gltf` falls back to a box: a stage that cannot finish
    because one asset went astray costs the whole scene, and the proxy tier
    already records that the geometry is approximate.
    """
    if not part.visual_mesh_path:
        return None
    path = Path(part.visual_mesh_path)
    if not path.exists():
        log.warning(
            "visual mesh %s missing for part %s; falling back to its box", path, part.part_id
        )
        return None
    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        log.warning("visual mesh %s for part %s is empty", path, part.part_id)
        return None
    return loaded


def _closed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A best-effort watertight copy, for measuring volume and inertia on.

    Both quantities are surface integrals over a closed boundary, and trimesh
    computes them on an open mesh without complaining — the number that comes back
    is whatever the unclosed surface happens to integrate to.

    **The merge has to cross texture and normal seams, and by default it does not.**
    That is the whole reason this function exists, and it took a diagnosis to find
    because every symptom pointed elsewhere. A textured glTF stores a separate
    vertex per UV corner, so `trimesh.load` on a SAM 3D mesh hands back a surface
    that is topologically shattered. Measured on `room.jpg`: a floor lamp arrives as
    211 disconnected components with 4780 boundary edges and **zero** non-manifold
    edges, with consistent winding throughout — which is the signature of a closed
    surface that was never welded, not of a broken one. `merge_vertices()` with its
    defaults preserves those seams, because for rendering they carry information.
    For a volume they carry none, and leaving them in place makes every object in
    every real scene fail the inertial axis for a reason that has nothing to do with
    its geometry. Welded, 5 of that scene's 9 objects close outright.

    The same defect is upstream in `app.pipeline.rigid._load_glb`, which is why
    `source_watertight` is False on all nine and case 1 of `_measure` never fires
    today. Worth fixing there too — it would recover the pre-decimation volume,
    which is strictly the better measurement — but it is stage 4's to fix and the
    meshes on disk would have to be re-fetched to benefit.

    Hole filling first, then MeshFix for what it cannot reach. `trimesh`'s
    `fill_holes` only spans simple planar boundary loops, and the defects that
    survive welding are usually neither. Measured on `room.png`, the two meshes it
    left open were not "open shells" in any meaningful sense:

    - the rug had **one hexagonal hole** — six boundary edges out of 9598 faces
    - the plant had **no holes at all** — five edges each shared by four faces,
      a topology pinch rather than an opening, plus three detached fragments

    Both closed in under 0.1 s with `pymeshfix`, at +0.1% and -2.6% of volume; the
    plant's loss is the three fragments, which are reconstruction debris. Meshes
    that were already watertight pass through unchanged, so this costs nothing on
    the objects that never needed it.

    Not `manifold3d`, which `app.certify.inertial` suggests: it refuses input that
    is not already a volume, which is the condition being repaired. Voxel remeshing
    would also work and is the fallback if MeshFix ever fails, but it destroys
    surface detail and pulls in a further dependency.

    **MeshFix prints "could not fix everything" even when it succeeds**, so its
    output is not a success signal. `is_watertight` on the result is, and it is what
    the caller reads.
    """
    repaired = mesh.copy()
    repaired.merge_vertices(merge_tex=True, merge_norm=True)
    repaired.update_faces(repaired.nondegenerate_faces())
    repaired.remove_unreferenced_vertices()
    # An open shell can integrate to zero volume, and trimesh divides by it to get a
    # centre of mass. The nan is the answer — it is what `is_watertight` is about to
    # reject — so the warning is noise rather than news.
    with np.errstate(invalid="ignore", divide="ignore"):
        trimesh.repair.fill_holes(repaired)
        trimesh.repair.fix_winding(repaired)
        trimesh.repair.fix_inversion(repaired)
        if repaired.is_watertight:
            return repaired
        return _meshfix(repaired)


def _meshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """MeshFix's repair, or the input back if it does not help.

    Returning the input on failure rather than raising keeps the contract of
    `_closed`: it is best effort, and an object whose volume cannot be trusted is
    reported as such rather than costing the scene.
    """
    import pymeshfix

    try:
        vertices, faces = pymeshfix.clean_from_arrays(
            np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
        )
    except Exception as exc:  # a repair that fails is not worth losing the scene over
        log.warning("mesh repair failed (%s); leaving the surface open", exc)
        return mesh

    if not len(faces):
        return mesh
    fixed = trimesh.Trimesh(vertices=vertices, faces=faces)
    with np.errstate(invalid="ignore", divide="ignore"):
        trimesh.repair.fix_inversion(fixed)
        if not fixed.is_watertight or fixed.volume <= 0.0:
            return mesh
    return fixed


def _box_mesh(dims: Vec3) -> trimesh.Trimesh:
    return trimesh.creation.box(extents=np.asarray(dims, dtype=float))


def _measure(
    part: PartGeometry, mesh: trimesh.Trimesh | None
) -> tuple[trimesh.Trimesh, float, bool]:
    """The solid to take inertia from, its volume at unit scale, and whether to trust it.

    Three cases, in preference order:

    1. Stage 4 measured the volume on the mesh as reconstructed and found it
       closed. That is the best number available — better than anything measurable
       here, because the file on disk has been decimated since. The shape still
       comes from the stored mesh; only the scalar is taken from upstream. (Not
       reachable on today's output; see `_closed` for why.)
    2. No trusted upstream volume, but hole-filling closes the stored mesh. Measure
       it here.
    3. Neither. Report the open mesh's volume with `watertight=False`, which fails
       the inertial axis — deliberately, and repairably. The open volume is used
       rather than the bounding box because it is much the closer of the two and
       the flag already says not to trust it.
    """
    if mesh is None:
        return _box_mesh(part.dims_m), float(np.prod(np.asarray(part.dims_m, dtype=float))), True

    solid = _closed(mesh)
    watertight_volume = float(solid.volume) if solid.is_watertight else 0.0

    if part.source_watertight and part.source_volume_m3 and part.source_volume_m3 > 0.0:
        return solid, float(part.source_volume_m3), True
    if watertight_volume > 0.0:
        return solid, watertight_volume, True

    log.warning("part %s is not watertight and could not be closed", part.part_id)
    with np.errstate(invalid="ignore", divide="ignore"):
        return solid, float(abs(solid.volume)), False


def _inertial(
    solid: trimesh.Trimesh,
    volume_unit: float,
    density: float,
    scale: float,
    watertight: bool,
) -> InertialProperties:
    """Mass properties at the object's current scale.

    `volume_unit` overrides the solid's own volume, so mass stays consistent with
    the trusted measurement even when the shape is taken from a decimated copy.
    The tensor is computed at unit density — where trimesh's mass is numerically
    the volume — and then rescaled by the mass actually wanted, which keeps the
    shape and the magnitude from having to agree about density.

    Scale enters once, here. Volume goes as `s^3` and lengths as `s`, so inertia,
    being mass times length squared, goes as `s^5`.
    """
    mass_unit = density * volume_unit

    solid.density = 1.0
    # Setting the density invalidates trimesh's cached mass properties, so this is
    # where an unclosable shell integrates to zero volume and trimesh divides by it
    # to get a centre of mass. The nan is the answer, and the branch below is what
    # answers it — the warning is noise rather than news.
    with np.errstate(invalid="ignore", divide="ignore"):
        shape_volume = float(solid.volume)
        usable = shape_volume > 0.0 and bool(
            np.all(np.isfinite(solid.principal_inertia_components))
        )

    if usable:
        components = np.array(solid.principal_inertia_components, dtype=float, copy=True)
        components *= mass_unit / shape_volume
        # Columns are the principal axes in the part frame, which is what MuJoCo's
        # `iquat` names: the orientation of the inertial frame *within* the body.
        axes = np.array(solid.principal_inertia_vectors, dtype=float, copy=True).T
        if np.linalg.det(axes) < 0.0:
            axes[:, 0] *= -1.0  # eigenvectors have no handedness; a quaternion does
        centre = np.asarray(solid.center_mass, dtype=float)
    else:
        # A solid with no volume has no principal frame either. This is the same
        # geometry the OBB geom uses, so the two at least agree.
        extents = np.asarray(solid.extents, dtype=float)
        components = (
            mass_unit
            / 12.0
            * np.array(
                [
                    extents[1] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[2] ** 2,
                    extents[0] ** 2 + extents[1] ** 2,
                ]
            )
        )
        axes = np.eye(3)
        centre = np.zeros(3)

    # `density * volume` in that order and nowhere else, because `recompute` does
    # the same and the two have to agree bit for bit — the inertial axis checks
    # mass against density times volume, and a one-ULP disagreement between the two
    # paths to the same scene is a difference the certificate can see.
    volume = volume_unit * scale**3
    return InertialProperties(
        density_kg_m3=density,
        volume_m3=volume,
        mass_kg=density * volume,
        com_m=tuple(float(v) for v in centre * scale),
        inertia_diag=tuple(float(v) for v in components * scale**5),
        principal_axes=matrix_to_quat(axes),
        watertight=watertight,
    )


# --- collision proxies --------------------------------------------------------


def _decompose(mesh: trimesh.Trimesh, ctx: PipelineContext) -> list[trimesh.Trimesh]:
    """Convex pieces via CoACD, or a single hull if it cannot manage one.

    The decimation here is not the same trade as stage 4's. That cap is high to
    keep UV coordinates, which are what make an object look like the thing in the
    photo; these pieces are convex hulls and carry no material at all, while
    CoACD's runtime grows with the input. So collision gets its own, much lower,
    cap.

    A single convex hull is the fallback rather than a failure. It is still
    strictly better than an OBB for anything non-boxy, and `ProxyTier` records
    which one the caller got — the cost axis reports the tier alongside the step
    time precisely so the trade is visible rather than assumed.
    """
    import coacd

    # CoACD logs its whole configuration and a progress trace at info level on every
    # call, straight to stdout. On a nine-object scene that buries the pipeline's own
    # reporting, and none of it is actionable unless a decomposition actually fails.
    coacd.set_log_level("error")

    source = mesh
    if len(source.faces) > ctx.settings.max_collision_faces:
        source = source.simplify_quadric_decimation(face_count=ctx.settings.max_collision_faces)

    def attempt(preprocess: str):
        return coacd.run_coacd(
            coacd.Mesh(np.asarray(source.vertices), np.asarray(source.faces)),
            threshold=ctx.settings.coacd_threshold,
            max_convex_hull=ctx.settings.max_convex_pieces,
            preprocess_mode=preprocess,
        )

    # Preprocessing off by default. CoACD's `auto` mode voxelises the input at
    # `preprocess_resolution` (50) to force manifoldness, and on a thin object that
    # voxel is thicker than the object: measured on a reconstructed rug, 111:1
    # aspect ratio, the proxy came back 33.7 mm thick against a 15.1 mm mesh — it
    # bulged 9.3 mm below the floor, which the stability axis reported as a 10.8 mm
    # penetration, and it lifted the side table resting on it 11.7 mm above the
    # surface the viewer draws. Raising the resolution enough to fix that (>300) is
    # too slow to be worth it.
    #
    # Measured across all seven objects in `room`, `off` is better or equal on every
    # one: identical or fewer pieces, inflation 1.00x throughout against 2.23x for
    # the rug, and faster — one plant went 19.1 s to 9.3 s. It also coped with both
    # of the non-watertight meshes in that scene.
    #
    # `auto` is kept as the retry rather than deleted: preprocessing exists for input
    # CoACD cannot otherwise handle, and paying for it only when the fast path fails
    # is the right way round.
    try:
        raw = attempt("off")
    except Exception as exc:
        log.info("CoACD without preprocessing failed (%s); retrying with it", exc)
        try:
            raw = attempt("auto")
        except Exception as retry_exc:  # degrade the tier rather than the scene
            log.warning("CoACD failed (%s); falling back to a convex hull", retry_exc)
            return []

    pieces = []
    for vertices, faces in raw:
        piece = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces))
        if len(piece.faces) and float(np.min(piece.extents)) >= MIN_PIECE_EXTENT_M:
            pieces.append(piece.convex_hull)
    return pieces


def _flatten_base(
    pieces: list[trimesh.Trimesh], obj: SceneObject, band_m: float
) -> list[trimesh.Trimesh]:
    """Bring the lowest vertices of a decomposition onto one plane.

    A convex decomposition of a reconstructed object almost never has a flat
    bottom. Measured on a floor lamp whose base is level to 0.01 degrees, the
    collision hulls still varied 1.6 mm across its 268 mm width, so exactly *one*
    vertex reached the floor and the other 42 hung between 0.1 and 1.6 mm above it.

    A rigid body needs three non-collinear contacts to stand. With one it pivots,
    and it topples no matter how well it is placed — that lamp had a support gap of
    0.1 mm and fell 331 mm. Nothing downstream can fix it either: it is not a
    placement error, so repair has nothing to move, and it is not a mass error, so
    the centre of mass is irrelevant. Verified: softer contacts reduced the fall to
    130 mm without stopping it, a contact margin made it worse, and levelling the
    object left it on zero contacts. Flattening the base took it to 0.2 mm.

    Contacts are also *stiffer* now than MuJoCo's default, which makes this worse
    rather than better — a body that sinks micrometres never sinks the 1.6 mm it
    would need to engage the rest of its own base.

    The band is the penetration tolerance. Geometry finer than that is already
    declared not to matter — `max_penetration_m` exists because "mesh discretisation
    produces sub-millimetre contacts on surfaces flush by design" — so moving a
    vertex within it cannot change a verdict that was ever trustworthy. It does put
    a small flat spot on a genuinely round bottom, which is the honest cost and is
    bounded by the same tolerance.

    Along the object's own down direction, not the mesh's -Z: what has to be flat
    is the face that meets the floor, and the two differ by the object's
    orientation.
    """
    if not pieces:
        return pieces

    rotation = quat_to_matrix(obj.orientation)
    up_local = rotation.T @ np.array([0.0, 0.0, 1.0])

    heights = [np.asarray(piece.vertices, dtype=float) @ up_local for piece in pieces]
    lowest = min(float(height.min()) for height in heights)

    flattened = []
    for piece, height in zip(pieces, heights, strict=True):
        near = height < lowest + band_m
        if not near.any():
            flattened.append(piece)
            continue
        vertices = np.asarray(piece.vertices, dtype=float).copy()
        vertices[near] -= np.outer(height[near] - lowest, up_local)
        # Re-hulled because moving vertices can make a hull non-convex, and MuJoCo
        # collides a mesh geom as its hull regardless — better to hand it one that
        # already is than to let it silently take a different shape.
        flattened.append(trimesh.Trimesh(vertices=vertices, faces=piece.faces).convex_hull)
    return flattened


def _write_proxies(
    obj: SceneObject, part: PartGeometry, pieces: list[trimesh.Trimesh], out: Path
) -> list[str]:
    """One OBJ per convex piece, at unit object scale.

    Absolute paths, because `mjcf.build_xml` is fed to `MjModel.from_xml_string` by
    every physics axis and by the solver — there is no model file for a relative
    `file` attribute to resolve against, so it would resolve against the working
    directory instead and find nothing.
    """
    paths = []
    for index, piece in enumerate(pieces):
        path = out / f"{obj.object_id}_{part.part_id}_{index:02d}{COLLISION_SUFFIX}"
        path.write_bytes(piece.export(file_type="obj").encode())
        paths.append(str(path))
    return paths


# --- the stage ----------------------------------------------------------------


def run(ctx: PipelineContext, graph: SceneGraph) -> SceneGraph:
    """Assign collision proxies and the inertial priors.

    Convex decomposition via CoACD over each visual mesh, preceded by
    watertightness repair — trimesh will hand you a non-watertight mesh and a
    meaningless volume without complaining, and mass computed from that volume is
    meaningless too. Record InertialProperties.watertight either way, because the
    inertial certification axis treats an untrustworthy mass as a failure rather
    than a caveat.

    Compute `volume_m3` from the **visual** mesh even though the collision geometry
    is the proxy. See the module docstring.

    This runs before solve, not after certify: the solver's physics block steps
    MuJoCo, and MuJoCo cannot settle a body with no mass.

    `source_volume_m3` and `source_watertight` are written back on every part,
    including the OBB ones. Stage 4 sets them only where it had a mesh to measure,
    but after this stage they are the unit-scale reference `recompute` divides by —
    so leaving them unset on some parts would make the pure update work for most of
    a scene and silently skip the rest.
    """
    working = graph.model_copy(deep=True)
    out = ctx.workdir()

    for obj in working.objects:
        density = MATERIAL_DENSITY_KG_M3[obj.label.material]
        for part in obj.parts:
            mesh = _load_visual(part)
            solid, volume_unit, watertight = _measure(part, mesh)

            # The welded, hole-filled copy, not the file as loaded. CoACD runs its
            # own manifoldness pass first, and a surface split at every UV seam is
            # hundreds of disconnected components for it to reconcile — see
            # `_closed`. Feeding it the same solid the volume was measured on also
            # keeps the two from describing different geometry.
            pieces = _decompose(solid, ctx) if mesh is not None else []
            if mesh is not None and not pieces:
                pieces = [solid.convex_hull]
            pieces = _flatten_base(pieces, obj, ctx.settings.max_penetration_m)

            part.collision_mesh_paths = _write_proxies(obj, part, pieces, out)
            part.proxy_tier = (
                ProxyTier.DECOMPOSED
                if len(pieces) > 1
                else ProxyTier.CONVEX_HULL
                if pieces
                else ProxyTier.OBB
            )
            part.source_volume_m3 = volume_unit
            part.source_watertight = watertight

            part.inertial = inertial = _inertial(solid, volume_unit, density, obj.scale, watertight)

            log.info(
                "%s/%s: %s, %d piece(s), %.4f m3 -> %.2f kg%s",
                obj.object_id,
                part.part_id,
                part.proxy_tier,
                len(pieces),
                inertial.volume_m3,
                inertial.mass_kg,
                "" if watertight else " (volume not trustworthy)",
            )

    _prune(working, out)
    return working


def _prune(graph: SceneGraph, out: Path) -> None:
    """Drop proxy files belonging to objects this scene no longer has.

    After writing, never before — the same rule stage 4 arrived at the hard way.
    Clearing first means a run that dies partway takes the previous good proxies
    with it and leaves the stored scene pointing at files that are gone.

    **By object, not by filename, and that distinction is load-bearing.** Pruning
    every file the current run did not write looks tidier and corrupts the artifact
    cache: `app.pipeline.base` stores this stage's output keyed on its inputs, that
    output names collision files by absolute path, and the files are as much a part
    of the artifact as the JSON is. A later run that decomposes the same object into
    fewer pieces would delete the higher-indexed files — and then a cache *hit*
    returns a graph pointing at geometry that no longer exists. Observed exactly
    that: `obj_548_711_body_15.collision.obj` deleted by a 14-piece run, and MuJoCo
    refusing to open it on the next run that hit the cache.

    Keeping every file for an object that is still in the scene costs a few stale
    OBJs when a decomposition shrinks. Deleting one the cache still refers to costs
    a crash that looks like it came from the solver.
    """
    live = {obj.object_id for obj in graph.objects}
    for stale in out.glob(f"*{COLLISION_SUFFIX}"):
        # `{object_id}_{part_id}_{index:02d}.collision.obj`; object ids carry no
        # underscore-delimited suffix of their own, so the owner is the prefix
        # before the part and index this stage appended.
        owner = stale.name[: -len(COLLISION_SUFFIX)].rsplit("_", 2)[0]
        if owner not in live:
            stale.unlink()


def recompute(graph: SceneGraph) -> SceneGraph:
    """Refresh mass and inertia from density and the current scale.

    Volume goes as scale^3, so every mass in the scene is stale the moment the
    solver moves a scale. Pure function, called inside the solve loop; the
    density assigned in run() is what stays fixed.

    The scale the stored values were last computed at is not recorded anywhere, and
    does not need to be: `volume_m3` is `source_volume_m3 * s^3` by construction, so
    the cube root of their ratio recovers it. That keeps this a function of the
    graph alone — the solver can call it after any of the several places that move a
    scale without any of them having to remember to pass the old one.

    Volume and mass are recomputed from the reference directly rather than stepped
    by the ratio, so repeated calls cannot accumulate drift. Inertia and the centre
    of mass have no stored unit-scale form to go back to, so those do step.
    """
    working = graph.model_copy(deep=True)

    for obj in working.objects:
        for part in obj.parts:
            inertial = part.inertial
            reference = part.source_volume_m3
            if inertial is None or not reference or reference <= 0.0:
                continue
            if inertial.volume_m3 <= 0.0:
                continue

            previous = (inertial.volume_m3 / reference) ** (1.0 / 3.0)
            if previous <= 0.0:
                continue
            ratio = obj.scale / previous

            volume = reference * obj.scale**3
            part.inertial = inertial.model_copy(
                update={
                    "volume_m3": volume,
                    "mass_kg": inertial.density_kg_m3 * volume,
                    "com_m": tuple(float(v * ratio) for v in inertial.com_m),
                    "inertia_diag": tuple(float(v * ratio**5) for v in inertial.inertia_diag),
                }
            )

    return working
