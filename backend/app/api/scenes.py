from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.certify import certify, repair
from app.config import get_settings
from app.db import get_db
from app.export import gltf, mjcf
from app.models import Scene
from app.pipeline import solve
from app.pipeline.base import PipelineContext
from app.schemas import (
    AxisStatus,
    ExportResult,
    Provenance,
    RepairAction,
    SceneEditRequest,
    SceneSpec,
)

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class SceneSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    object_count: int
    certified: bool
    axes: dict[str, AxisStatus]


class SceneEnvelope(BaseModel):
    """A spec plus the row metadata a client needs but the spec does not carry.

    `updated_at` is here because the exported `scene.glb` is rewritten in place at
    the same URL every time a scene changes. Without a version to hang off the
    request the browser serves its cached copy, and a repair looks like it did
    nothing.
    """

    spec: SceneSpec
    updated_at: datetime


class PhysicsBundle(BaseModel):
    """Everything the browser needs to compile this scene in MuJoCo, in one request.

    The frontend runs the same MuJoCo — `@mujoco/mujoco` 3.11.0 against the
    backend's 3.11.0 — over this exact MJCF, which is what lets it certify an edit
    locally instead of approximating the server. `mjcf.build_xml` names mesh assets
    by bare filename precisely so a consumer without a filesystem can satisfy them;
    `meshes` maps each of those names to the URL to fetch it from.

    One request rather than letting the client discover 109 asset URLs by parsing
    the XML: the naming is `mjcf`'s business and a second implementation of it in
    TypeScript would be one more thing to keep in step.
    """

    mjcf: str
    meshes: dict[str, str] = Field(
        description="Mesh asset filename, as it appears in the MJCF, to its "
        "`/storage`-relative URL."
    )


class RepairResponse(BaseModel):
    scene: SceneEnvelope
    actions: list[RepairAction]
    converged: bool
    rounds_used: int


def _envelope(scene: Scene) -> SceneEnvelope:
    return SceneEnvelope(spec=SceneSpec.model_validate(scene.spec), updated_at=scene.updated_at)


@router.get("", response_model=list[SceneSummary])
def list_scenes(db: Session = Depends(get_db)) -> list[SceneSummary]:
    scenes = db.query(Scene).order_by(Scene.created_at.desc()).all()
    summaries = []
    for scene in scenes:
        spec = SceneSpec.model_validate(scene.spec)
        summaries.append(
            SceneSummary(
                id=scene.id,
                name=scene.name,
                created_at=scene.created_at,
                updated_at=scene.updated_at,
                object_count=len(spec.graph.objects),
                certified=spec.certificate.passed,
                axes=spec.certificate.axes,
            )
        )
    return summaries


@router.get("/{scene_id}", response_model=SceneEnvelope)
def get_scene(scene_id: str, db: Session = Depends(get_db)) -> SceneEnvelope:
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")
    return _envelope(scene)


@router.get("/{scene_id}/physics", response_model=PhysicsBundle)
def scene_physics(scene_id: str, db: Session = Depends(get_db)) -> PhysicsBundle:
    """The MJCF and its mesh assets, for running this scene in the browser.

    Built from the stored graph rather than read off disk, so it cannot disagree
    with the scene the client is looking at — the exported `scene.xml` is rewritten
    in place on every repair, and a client holding a stale URL would otherwise
    compile the pre-repair scene.
    """
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")

    settings = get_settings()
    graph = SceneSpec.model_validate(scene.spec).graph
    meshes = {
        name: settings.storage_relative(path) for name, path in mjcf.mesh_files(graph).items()
    }
    return PhysicsBundle(mjcf=mjcf.build_xml(graph), meshes=meshes)


@router.post("/{scene_id}/repair", response_model=RepairResponse)
def repair_scene(scene_id: str, db: Session = Depends(get_db)) -> RepairResponse:
    """Re-certify, apply the minimal correction, and re-export.

    Synchronous, unlike the upload path: measured at roughly 70 ms to certify plus
    20 ms to repair on a small scene, which is well inside a request. If scenes get
    big enough that this stops being true it should move behind the job queue that
    already exists rather than growing a timeout.

    Both exports are regenerated because they are pure functions of the graph, and
    a repaired scene whose MJCF still described the broken one would defeat the
    point of certifying at all.
    """
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")

    settings = get_settings()
    spec = SceneSpec.model_validate(scene.spec)
    ctx = PipelineContext.create(scene.job_id, settings.storage_dir / spec.image_path)

    certificate = certify.run(ctx, spec.graph)
    result = repair.run(ctx, spec.graph, certificate)

    out = ctx.workdir()
    spec.graph = result.graph
    spec.certificate = result.certificate
    # Appended, not replaced: a scene can be repaired more than once and the
    # history of what was corrected is the interesting part.
    spec.repairs_applied = [*spec.repairs_applied, *result.actions]
    spec.exports = ExportResult(
        mjcf_path=settings.storage_relative(mjcf.write_mjcf(result.graph, out)),
        gltf_path=settings.storage_relative(gltf.write_gltf(result.graph, out)),
    )

    scene.spec = spec.model_dump(mode="json")
    db.commit()
    db.refresh(scene)

    return RepairResponse(
        scene=_envelope(scene),
        actions=result.actions,
        converged=result.converged,
        rounds_used=result.rounds_used,
    )


@router.put("/{scene_id}", response_model=SceneEnvelope)
def edit_scene(
    scene_id: str,
    edit: SceneEditRequest,
    db: Session = Depends(get_db),
) -> SceneEnvelope:
    """Apply a user edit, re-solve from it, and re-certify.

    Everything is editable — pose, scale, support parent, mass. There is no wizard
    and no gating, because the system does not know enough to decide what the user
    is allowed to touch.

    **The edit seeds the solve; it does not constrain it.** `solve.run` starts from
    whatever pose the graph carries, so writing the new position and re-running is
    the whole mechanism: the object is refined from where the user put it rather
    than from where reconstruction did, and the objects resting on it follow. A
    typed dimension is different and already has its own hard-constraint path —
    `ScaleAnchor`, which enters `E_prior` with sigma to zero. A dragged position is
    an eyeball estimate, and holding one infinitely certain would throw away the
    depth measurement in its favour.

    **No repair.** Repair is a separate, deliberate action with its own endpoint. An
    edit that silently moved objects the user did not touch would be a surprising
    thing for a drag to do, and the certificate returned here is the honest answer
    to "what did my edit do" — including when the answer is that it made things
    worse.

    Values the user set are marked `Provenance.USER`, which nothing reads yet. It
    records what a person chose as distinct from what a model predicted, and that
    is worth keeping whether or not the solver ever holds it fixed.
    """
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")

    settings = get_settings()
    spec = SceneSpec.model_validate(scene.spec)

    known_objects = {o.object_id for o in spec.graph.objects}
    if unknown := {e.object_id for e in edit.objects} - known_objects:
        raise HTTPException(422, f"unknown object ids: {sorted(unknown)}")
    if unknown := {a.object_id for a in edit.anchors} - known_objects:
        raise HTTPException(422, f"unknown anchor object ids: {sorted(unknown)}")
    supports = {e.supported_by for e in edit.objects if e.supported_by is not None}
    if unknown := supports - known_objects:
        raise HTTPException(422, f"unknown support ids: {sorted(unknown)}")

    for change in edit.objects:
        obj = spec.graph.get(change.object_id)
        if obj is None:
            continue
        for field in ("scale", "position_m", "orientation", "supported_by"):
            value = getattr(change, field)
            if value is not None:
                setattr(obj, field, value)
                obj.provenance[field] = Provenance.USER
        if change.mass_kg is not None:
            # Mass lives on the root part, and setting it breaks the
            # density-times-volume identity the inertial axis checks — so the
            # density is moved with it rather than left describing the old mass.
            part = obj.root_part
            if part.inertial is not None and part.inertial.volume_m3 > 0.0:
                part.inertial = part.inertial.model_copy(
                    update={
                        "mass_kg": change.mass_kg,
                        "density_kg_m3": change.mass_kg / part.inertial.volume_m3,
                    }
                )
                obj.provenance["mass_kg"] = Provenance.USER

    spec.anchors = edit.anchors or spec.anchors

    ctx = PipelineContext.create(scene.job_id, settings.storage_dir / spec.image_path)
    if edit.resolve:
        # No depth map and no masks: a re-solve reads the per-object measurement
        # carried on the graph instead. See `DepthObservation`.
        solved = solve.run(ctx, spec.graph, None, None, spec.weights, spec.anchors)
        spec.graph = solved.graph
        spec.diagnostics = solved.diagnostics

    spec.certificate = certify.run(ctx, spec.graph)

    out = ctx.workdir()
    # Both exports, because they are pure functions of the graph and a viewer
    # showing the pre-edit geometry would make the edit look like it did nothing.
    spec.exports = ExportResult(
        mjcf_path=settings.storage_relative(mjcf.write_mjcf(spec.graph, out)),
        gltf_path=settings.storage_relative(gltf.write_gltf(spec.graph, out)),
    )

    scene.spec = spec.model_dump(mode="json")
    db.commit()
    db.refresh(scene)
    return _envelope(scene)
