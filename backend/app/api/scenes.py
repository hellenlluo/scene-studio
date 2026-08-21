from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.certify import certify, repair
from app.config import get_settings
from app.db import get_db
from app.export import gltf, mjcf
from app.models import Scene
from app.pipeline.base import PipelineContext
from app.schemas import AxisStatus, ExportResult, RepairAction, SceneEditRequest, SceneSpec

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
    """Apply a user edit, re-solve around it, and re-certify.

    Everything is editable — pose, scale, support parent, mass. There is no wizard
    and no gating, because the system does not know enough to decide what the user
    is allowed to touch.

    Edited values are written with Provenance.USER, which is what makes them
    pinned: the next solve holds them fixed and moves everything else around
    them. A scale anchor is the same mechanism expressed as an E_prior term with
    sigma -> 0, and it propagates as far as the constraint graph is connected.
    """
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")

    spec = SceneSpec.model_validate(scene.spec)

    known_objects = {o.object_id for o in spec.graph.objects}
    if unknown := {e.object_id for e in edit.objects} - known_objects:
        raise HTTPException(422, f"unknown object ids: {sorted(unknown)}")
    if unknown := {a.object_id for a in edit.anchors} - known_objects:
        raise HTTPException(422, f"unknown anchor object ids: {sorted(unknown)}")

    # TODO: apply the edits with Provenance.USER, re-run app.pipeline.solve with
    # the pinned set held out, then app.certify.certify.
    #
    # The browser runs the same MuJoCo build against the same MJCF, so it can
    # certify the edit locally and immediately; this endpoint is the durable
    # write and the batch path, not a second opinion that could disagree.
    spec.anchors = edit.anchors or spec.anchors
    scene.spec = spec.model_dump(mode="json")
    db.commit()
    db.refresh(scene)
    return _envelope(scene)
