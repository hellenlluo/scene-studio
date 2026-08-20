from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scene
from app.schemas import AxisStatus, SceneEditRequest, SceneSpec

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class SceneSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    object_count: int
    joint_count: int
    certified: bool
    axes: dict[str, AxisStatus]


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
                object_count=len(spec.graph.objects),
                joint_count=spec.graph.joint_count,
                certified=spec.certificate.passed,
                axes=spec.certificate.axes,
            )
        )
    return summaries


@router.get("/{scene_id}", response_model=SceneSpec)
def get_scene(scene_id: str, db: Session = Depends(get_db)) -> SceneSpec:
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")
    return SceneSpec.model_validate(scene.spec)


@router.put("/{scene_id}", response_model=SceneSpec)
def edit_scene(
    scene_id: str,
    edit: SceneEditRequest,
    db: Session = Depends(get_db),
) -> SceneSpec:
    """Apply a user edit, re-solve around it, and re-certify.

    Everything is editable — pose, scale, support parent, joint axis and limits,
    mass, and the rigid-versus-articulated routing itself. There is no wizard and
    no gating, because the system does not know enough to decide what the user is
    allowed to touch.

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
    known_joints = {j.joint_id for o in spec.graph.objects for j in o.joints}
    if unknown := {e.object_id for e in edit.objects} - known_objects:
        raise HTTPException(422, f"unknown object ids: {sorted(unknown)}")
    if unknown := {a.object_id for a in edit.anchors} - known_objects:
        raise HTTPException(422, f"unknown anchor object ids: {sorted(unknown)}")
    if unknown := {e.joint_id for e in edit.joints} - known_joints:
        raise HTTPException(422, f"unknown joint ids: {sorted(unknown)}")

    # TODO: apply the edits with Provenance.USER, re-run app.pipeline.solve with
    # the pinned set held out, then app.certify.certify. Re-routing an object
    # sends it back through the other branch of stage 4 first, which is the one
    # edit that cannot be answered inside the solve.
    #
    # The browser runs the same MuJoCo build against the same MJCF, so it can
    # certify the edit locally and immediately; this endpoint is the durable
    # write and the batch path, not a second opinion that could disagree.
    spec.anchors = edit.anchors or spec.anchors
    scene.spec = spec.model_dump(mode="json")
    db.commit()
    return spec
