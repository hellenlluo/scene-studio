from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Scene
from app.schemas import SceneLayout, SceneSpec

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


class SceneSummary(BaseModel):
    id: str
    name: str
    created_at: datetime
    object_count: int
    pass_rate: float


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
                object_count=len(spec.geometries),
                pass_rate=spec.validation.pass_rate,
            )
        )
    return summaries


@router.get("/{scene_id}", response_model=SceneSpec)
def get_scene(scene_id: str, db: Session = Depends(get_db)) -> SceneSpec:
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")
    return SceneSpec.model_validate(scene.spec)


@router.put("/{scene_id}/layout", response_model=SceneSpec)
def update_layout(
    scene_id: str,
    layout: SceneLayout,
    db: Session = Depends(get_db),
) -> SceneSpec:
    """Commit a user edit and revalidate against MuJoCo.

    The server is the sole authority on stability; the browser's kinematic
    preview is a latency hiding measure, not a second opinion.
    """
    scene = db.get(Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")

    spec = SceneSpec.model_validate(scene.spec)
    known = {o.object_id for o in spec.geometries}
    if unknown := {o.object_id for o in layout.objects} - known:
        raise HTTPException(422, f"unknown object ids: {sorted(unknown)}")

    # TODO: re-run app.pipeline.validate against the edited layout and write the
    # fresh ValidationResult back into the spec. Debounced on the client, so this
    # is called on every drag settle — keep it well under a second.
    spec.layout = layout
    scene.spec = spec.model_dump(mode="json")
    db.commit()
    return spec
