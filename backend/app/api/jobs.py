import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import Job, JobState, Scene
from app.worker import run_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class JobStatus(BaseModel):
    id: str
    state: JobState
    current_stage: str | None
    stage_status: dict[str, Any]
    error: str | None
    scene_id: str | None
    created_at: datetime
    finished_at: datetime | None


def _to_status(job: Job, scene_id: str | None) -> JobStatus:
    return JobStatus(
        id=job.id,
        state=job.state,
        current_stage=job.current_stage,
        stage_status=job.stage_status,
        error=job.error,
        scene_id=scene_id,
        created_at=job.created_at,
        finished_at=job.finished_at,
    )


@router.post("", response_model=JobStatus, status_code=202)
async def create_job(
    background: BackgroundTasks,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> JobStatus:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"unsupported image type: {image.content_type}")

    payload = await image.read()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image exceeds 20 MB")

    settings = get_settings()
    job_id = str(uuid.uuid4())
    suffix = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}[image.content_type]
    path = settings.uploads_dir / f"{job_id}{suffix}"
    path.write_bytes(payload)

    job = Job(id=job_id, image_path=str(path))
    db.add(job)
    db.commit()

    background.add_task(run_job, job_id)
    return _to_status(job, scene_id=None)


@router.get("/{job_id}", response_model=JobStatus)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobStatus:
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    scene = db.query(Scene).filter(Scene.job_id == job_id).one_or_none()
    return _to_status(job, scene_id=scene.id if scene else None)
