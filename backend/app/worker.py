"""In-process job runner.

Reconstruction takes minutes, so the API enqueues and the client polls. This is
deliberately the simplest thing that works for a single-user demo — the job
table and the status contract are the parts that matter, and swapping this for
a real queue later does not change the API surface.
"""

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.db import SessionLocal
from app.models import Job, JobState, Scene
from app.pipeline.orchestrator import run_pipeline
from app.schemas import StageName

log = logging.getLogger(__name__)


def run_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if job is None:
            log.error("job %s vanished before it ran", job_id)
            return

        job.state = JobState.RUNNING
        db.commit()

        def report(stage: StageName, state: str, seconds: float | None) -> None:
            # Reassign rather than mutate: SQLAlchemy does not track in-place
            # edits to a JSON column.
            job.stage_status = {
                **job.stage_status,
                str(stage): {"state": state, "seconds": seconds},
            }
            job.current_stage = str(stage)
            db.commit()

        try:
            spec = run_pipeline(job_id, Path(job.image_path), report)
        except Exception as exc:  # a stage failure belongs in the job record, not the logs alone
            log.exception("job %s failed in stage %s", job_id, job.current_stage)
            job.state = JobState.FAILED
            job.error = f"{job.current_stage or 'pipeline'}: {exc}"
            job.finished_at = datetime.now(UTC)
            db.commit()
            return

        db.add(
            Scene(
                id=str(uuid.uuid4()),
                job_id=job.id,
                image_path=job.image_path,
                spec=spec.model_dump(mode="json"),
            )
        )
        job.state = JobState.SUCCEEDED
        job.current_stage = None
        job.finished_at = datetime.now(UTC)
        db.commit()
    finally:
        db.close()
