"""Run the pipeline on a photo and store the result so the viewer can load it.

`uv run python -m app.seed backend/storage/uploads/room2.png [scene_id]`

This is a development entry point. It does exactly what `POST /api/jobs` does —
`run_pipeline` end to end, then a Job and Scene row — but takes a photo already
on disk and gives it a readable scene id instead of a uuid, so a bookmarked URL
keeps working and re-running replaces rather than accumulates.

Stages go through the orchestrator's cache, keyed on the image bytes, so a
second run on the same photo costs nothing. The first costs a few API calls and
a couple of minutes.
"""

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings
from app.db import SessionLocal, init_db
from app.models import Job, JobState, Scene
from app.pipeline.orchestrator import run_pipeline
from app.schemas import SceneSpec, StageName

log = logging.getLogger(__name__)


def seed_photo(image_path: Path, scene_id: str | None = None) -> str:
    """Run every stage on a real photo and store the resulting scene."""
    settings = get_settings()
    settings.ensure_dirs()

    scene_id = scene_id or image_path.stem

    def report(stage: StageName, state: str, seconds: float | None) -> None:
        log.info("  %-12s %s%s", stage, state, f" ({seconds:.1f}s)" if seconds else "")

    spec = run_pipeline(scene_id, image_path, report)
    _store(scene_id, image_path.stem, spec)

    cert = spec.certificate
    print(f"\n  {scene_id}: {len(spec.graph.objects)} objects")
    print(f"    axes     : { {k: v.value for k, v in cert.axes.items()} }")
    print(f"    certified: {cert.passed}")
    print(f"    failing  : {sorted(cert.failing_object_ids())}")
    print(f"    exports  : {spec.exports.gltf_path}")
    return scene_id


def _store(scene_id: str, name: str, spec: SceneSpec) -> None:
    init_db()
    db = SessionLocal()
    try:
        # Replace rather than accumulate, so re-running is idempotent. Scene.job_id
        # is a foreign key, so the job has to exist and the scene goes first.
        for existing in (db.get(Scene, scene_id), db.get(Job, scene_id)):
            if existing is not None:
                db.delete(existing)
        db.flush()
        db.add(
            Job(
                id=scene_id,
                state=JobState.SUCCEEDED,
                image_path=spec.image_path,
                finished_at=datetime.now(UTC),
            )
        )
        db.add(
            Scene(
                id=scene_id,
                job_id=scene_id,
                name=name,
                image_path=spec.image_path,
                spec=spec.model_dump(mode="json"),
            )
        )
        db.commit()
    finally:
        db.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    image_path = Path(sys.argv[1]).resolve()
    if not image_path.exists():
        print(f"no such image: {image_path}")
        return 2
    scene_id = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"running the pipeline on {image_path.name}:")
    seed_photo(image_path, scene_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
