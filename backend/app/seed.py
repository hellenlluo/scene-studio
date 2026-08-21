"""Put fixture scenes into the database so the viewer has something to load.

`uv run python -m app.seed`

Six pipeline stages are still stubs, so no photo can produce a scene and no Scene
row can exist. Until the front half is written this is the only way to get the
frontend onto the real API path — it certifies a hand-authored graph, writes the
same MJCF and glTF the pipeline would, and inserts a row the API serves like any
other.

Two scenes on purpose: one that certifies and one that does not but can be
repaired. A viewer only ever tested against a passing scene tells you nothing
about whether it renders failure correctly.

**This imports from `tests/`, which is normally backwards.** It is right here
because the fixtures *are* the stand-in for stages 1-6, and a second copy of the
same scene living under `app/` would drift from the one every certification test
runs against. Keep the import inside this module, which is a development entry
point rather than part of the pipeline.
"""

import logging
import sys
from datetime import UTC, datetime

from app.certify import certify
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.export import gltf, mjcf
from app.models import Job, JobState, Scene
from app.pipeline.base import PipelineContext
from app.schemas import ExportResult, Intrinsics, SceneSpec

log = logging.getLogger(__name__)

# scene_id -> fixture factory name. The ids are stable so re-seeding replaces
# rather than accumulates, and a bookmarked URL keeps working.
FIXTURES = {
    "fixture-kitchen": "kitchen",
    "fixture-mug-sunk": "mug_sunk_into_table",
}

# The fixtures were authored by hand, not photographed, so there is no camera and
# no image. A plausible 640x480 pinhole keeps SceneSpec well-formed; nothing in the
# certification path reads it.
PLACEHOLDER_INTRINSICS = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def build_spec(scene_id: str, factory_name: str) -> SceneSpec:
    from tests.fixtures import scenes as fixtures

    settings = get_settings()
    settings.ensure_dirs()

    graph = getattr(fixtures, factory_name)()
    ctx = PipelineContext.create(scene_id, settings.uploads_dir / f"{scene_id}.placeholder")

    # The image only feeds the artifact cache key, and PipelineContext hashes its
    # bytes, so it has to exist even though nothing looks at the pixels.
    ctx.image_path.write_bytes(scene_id.encode())

    out = ctx.workdir()
    return SceneSpec(
        scene_id=scene_id,
        image_path=settings.storage_relative(ctx.image_path),
        intrinsics=PLACEHOLDER_INTRINSICS,
        graph=graph,
        certificate=certify.run(ctx, graph),
        exports=ExportResult(
            mjcf_path=settings.storage_relative(mjcf.write_mjcf(graph, out)),
            gltf_path=settings.storage_relative(gltf.write_gltf(graph, out)),
        ),
    )


def seed() -> list[str]:
    init_db()
    db = SessionLocal()
    try:
        written = []
        for scene_id, factory_name in FIXTURES.items():
            spec = build_spec(scene_id, factory_name)

            # Replace rather than accumulate, so re-running is idempotent.
            for existing in (db.get(Scene, scene_id), db.get(Job, scene_id)):
                if existing is not None:
                    db.delete(existing)
            db.flush()

            # Scene.job_id is a foreign key, so the job has to exist. SQLite would
            # not complain — it does not enforce foreign keys unless asked — but a
            # dangling reference would break GET /api/jobs/{id} and any later join.
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
                    name=factory_name.replace("_", " "),
                    image_path=spec.image_path,
                    spec=spec.model_dump(mode="json"),
                )
            )
            written.append(scene_id)
            print(
                f"  {scene_id:22s} {factory_name:22s} "
                f"certified={spec.certificate.passed}  {spec.exports.gltf_path}"
            )
        db.commit()
        return written
    finally:
        db.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("seeding fixture scenes:")
    seed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
