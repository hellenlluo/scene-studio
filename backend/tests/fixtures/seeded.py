"""Put hand-authored scenes into the database so the API can be tested end to end.

This is the stand-in for a reconstruction run. The scenes API tests need rows that
came through the same code the real path uses — certified, exported, inserted —
without spending a couple of minutes and a few paid API calls per test.

It lives under `tests/` rather than `app/` because it is test scaffolding: these
graphs are authored, not photographed, and nothing outside the suite should be able
to conjure a Scene row that no job produced.
"""

from datetime import UTC, datetime

from app.certify import certify
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.export import gltf, mjcf
from app.geometry import recentre
from app.models import Job, JobState, Scene
from app.pipeline.base import PipelineContext
from app.schemas import ExportResult, Intrinsics, SceneSpec
from tests.fixtures import scenes

# scene_id -> factory name. Two on purpose: one that certifies and one that does
# not but can be repaired. A viewer only ever tested against a passing scene tells
# you nothing about whether it renders failure correctly.
FIXTURES = {
    "test-sound": "kitchen",
    "test-sunk": "mug_sunk_into_table",
}

SOUND = "test-sound"
SUNK = "test-sunk"

# The fixtures were authored by hand, so there is no camera and no image. A
# plausible 640x480 pinhole keeps SceneSpec well-formed; nothing in the
# certification path reads it.
PLACEHOLDER_INTRINSICS = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def build_spec(scene_id: str, factory_name: str) -> SceneSpec:
    settings = get_settings()
    settings.ensure_dirs()

    # Recentred like a reconstructed scene, so the viewer's one fixed camera frames
    # these the same way it frames a photo. `reconcile` does this for real scenes.
    graph = recentre(getattr(scenes, factory_name)())
    ctx = PipelineContext.create(scene_id, settings.uploads_dir / f"{scene_id}.png")

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
    """Insert every fixture scene, replacing any row already under that id."""
    init_db()
    db = SessionLocal()
    try:
        written = []
        for scene_id, factory_name in FIXTURES.items():
            spec = build_spec(scene_id, factory_name)

            # Replace rather than accumulate, so re-seeding is idempotent.
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
        db.commit()
        return written
    finally:
        db.close()
