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
from pathlib import Path

from app.certify import certify
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.export import gltf, mjcf
from app.geometry import recentre
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

    # Recentred like a reconstructed scene, so the viewer's one fixed camera frames
    # the fixtures the same way it frames a photo. `reconcile` does this for real
    # scenes; the fixtures never go through it.
    graph = recentre(getattr(fixtures, factory_name)())
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


def seed_photo(image_path: Path, scene_id: str | None = None) -> str:
    """Run the real stages on a real photo and store the result.

    Stops before `solve` and `inertia`, which do not exist yet. What lands in the
    database is therefore the **uncoupled reconstruction**: objects sized and placed
    from depth, levelled to a fitted floor, with no physics feedback and no mass.

    It is not expected to certify, and that is the point — this is row one of the
    ablation, and the certificate measures how far off it is. The inertial axis
    reports NOT_APPLICABLE rather than inventing masses nobody computed.

    Stages go through the orchestrator's cache, so a second run on the same photo
    costs nothing. The first costs a few API calls and a couple of minutes.
    """
    from app.pipeline import depth, labeling, reconcile, rigid, segment
    from app.pipeline.orchestrator import _run_stage
    from app.schemas import StageName

    settings = get_settings()
    settings.ensure_dirs()

    scene_id = scene_id or f"photo-{image_path.stem}"
    ctx = PipelineContext.create(scene_id, image_path)
    report = lambda stage, state, secs: log.info("  %s: %s", stage, state)  # noqa: E731

    concepts = labeling.inventory(ctx)
    seg = _run_stage(
        ctx,
        StageName.SEGMENT,
        segment.SegmentResult,
        lambda: segment.run(ctx, concepts),
        (concepts,),
        report,
    )
    dep = _run_stage(ctx, StageName.DEPTH, depth.DepthResult, lambda: depth.run(ctx), (), report)
    lab = _run_stage(
        ctx,
        StageName.LABEL,
        labeling.LabelResult,
        lambda: labeling.run(ctx, seg),
        (seg,),
        report,
    )
    rec = _run_stage(
        ctx,
        StageName.RECONSTRUCT,
        rigid.ReconstructionResult,
        lambda: rigid.run(ctx, seg, lab),
        (seg, lab),
        report,
    )
    graph = _run_stage(
        ctx,
        StageName.RECONCILE,
        reconcile.SceneGraph,
        lambda: reconcile.run(ctx, rec, seg, dep, lab),
        (rec, seg, dep, lab),
        report,
    )

    out = ctx.workdir()
    spec = SceneSpec(
        scene_id=scene_id,
        image_path=settings.storage_relative(image_path),
        intrinsics=dep.intrinsics,
        graph=graph,
        certificate=certify.run(ctx, graph),
        exports=ExportResult(
            mjcf_path=settings.storage_relative(mjcf.write_mjcf(graph, out)),
            gltf_path=settings.storage_relative(gltf.write_gltf(graph, out)),
        ),
    )
    _store(scene_id, image_path.stem, spec)

    cert = spec.certificate
    print(f"  {scene_id}: {len(graph.objects)} objects")
    print(f"    axes    : { {k: v.value for k, v in cert.axes.items()} }")
    print(f"    failing : {sorted(cert.failing_object_ids())}")
    return scene_id


def _store(scene_id: str, name: str, spec: SceneSpec) -> None:
    init_db()
    db = SessionLocal()
    try:
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
    if len(sys.argv) > 1:
        print(f"running the pipeline on {sys.argv[1]}:")
        seed_photo(Path(sys.argv[1]))
        return 0
    print("seeding fixture scenes:")
    seed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
