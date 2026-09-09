from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.certify import certify, repair
from app.config import get_settings
from app.db import get_db
from app.export import gltf, mjcf
from app.geometry import dependents
from app.models import Scene
from app.pipeline import reconcile, solve
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


class SceneEditResponse(BaseModel):
    """The committed scene, plus what repairing it did.

    The actions are not decoration. A commit now repairs as well as re-solves, and
    a repair that moved an object without saying so is the silent-change problem
    the endpoint used to avoid by not repairing at all. Reported here so the client
    can show what a drag actually cost — including the proposals that were tried
    and reverted, which `RepairAction.improved` distinguishes.
    """

    scene: SceneEnvelope
    actions: list[RepairAction] = []


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


@router.put("/{scene_id}", response_model=SceneEditResponse)
def edit_scene(
    scene_id: str,
    edit: SceneEditRequest,
    db: Session = Depends(get_db),
) -> SceneEditResponse:
    """Apply a user edit, re-derive what it rests on, re-solve, certify and repair.

    Everything is editable — pose, scale, support parent, mass. There is no wizard
    and no gating, because the system does not know enough to decide what the user
    is allowed to touch.

    **A drag revises the scene graph, not just the pose.** Dropping a book on a
    table makes the table what it rests on, and until that edge is rewritten every
    later stage reads the old one — the support term closes the gap to the floor and
    puts the book underneath the table it was dropped on. So the parent is
    re-derived from geometry before the solve, scoped to the objects the user moved.

    **The edit is a measurement, not just a seed.** `solve.run` starts from whatever
    pose the graph carries, but starting there is not enough on its own: the stored
    depth centre still pointed at the pre-edit position and simply pulled the object
    back, keeping 0% of a one-metre drag. Properties the user has set are now what
    the depth term aims at for that object — see `solve._edit_priors`. Softly, at
    the depth weight, so an impossible drop is still corrected by support and
    penetration. `ScaleAnchor` is still the hard constraint, for a dimension the
    user has actually measured.

    **Repair runs here, bounded to the edit.** It used to be refused outright,
    because "an edit that silently moved objects the user did not touch would be a
    surprising thing for a drag to do" — true, and it left a commit able to produce
    a visibly broken scene whose fix sat behind a button. Scoping repair to the
    edited objects and their dependents keeps that guarantee and still finishes the
    job; `SceneEditResponse.actions` reports every correction, so nothing is silent.
    The whole-scene `/repair` endpoint remains for a scene loaded without an edit.

    Values the user set are marked `Provenance.USER`. `solve._edit_priors` reads it,
    so it is now load-bearing rather than merely recorded — see the note there about
    solve no longer stamping over it.
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
    actions: list[RepairAction] = []
    free: set[str] = set()
    if edit.resolve:
        touched = {change.object_id for change in edit.objects}
        touched |= {anchor.object_id for anchor in edit.anchors}

        # **Re-derive support before anything reads it.** A drag carries a new
        # position and no new parent, so without this the graph still records where
        # the object used to rest and stage 6 dutifully closes the gap to it:
        # measured on `room2`, a book dropped on the side table at z=0.584 came back
        # from solve at z=0.015, on the floor underneath it. Same arbitration stage 5
        # uses, scoped to the objects the user moved — see `resolve_supports`.
        #
        # Except where the user named a parent themselves. An explicit choice is not
        # something to re-derive; that is the one case where the recorded edge is
        # about the present rather than the past.
        reconsider = touched - {
            change.object_id for change in edit.objects if change.supported_by is not None
        }
        if reconsider:
            spec.graph = reconcile.resolve_supports(spec.graph, settings, reconsider)

        # Only what the user touched, plus whatever rests on it. A full re-solve
        # moves objects nobody edited: measured on `room2.png` it took the scale axis
        # from 6 failures to 9, and with a third of the scene sitting within 4 mm of
        # the gap tolerance that reads as objects flickering between certified and
        # not for no visible reason. Repair below is scoped to the same set for the
        # same reason.
        #
        # Computed *after* reparenting, because reparenting is what decides who the
        # dependents are: a book moved from the basket to the side table takes its
        # own stack with it, and the basket's remaining contents are no longer in
        # scope.
        free = set(touched)
        for object_id in touched:
            free |= dependents(spec.graph.objects, object_id)

        solved = solve.run(
            ctx, spec.graph, None, None, spec.weights, spec.anchors, free_ids=free or None
        )
        spec.graph = solved.graph
        spec.diagnostics = solved.diagnostics

    spec.certificate = certify.run(ctx, spec.graph)

    if edit.resolve and free:
        # Repair, scoped to the same objects the solve was free to move.
        #
        # This endpoint used to decline to repair at all, on the grounds that
        # "repair changes objects the user did not touch, which is a surprising
        # thing for a drag to do". That reasoning was right about the danger and
        # wrong about the remedy: it left a commit able to produce a scene that
        # visibly fails, with the fix behind a separate button the user had to know
        # to press. Bounding repair to the edit answers the objection directly —
        # nothing outside the set is ever proposed — and lets a commit finish the
        # job it started.
        repaired = repair.run(ctx, spec.graph, spec.certificate, only_ids=free)
        spec.graph = repaired.graph
        spec.certificate = repaired.certificate
        actions = repaired.actions
        # Appended rather than replaced, as `/repair` does: the history of what was
        # corrected is the interesting part.
        spec.repairs_applied = [*spec.repairs_applied, *actions]

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
    return SceneEditResponse(scene=_envelope(scene), actions=actions)
