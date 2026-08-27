"""Stage 4: per-object mesh reconstruction, via SAM 3D Objects on fal.

One call for the whole scene: `mask_urls` takes the list stage 1 produced and
`individual_glbs` comes back with a mesh per object, metadata in the same order.
That is a single billed generation rather than one per object.

Two things the response needs unpicking for, and both fail silently if got wrong:

**The rotation is XYZW; everything here is WXYZ.** fal returns the quaternion in
scalar-last order, MuJoCo and this schema use scalar-first. Passing one through as
the other yields a perfectly valid rotation that is simply the wrong one, and
nothing downstream can tell — objects just come out facing oddly.

**The scale is anisotropic and not metric.** Metadata carries per-axis factors,
while `SceneObject.scale` is one number by design, because anisotropic scale breaks
inertia. So the anisotropy is baked into the mesh, where it belongs as *shape*, and
only the isotropic residual is recorded as `normalization_scale`. That residual is
the model's own guess at overall size, in its own units, and is never treated as
metres: scale inherited from a generative model and never reconciled is precisely
the failure this project exists to fix.

`pointmap_url` is deliberately unused. Feeding stage 2's depth in would condition
the mesh on the same measurement the solver scores against, which is worth having —
but the endpoint documents only "NPY or NPZ" with no shape or dtype, and sending a
guess risks breaking the call for an optimisation rather than a requirement.
"""

import concurrent.futures
import io
import logging
import time

import httpx
import numpy as np
import trimesh

from app.pipeline.base import PipelineContext
from app.schemas import (
    AssetFrame,
    LabelResult,
    ObjectAssets,
    PartGeometry,
    ReconstructionResult,
    SegmentResult,
)

__all__ = ["ReconstructionResult", "run"]

log = logging.getLogger(__name__)

# fal_client polls status every 0.1s by default. A mesh job runs for minutes, so
# that is thousands of requests for one result — enough to look like abuse and to
# risk rate limiting. Nothing here needs sub-second latency on a job this long.
POLL_INTERVAL_S = 2.0

# Downloads are the other half of the wall clock: sequentially, 72 MB took 199s.
# But concurrency has a cost that only showed up on a second run — at six workers,
# five of nine downloads died with "peer closed connection without sending complete
# message body" at 40-90% transferred. Several 8-14 MB streams at once is enough to
# make the CDN drop connections, so this trades some overlap for completing.
DOWNLOAD_WORKERS = 3
DOWNLOAD_TIMEOUT_S = 300.0

# A dropped connection is transient and losing an object to one is not acceptable —
# the object cannot be recovered later, it is simply missing from the scene.
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_S = 2.0

SOURCE = "sam3d"
# Something has to stand in the scene when a mesh fails, so reconcile can fall it
# back to an OBB rather than dropping the object entirely.
FALLBACK_DIMS = (0.1, 0.1, 0.1)


def _client(ctx: PipelineContext):
    """A fal client holding the key explicitly; see `app.pipeline.segment`."""
    import fal_client

    return fal_client.SyncClient(key=ctx.settings.require("fal_key", "reconstruct"))


def _unwrap(raw):
    """Peel the extra nesting fal wraps each metadata value in.

    Observed in a real response: `"scale": [[0.99, 0.99, 0.99]]`, and the same for
    rotation and translation. Reading it as a flat list silently produces a
    one-element sequence, which every length check below rejects — so the whole
    object quietly falls back to identity rotation and unit scale. That is exactly
    what happened on the first live run, and nothing raised.
    """
    if isinstance(raw, list | tuple) and len(raw) == 1 and isinstance(raw[0], list | tuple):
        return raw[0]
    return raw


def _quat_wxyz(raw) -> tuple[float, float, float, float]:
    """fal's scalar-last quaternion to this project's scalar-first one."""
    raw = _unwrap(raw)
    if raw is None or len(raw) != 4:
        return (1.0, 0.0, 0.0, 0.0)
    x, y, z, w = (float(v) for v in raw)
    return (w, x, y, z)


def _split_scale(raw) -> tuple[np.ndarray, float]:
    """Separate an anisotropic scale into a shape part and a size part.

    The geometric mean is the isotropic component; dividing it out leaves factors
    whose product is one, which change an object's proportions without changing its
    overall size. The first belongs in the mesh, the second is recorded.
    """
    raw = _unwrap(raw)
    if raw is None or len(raw) != 3:
        return np.ones(3), 1.0
    factors = np.abs(np.asarray(raw, dtype=float))
    if not np.all(np.isfinite(factors)) or np.any(factors <= 0):
        return np.ones(3), 1.0
    isotropic = float(np.cbrt(factors.prod()))
    return factors / isotropic, isotropic


def _load_glb(data: bytes, max_faces: int) -> tuple[trimesh.Trimesh, float | None, bool] | None:
    """Return the mesh to store, plus the volume and watertightness to trust.

    The volume is measured *before* decimation and carried separately, because the
    two cannot both come from the stored file. Decimation is what makes a 22 MB
    mesh shippable, but it opens the surface — measured, and a repair pass does not
    close it again — and trimesh's volume is only meaningful on a closed one.

    So mass is computed from the mesh as reconstructed, and the file that ships is
    the small one. The volume difference between them was 0.18%; what decimation
    actually costs is the *guarantee*, not the number.

    **Measure on a welded copy, store the original.** A textured glTF holds a
    separate vertex per UV corner, so the mesh arrives split at every seam —
    measured on `room.jpg`, a floor lamp loads as 211 disconnected components with
    4780 boundary edges and zero non-manifold ones, which is an unwelded closed
    surface rather than a broken one. Ask *that* whether it is watertight and the
    answer is no for every object in every scene, so `volume` is None for every
    object in every scene and the measurement this function exists to take is never
    taken.

    The copy is not fastidiousness. `merge_vertices(merge_tex=True)` is what welds
    across the seams, and it has to pick one UV per merged vertex: on one side table
    it collapsed 11401 vertices to 9730, so 1671 of them lose a distinct UV and the
    texture stretches along every seam. Keeping textures is why `max_mesh_faces` is
    40k rather than 5k, so the shipped mesh stays exactly as it arrived. Welding
    changes no volume anyway — trimesh integrates per triangle, and across all nine
    objects the welded and unwelded volumes agree to every digit. Only the
    watertight *verdict* changes, and that is the whole point.
    """
    loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.vertices):
        return None

    welded = loaded.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)
    watertight = bool(welded.is_watertight)
    volume = float(welded.volume) if watertight else None

    if len(loaded.faces) <= max_faces:
        return loaded, volume, watertight

    # Decimation discards UVs and the material with them — measured: a mesh that
    # arrives as TextureVisuals with a 1024x1024 base colour map comes back as
    # ColorVisuals with one flat grey. So say so when it happens, rather than letting
    # an object quietly turn grey while its neighbours keep their texture.
    textured = isinstance(loaded.visual, trimesh.visual.TextureVisuals)
    simplified = loaded.simplify_quadric_decimation(face_count=max_faces)
    log.log(
        logging.WARNING if textured else logging.INFO,
        "decimated %d faces to %d%s",
        len(loaded.faces),
        len(simplified.faces),
        ", losing its texture" if textured else "",
    )
    return simplified, volume, watertight


def _download_mesh(url: str, max_faces: int):
    """Fetch and decimate one mesh, retrying a dropped transfer.

    Retries the *whole* file rather than resuming: fal's CDN gives no reliable
    range support to resume against, and at these sizes a clean retry costs less
    than the machinery to do it properly.
    """
    last: Exception | None = None
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            response = httpx.get(url, timeout=DOWNLOAD_TIMEOUT_S, follow_redirects=True)
            response.raise_for_status()
            return _load_glb(response.content, max_faces)
        except Exception as exc:
            last = exc
            if attempt + 1 < DOWNLOAD_ATTEMPTS:
                log.info("download attempt %d failed (%s); retrying", attempt + 1, exc)
                time.sleep(RETRY_BACKOFF_S * (attempt + 1))
    raise last if last else RuntimeError("download failed")


def _entry_url(entry) -> str | None:
    if isinstance(entry, dict):
        return entry.get("url")
    return entry if isinstance(entry, str) else None


def _failed(object_id: str, reason: str) -> ObjectAssets:
    return ObjectAssets(
        object_id=object_id,
        frame=AssetFrame(source=SOURCE),
        parts=[PartGeometry(part_id="body", name="body", dims_m=FALLBACK_DIMS)],
        failed=True,
        failure_reason=reason,
    )


def run(ctx: PipelineContext, segments: SegmentResult, labels: LabelResult) -> ReconstructionResult:
    if not segments.masks:
        return ReconstructionResult(objects=[])

    client = _client(ctx)
    workdir = ctx.workdir()

    # fal fetches over HTTP, so everything local is uploaded first.
    image_url = client.upload_file(ctx.image_path)
    mask_urls = [client.upload_file(mask.mask_path) for mask in segments.masks]
    log.info("reconstructing %d objects", len(mask_urls))

    result = client.subscribe(
        ctx.settings.fal_mesh_endpoint,
        # Textured output is not just prettier, it is *smaller*: SAM 3D returns a
        # low-poly mesh plus a 1024x1024 base colour map rather than baking the detail
        # into geometry. Measured on room.jpg, the same objects came back at 5.6k and
        # 12k faces textured against 138k and 415k untextured — under the decimation
        # threshold, so the texture survives.
        {"image_url": image_url, "mask_urls": mask_urls, "export_textured_glb": True},
        interval=POLL_INTERVAL_S,
    )

    glbs = result.get("individual_glbs") or []
    if not glbs and result.get("model_glb") and len(segments.masks) == 1:
        # A single-object scene comes back as one combined mesh, not a list.
        glbs = [result["model_glb"]]
    metadata = result.get("metadata") or []
    by_index = {int(m.get("object_index", i)): m for i, m in enumerate(metadata)}

    # Concurrent: sequentially these took 199s of the 399s total, and one hit the
    # timeout. Decimation runs on the worker too, so the CPU work overlaps the I/O.
    meshes: dict[int, tuple | None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        pending = {}
        for index in range(len(segments.masks)):
            url = _entry_url(glbs[index] if index < len(glbs) else None)
            if url:
                pending[pool.submit(_download_mesh, url, ctx.settings.max_mesh_faces)] = index
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            try:
                meshes[index] = future.result()
            except Exception as exc:
                log.warning("%s: mesh download failed: %s", segments.masks[index].object_id, exc)
                meshes[index] = None

    objects: list[ObjectAssets] = []
    for index, mask in enumerate(segments.masks):
        entry = meshes.get(index)

        if entry is None:
            # One lost mesh costs fidelity, not the scene: the object keeps its
            # place and reconcile falls it back to an OBB.
            log.warning("%s: no usable mesh returned", mask.object_id)
            objects.append(_failed(mask.object_id, "reconstruction returned no mesh"))
            continue

        mesh, source_volume, source_watertight = entry
        info = by_index.get(index, {})
        anisotropy, isotropic = _split_scale(info.get("scale"))
        # Proportions are shape, and shape belongs to the mesh. What is left for the
        # solver is a single size variable.
        mesh.apply_scale(anisotropy)

        path = workdir / f"{mask.object_id}.glb"
        path.write_bytes(trimesh.Scene(mesh).export(file_type="glb"))

        translation = _unwrap(info.get("translation"))
        objects.append(
            ObjectAssets(
                object_id=mask.object_id,
                frame=AssetFrame(
                    source=SOURCE,
                    rotation=_quat_wxyz(info.get("rotation")),
                    normalization_scale=isotropic,
                ),
                camera_translation=(
                    tuple(float(v) for v in translation)
                    if translation is not None and len(translation) == 3
                    else None
                ),
                parts=[
                    PartGeometry(
                        part_id="body",
                        name="body",
                        visual_mesh_path=str(path),
                        dims_m=tuple(float(v) for v in mesh.extents),
                        source_volume_m3=source_volume,
                        source_watertight=source_watertight,
                    )
                ],
            )
        )

    # Prune *after* writing, never before. A re-run whose segmentation changed
    # leaves meshes under ids nothing references and they are megabytes each — but
    # clearing first means a run that dies partway (a dropped API call, a killed
    # process) takes the previous good meshes with it and leaves the stored scene
    # pointing at nothing. Learned the hard way.
    written = {f"{obj.object_id}.glb" for obj in objects if not obj.failed}
    for stale in workdir.glob("*.glb"):
        if stale.name not in written:
            stale.unlink()

    recovered = sum(not o.failed for o in objects)
    log.info("%d/%d objects reconstructed", recovered, len(objects))
    return ReconstructionResult(objects=objects)
