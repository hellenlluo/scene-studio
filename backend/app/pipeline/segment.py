"""Stage 1: instance masks, via SAM 3 on fal.

SAM 3 is concept-prompted rather than class-agnostic — its endpoint default prompt
is literally "car" — and it takes **one concept per call**. So this stage fans out
over the noun list `labeling.inventory` produced and gathers the results.

Three things it has to do beyond calling the API:

**Run concurrently.** A room yields ten to fifteen nouns. Serially that is minutes;
concurrently it is seconds, and end-to-end latency is a reported result.

**Deduplicate.** Two concepts routinely find the same object — "sofa" and "couch",
or "side table" and "nightstand" — and the inventory pass over-generates on
purpose, because a noun that finds nothing costs one wasted call while a missing
noun costs an object. Overlapping detections are merged by IoU, keeping the
higher-scoring one.

**Filter.** The inventory also hallucinates: on the test image it asked for a
"doily" that is not there. SAM 3's own confidence is the arbiter of what actually
exists, which is the right division — the language model proposes, the segmenter
disposes.

Object ids are derived from the concept that found them and feed straight into
MJCF and glTF node names, so they are sanitised to what survives a glTF loader.
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np

from app.pipeline.base import PipelineContext
from app.schemas import ObjectMask, SegmentResult

__all__ = ["SegmentResult", "run"]

log = logging.getLogger(__name__)

MAX_WORKERS = 8
# See rigid.POLL_INTERVAL_S — the default 0.1s hammers the status endpoint.
POLL_INTERVAL_S = 1.0
# Below this, a detection is more likely the inventory pass hallucinating than an
# object. Deliberately generous: a spurious object is visible and removable in the
# viewer, a missing one is not.
MIN_SCORE = 0.35
# Two masks overlapping this much are the same object found by two names.
DEDUP_IOU = 0.7
# Smaller than this and there is nothing to reconstruct, just segmenter noise.
MIN_AREA_FRACTION = 0.0005


@dataclass
class _Detection:
    concept: str
    score: float
    mask: np.ndarray  # boolean, image resolution


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(xs.mean()), float(ys.mean())


def _object_id(mask: np.ndarray, taken: set[str]) -> str:
    """An id derived from *where* the object is, not from what it was called.

    Naming after the concept that found it reads nicely and is wrong twice over.
    It is unstable — the inventory pass returns "couch" one run and "sofa" the
    next, so the same object gets a different id and nothing can tell it is the
    same thing across a re-run. And it lies: on one real image the right-hand floor
    lamp came out as `table_lamp_5`, because "table lamp" beat "floor lamp" on score
    during deduplication, while the labelling pass correctly called it a lamp
    resting on the floor.

    A centroid in permille of the image is stable under whichever word found the
    object, and under other objects appearing or disappearing — which a positional
    index is not. `ObjectLabel.category` carries the semantics and is authoritative.

    Word characters only, because these become MJCF body names and then glTF node
    names, and a glTF loader strips the characters reserved for animation paths.
    """
    cx, cy = _centroid(mask)
    base = f"obj_{round(cx / mask.shape[1] * 1000):03d}_{round(cy / mask.shape[0] * 1000):03d}"

    candidate, suffix = base, 1
    # Two objects can share a centroid — one nested inside another, say.
    while candidate in taken:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _fetch_mask(url: str, shape: tuple[int, int]) -> np.ndarray | None:
    response = httpx.get(url, timeout=60, follow_redirects=True)
    response.raise_for_status()
    data = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
    if data is None:
        return None
    if data.ndim == 3:
        # SAM 3 returns RGBA when apply_mask is on; alpha is the mask.
        data = data[:, :, 3] if data.shape[2] == 4 else cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    if data.shape != shape:
        data = cv2.resize(data, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return data > 127


def _client(ctx: PipelineContext):
    """A fal client holding the key explicitly.

    `fal_client`'s module-level functions read FAL_KEY from os.environ, which
    pydantic-settings never populates — it loads backend/.env into the Settings
    object and leaves the process environment alone. Passing the key in is the fix;
    setting os.environ from settings would work too but leaks process-global state
    out of a pipeline stage.
    """
    import fal_client

    return fal_client.SyncClient(key=ctx.settings.require("fal_key", "segment"))


def _segment_concept(client, ctx: PipelineContext, image_url: str, concept: str, shape):
    try:
        result = client.subscribe(
            ctx.settings.fal_segment_endpoint,
            {
                "image_url": image_url,
                "prompt": concept,
                "return_multiple_masks": True,
                "include_scores": True,
                "include_boxes": True,
                "output_format": "png",
            },
            interval=POLL_INTERVAL_S,
        )
    except Exception as exc:
        # One concept failing is one missing object, not a dead scene.
        log.warning("segment %r failed: %s", concept, exc)
        return []

    masks = result.get("masks") or []
    scores = result.get("scores") or [1.0] * len(masks)

    found = []
    for entry, score in zip(masks, scores, strict=False):
        url = entry.get("url") if isinstance(entry, dict) else entry
        if not url or score < MIN_SCORE:
            continue
        mask = _fetch_mask(url, shape)
        if mask is None or mask.sum() < MIN_AREA_FRACTION * shape[0] * shape[1]:
            continue
        found.append(_Detection(concept=concept, score=float(score), mask=mask))
    return found


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _deduplicate(detections: list[_Detection]) -> list[_Detection]:
    """Merge detections of the same object found under different names.

    Highest score first, so the surviving name is the one the segmenter was most
    confident about — "sofa" over "couch" if it scored better on that word.
    """
    kept: list[_Detection] = []
    # Score descending, then concept name to break ties. Without the tiebreak the
    # surviving name depends on which thread finished first, so the same photo can
    # deduplicate differently between identical runs.
    for detection in sorted(detections, key=lambda d: (-d.score, d.concept)):
        duplicate = next((k for k in kept if _iou(detection.mask, k.mask) > DEDUP_IOU), None)
        if duplicate is not None:
            log.info(
                "%r duplicates %r (IoU > %.2f); keeping the higher score",
                detection.concept,
                duplicate.concept,
                DEDUP_IOU,
            )
            continue
        kept.append(detection)
    return kept


def run(ctx: PipelineContext, concepts: list[str]) -> SegmentResult:
    client = _client(ctx)

    # fal fetches inputs over HTTP, so a local path is not reachable — the photo is
    # uploaded once and every concept call references the same URL.
    image = cv2.imread(str(ctx.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image at {ctx.image_path}")
    shape = image.shape[:2]

    image_url = client.upload_file(ctx.image_path)
    log.info("segmenting %d concepts", len(concepts))

    detections: list[_Detection] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(_segment_concept, client, ctx, image_url, concept, shape)
            for concept in concepts
        ]
        for future in concurrent.futures.as_completed(futures):
            detections.extend(future.result())

    log.info("%d raw detections", len(detections))
    kept = _deduplicate(detections)
    log.info("%d after dedup", len(kept))

    # Reading order, so the numbers `annotate_masks` draws follow the picture and
    # the list does not reshuffle because two API calls finished in a different
    # order. Everything downstream indexes into this list by position.
    kept.sort(key=lambda d: tuple(reversed(_centroid(d.mask))))

    workdir = ctx.workdir()
    masks = []
    taken: set[str] = set()
    for index, detection in enumerate(kept):
        path = workdir / f"mask_{index:02d}.png"
        cv2.imwrite(str(path), detection.mask.astype(np.uint8) * 255)

        object_id = _object_id(detection.mask, taken)
        taken.add(object_id)

        ys, xs = np.nonzero(detection.mask)
        masks.append(
            ObjectMask(
                object_id=object_id,
                mask_path=str(path),
                bbox_px=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                area_px=int(detection.mask.sum()),
            )
        )

    # Pruned after writing rather than before, so a run that dies partway does not
    # take the previous good masks with it. See the same note in `rigid`.
    written = {Path(mask.mask_path).name for mask in masks}
    for stale in workdir.glob("mask_*.png"):
        if stale.name not in written:
            stale.unlink()

    return SegmentResult(masks=masks, image_size_px=(shape[1], shape[0]))
