"""Stage 3.5: give an occluder's pixels back to the object it splits.

Runs between `label` and `reconstruct`, which is the only window where both
pieces of evidence exist: the masks (from stage 1) and what each one *is* (from
stage 3), while nothing has been reconstructed yet.

**The failure it exists for.** A throw draped over an armchair is its own object
to a segmenter, and SAM 3 gives every pixel to exactly one instance. So the chair
comes back as two disconnected components with a throw-shaped hole between them,
and SAM 3D — handed that mask — reconstructs a chair with one arm sheared off.
Measured on `room2.png`: 127,411 px and 30,344 px, and an armchair missing its
left side. The throw's own crop, meanwhile, gets completed into a solid block
that reads as a second chair and overlaps the first by 35 mm.

**Why the trigger is geometric, not a word list.** The obvious fix — name the
covering with its support at inventory time — does not work, and `labeling`
records the measurement: the armchair mask was byte-identical with and without
"throw blanket" in the concept list, because SAM 3 segments "armchair" from
armchair pixels no matter what else was asked for. The obvious second fix — a
DRAPE_CATEGORIES set — fails differently: throw, blanket, sheet, tablecloth,
coat, towel, runner, cover is an open vocabulary, and the one word that matters
is always the one not in the set.

What is actually diagnostic needs no vocabulary at all:

    if mask A is disconnected and mask B lies in the gap, B is occluding A

That is a statement about pixels, and it is true of a cable across a desk or a
cat on a sofa exactly as it is true of a throw. A mask that is *not* split
triggers nothing, so the rule costs nothing on the scenes that never had the
problem — measured on `room1.png`, zero merges.

**The support relation is the first gate.** Geometry alone would also absorb a
person standing in front of a bookshelf, which splits it just as well and is
emphatically not part of it. So a merge additionally requires that stage 3 called
B *supported by* A: a throw rests on the chair it covers, a person does not rest
on the bookshelf.

**Depth is the second gate, and it is the one that matters.** Support is not
enough on its own, because the awkward case passes it: a small plant standing on
a bookshelf shelf *is* supported by that bookshelf and *does* sever its mask, and
merging the two would be nonsense. Both room2 labels — a plant and a vase on the
bookshelf — are exactly this. What distinguishes them from a throw is not
semantics but distance. A covering lies *on* what it covers, so the two surfaces
are continuous across their shared boundary; an object merely standing in front
of another is separated by its standoff, and depth steps across that seam by tens
of centimetres. So the third condition is that the two agree in depth along the
seam they share, within `max_occluder_depth_step_m`.

This is the check that answers "are these actually one thing", and it needs no
amodal completion to do it: the question is whether two visible surfaces touch,
which stage 2 already measured. Completing the occluded object would be a way to
*recover* what was lost, but SAM 3D already does that from a whole mask — the
merge exists to hand it one.

**What a merge costs.** The occluder stops being its own object, so a scene loses
the throw as something you can select or move. That is the right trade here and
not a general one: cloth has no rigid pose to recover, every part in this pipeline
gets an OBB or a convex hull, and a blanket modelled as a solid block was the
original bug. An occluder that is genuinely rigid and genuinely separable would be
better served by reconstructing both and letting `verify` arbitrate. So every merge
is logged with the seam it was decided on, and returned in `OcclusionResult.merged`
for a caller that wants to surface it. Nothing persists it into `SceneSpec` yet —
`removed_objects` is the precedent to follow if a scene should carry its merges the
way it carries its deletions.
"""

import logging

import cv2
import numpy as np
from pydantic import BaseModel
from scipy import ndimage

from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, LabelResult, ObjectMask, SegmentResult

__all__ = ["MergedOccluder", "OcclusionResult", "run"]

log = logging.getLogger(__name__)

# A component below this share of the mask's area is speckle, not a severed limb.
# Segmentation edges are ragged and routinely leave a few dozen stray pixels; without
# a floor here every second mask would look "disconnected" and the rule would fire on
# noise. On room2 the armchair's two real pieces are 81% and 19% of its area.
MIN_COMPONENT_FRACTION = 0.05

# Masks are adjacent rather than overlapping — segmentation cuts the occluder out of
# the object, so the two share a boundary and intersect in nothing. Dilating by a few
# pixels is what turns "touches" into a test that can succeed at all.
BRIDGE_DILATION_PX = 3

# Below this the seam is a few stray pixels and its median says nothing.
MIN_SEAM_PX = 20


class MergedOccluder(BaseModel):
    """One occluder folded into the object it was splitting."""

    occluder_id: str
    absorbed_into: str
    components_bridged: int
    seam_depth_step_m: float = 0.0
    reason: str = ""


class OcclusionResult(BaseModel):
    segments: SegmentResult
    labels: LabelResult
    merged: list[MergedOccluder] = []


def _significant_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Label connected components and return the ones too big to be speckle."""
    labelled, count = ndimage.label(mask)
    if count < 2:
        return labelled, list(range(1, count + 1))
    total = float(mask.sum())
    sizes = ndimage.sum_labels(mask, labelled, index=range(1, count + 1))
    return labelled, [
        i + 1 for i, size in enumerate(sizes) if size >= MIN_COMPONENT_FRACTION * total
    ]


def _bridged(labelled: np.ndarray, keep: list[int], occluder: np.ndarray) -> int:
    """How many distinct components of the split object this occluder reaches."""
    grown = cv2.dilate(
        occluder.astype(np.uint8),
        np.ones((BRIDGE_DILATION_PX * 2 + 1,) * 2, np.uint8),
    ).astype(bool)
    return sum(1 for component in keep if (labelled == component)[grown].any())


def _read(mask: ObjectMask) -> np.ndarray:
    return cv2.imread(mask.mask_path, cv2.IMREAD_GRAYSCALE) > 127


def _seam_depth_step(
    depth: np.ndarray, host: np.ndarray, occluder: np.ndarray
) -> float | None:
    """Median depth disagreement along the boundary the two masks share.

    Sampled on both sides of the seam rather than across the whole mask, because
    a draped textile and the thing under it only have to meet *where they touch* —
    a throw hanging off the front of a chair is metres from its backrest and that
    is not evidence of anything.

    None when the seam is too thin to measure, which is treated as no contact: an
    occluder that barely touches its host is the case this gate exists to reject.
    """
    kernel = np.ones((BRIDGE_DILATION_PX * 2 + 1,) * 2, np.uint8)
    near_occluder = cv2.dilate(occluder.astype(np.uint8), kernel).astype(bool)
    near_host = cv2.dilate(host.astype(np.uint8), kernel).astype(bool)

    host_side = depth[near_occluder & host]
    occluder_side = depth[near_host & occluder]
    host_side = host_side[np.isfinite(host_side)]
    occluder_side = occluder_side[np.isfinite(occluder_side)]
    if len(host_side) < MIN_SEAM_PX or len(occluder_side) < MIN_SEAM_PX:
        return None
    return abs(float(np.median(host_side) - np.median(occluder_side)))


def run(
    ctx: PipelineContext,
    segments: SegmentResult,
    labels: LabelResult,
    depth: DepthResult,
) -> OcclusionResult:
    """Union each occluding mask into the object it splits, and drop it."""
    if not segments.masks:
        return OcclusionResult(segments=segments, labels=labels)

    parent_of = {label.object_id: label.support_parent for label in labels.labels}
    by_id = {mask.object_id: mask for mask in segments.masks}
    children: dict[str, list[str]] = {}
    for child, parent in parent_of.items():
        if parent in by_id and child in by_id:
            children.setdefault(parent, []).append(child)

    pixels = {mask.object_id: _read(mask) for mask in segments.masks}
    depth_m = np.load(depth.depth_path)
    max_step = ctx.settings.max_occluder_depth_step_m
    absorbed: dict[str, MergedOccluder] = {}
    rewritten: dict[str, np.ndarray] = {}

    for mask in segments.masks:
        candidates = children.get(mask.object_id, [])
        if not candidates:
            continue

        target = pixels[mask.object_id]
        labelled, keep = _significant_components(target)
        if len(keep) < 2:
            continue  # not split, so nothing is splitting it

        merged_here = []
        for child in candidates:
            if child in absorbed:
                continue
            bridged = _bridged(labelled, keep, pixels[child])
            if bridged < 2:
                continue  # rests on it, but does not sever it — an ordinary child

            step = _seam_depth_step(depth_m, target, pixels[child])
            if step is None or step > max_step:
                # Severs it and rests on it, but stands clear of it — a plant on a
                # shelf, not a throw on a chair. Reconstructing the two as one shape
                # would be worse than the split it would have fixed.
                log.info(
                    "occlusion: %s splits %s but is %s from it; left separate",
                    child,
                    mask.object_id,
                    "not in contact" if step is None else f"{1000 * step:.0f} mm",
                )
                continue

            absorbed[child] = MergedOccluder(
                occluder_id=child,
                absorbed_into=mask.object_id,
                components_bridged=bridged,
                seam_depth_step_m=step,
                reason=(
                    f"splits {mask.object_id} into {len(keep)} pieces, rests on it, and "
                    f"meets it within {1000 * step:.0f} mm at their shared boundary"
                ),
            )
            merged_here.append(child)

        if merged_here:
            union = target.copy()
            for child in merged_here:
                union |= pixels[child]
            rewritten[mask.object_id] = union

    if not absorbed:
        return OcclusionResult(segments=segments, labels=labels)

    # A new file rather than an overwrite: `mask_NN.png` is referenced by the cached
    # SegmentResult, and rewriting it in place would leave a cache hit returning the
    # original mask list pointing at merged pixels.
    workdir = ctx.workdir()
    masks: list[ObjectMask] = []
    for index, mask in enumerate(segments.masks):
        if mask.object_id in absorbed:
            continue
        union = rewritten.get(mask.object_id)
        if union is None:
            masks.append(mask)
            continue
        path = workdir / f"mask_{index:02d}.merged.png"
        cv2.imwrite(str(path), union.astype(np.uint8) * 255)
        ys, xs = np.nonzero(union)
        masks.append(
            mask.model_copy(
                update={
                    "mask_path": str(path),
                    "bbox_px": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    "area_px": int(union.sum()),
                }
            )
        )

    # Anything resting *on* an absorbed occluder now rests on its host — a book on the
    # throw is on the armchair. Left pointing at a dropped id it would read as an
    # object whose support is missing, which `scale` reports as a floor contact.
    kept_labels = []
    for label in labels.labels:
        if label.object_id in absorbed:
            continue
        parent = label.support_parent
        if parent in absorbed:
            label = label.model_copy(update={"support_parent": absorbed[parent].absorbed_into})
        kept_labels.append(label)

    for merge in absorbed.values():
        log.info(
            "occlusion: %s absorbed into %s (bridged %d components)",
            merge.occluder_id,
            merge.absorbed_into,
            merge.components_bridged,
        )

    return OcclusionResult(
        segments=segments.model_copy(update={"masks": masks}),
        labels=labels.model_copy(update={"labels": kept_labels}),
        merged=list(absorbed.values()),
    )
