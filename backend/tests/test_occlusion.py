"""Stage 3.5: folding an occluder back into the object it splits.

Every case is built from rectangles rather than a real photo, because what the
stage decides is a question about pixels and support relations only — a mask that
is split, and something lying in the split. Hand-drawn masks make the two
conditions independently switchable, which is what pins that *both* are required.
"""

import cv2
import numpy as np
import pytest

from app.pipeline import occlusion
from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, Intrinsics, LabelResult, ObjectLabel, ObjectMask, SegmentResult

H, W = 200, 200


@pytest.fixture
def ctx(tmp_path):
    image = tmp_path / "room.png"
    image.write_bytes(b"only the bytes are hashed")
    return PipelineContext.create("occlusion-test", image)


def _depth(ctx, arr: np.ndarray | None = None) -> DepthResult:
    """A depth map in metres. Flat by default, so contact is the uninteresting case
    and only the tests that care about standoff have to say so."""
    if arr is None:
        arr = np.full((H, W), 2.0, np.float32)
    path = ctx.workdir() / "depth.npy"
    np.save(path, arr.astype(np.float32))
    return DepthResult(
        depth_path=str(path),
        intrinsics=Intrinsics(fx=500.0, fy=500.0, cx=W / 2, cy=H / 2),
        is_metric=True,
    )


def _mask(ctx, name: str, *boxes: tuple[int, int, int, int]) -> tuple[ObjectMask, np.ndarray]:
    """A mask covering one or more axis-aligned boxes, written where the stage reads."""
    arr = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        arr[y0:y1, x0:x1] = True
    return _written(ctx, name, arr)


def _written(ctx, name: str, arr: np.ndarray) -> tuple[ObjectMask, np.ndarray]:
    path = ctx.workdir() / f"{name}.png"
    cv2.imwrite(str(path), arr.astype(np.uint8) * 255)
    ys, xs = np.nonzero(arr)
    return (
        ObjectMask(
            object_id=name,
            mask_path=str(path),
            bbox_px=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            area_px=int(arr.sum()),
        ),
        arr,
    )


def _labels(**parents: str | None) -> LabelResult:
    return LabelResult(
        labels=[
            ObjectLabel(object_id=oid, category="other", support_parent=parent)
            for oid, parent in parents.items()
        ]
    )


def _split_chair(ctx):
    """A chair severed into two halves by a throw lying in the gap between them."""
    chair, _ = _mask(ctx, "chair", (10, 10, 80, 190), (120, 10, 190, 190))
    throw, _ = _mask(ctx, "throw", (80, 40, 120, 160))
    return chair, throw


# --- the two conditions, together and separately -------------------------------


def test_an_occluder_that_splits_its_support_is_absorbed(ctx):
    chair, throw = _split_chair(ctx)
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx),
    )

    assert [m.occluder_id for m in result.merged] == ["throw"]
    assert result.merged[0].absorbed_into == "chair"
    assert [m.object_id for m in result.segments.masks] == ["chair"]


def test_the_absorbed_pixels_join_the_object(ctx):
    """The point of the merge: SAM 3D must see one continuous shape, not two."""
    chair, throw = _split_chair(ctx)
    before = chair.area_px
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx),
    )

    merged = result.segments.masks[0]
    assert merged.area_px == before + throw.area_px
    assert merged.bbox_px == (10, 10, 189, 189)

    arr = cv2.imread(merged.mask_path, cv2.IMREAD_GRAYSCALE) > 127
    from scipy import ndimage

    assert ndimage.label(arr)[1] == 1, "the halves have to end up connected"


def test_an_unsplit_object_absorbs_nothing(ctx):
    """A cushion resting on a whole sofa is an ordinary child, not an occluder.

    This is the case that makes the rule cheap: scenes without the problem pay
    nothing, because a mask that is not severed cannot trigger the test at all.
    """
    sofa, _ = _mask(ctx, "sofa", (10, 10, 190, 190))
    cushion, _ = _mask(ctx, "cushion", (60, 60, 100, 100))
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[sofa, cushion], image_size_px=(W, H)),
        _labels(sofa=None, cushion="sofa"),
        _depth(ctx),
    )

    assert result.merged == []
    assert len(result.segments.masks) == 2


def test_something_that_splits_an_object_without_resting_on_it_is_left_alone(ctx):
    """The safety gate. A person in front of a bookshelf severs it exactly as a
    throw severs a chair, and is emphatically not part of it. Geometry cannot tell
    those apart; the support relation can, and here it says no."""
    shelf, person = _split_chair(ctx)
    shelf = shelf.model_copy(update={"object_id": "shelf"})
    person = person.model_copy(update={"object_id": "person"})

    result = occlusion.run(
        ctx,
        SegmentResult(masks=[shelf, person], image_size_px=(W, H)),
        _labels(shelf=None, person=None),  # stands on the floor, not on the shelf
        _depth(ctx),
    )

    assert result.merged == []
    assert len(result.segments.masks) == 2


def test_an_occluder_touching_only_one_half_is_left_alone(ctx):
    """Lying against an object is not lying across it. Requiring two components is
    what separates a throw draped over a chair from a blanket folded on its seat."""
    # The folded throw is cut out of the left half, the way segmentation cuts a
    # covering out of what it covers — so the two masks are disjoint, and the left
    # half is notched but still one piece.
    chair = np.zeros((H, W), bool)
    chair[10:190, 10:80] = True
    chair[10:190, 120:190] = True
    chair[40:160, 55:80] = False
    chair, _ = _written(ctx, "chair", chair)
    folded, _ = _mask(ctx, "folded", (55, 40, 80, 160))
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, folded], image_size_px=(W, H)),
        _labels(chair=None, folded="chair"),
        _depth(ctx),
    )

    assert result.merged == []


def test_speckle_does_not_count_as_a_severed_object(ctx):
    """Segmentation edges are ragged and leave stray pixels. Without an area floor
    every second mask would look split and the rule would fire on noise."""
    chair, _ = _mask(ctx, "chair", (10, 10, 180, 190), (194, 194, 197, 197))
    throw, _ = _mask(ctx, "throw", (150, 40, 196, 196))
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx),
    )

    assert result.merged == [], "a 9 px fleck is not a second component"


# --- knock-on effects ----------------------------------------------------------


def test_a_child_of_the_absorbed_object_is_reparented(ctx):
    """A book on the throw is on the armchair once the throw stops existing.

    Left pointing at a dropped id, `scale` reads the missing parent as a floor
    contact and reports the book's height as a gap against the floor.
    """
    chair, throw = _split_chair(ctx)
    book, _ = _mask(ctx, "book", (85, 60, 115, 80))
    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw, book], image_size_px=(W, H)),
        _labels(chair=None, throw="chair", book="throw"),
        _depth(ctx),
    )

    assert [m.occluder_id for m in result.merged] == ["throw"]
    parents = {label.object_id: label.support_parent for label in result.labels.labels}
    assert parents == {"chair": None, "book": "chair"}


def test_the_original_mask_file_is_not_overwritten(ctx):
    """`mask_NN.png` is referenced by the cached SegmentResult. Rewriting it in place
    would leave a segment cache hit returning the original mask list — pointing at
    merged pixels."""
    chair, throw = _split_chair(ctx)
    original = chair.mask_path
    before = cv2.imread(original, cv2.IMREAD_GRAYSCALE).sum()

    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx),
    )

    assert result.segments.masks[0].mask_path != original
    assert cv2.imread(original, cv2.IMREAD_GRAYSCALE).sum() == before


def test_a_scene_with_nothing_to_merge_is_returned_unchanged(ctx):
    sofa, _ = _mask(ctx, "sofa", (10, 10, 190, 190))
    segments = SegmentResult(masks=[sofa], image_size_px=(W, H))
    labels = _labels(sofa=None)
    result = occlusion.run(ctx, segments, labels, _depth(ctx))

    assert result.segments == segments
    assert result.labels == labels


def test_an_empty_scene_is_handled(ctx):
    empty = SegmentResult(masks=[], image_size_px=(W, H))
    result = occlusion.run(ctx, empty, LabelResult(labels=[]), _depth(ctx))
    assert result.merged == []


# --- the depth gate ------------------------------------------------------------


def test_something_standing_in_front_of_its_own_support_is_left_alone(ctx):
    """The case support alone cannot reject.

    A small plant standing on a bookshelf shelf is genuinely `supported_by` that
    bookshelf and genuinely severs its mask, so it passes both geometric tests. Only
    depth separates it from a throw: the plant stands 40 cm clear of the shelves
    behind it, and reconstructing the two as one shape would be worse than the split.
    """
    shelf, plant = _split_chair(ctx)
    shelf = shelf.model_copy(update={"object_id": "shelf"})
    plant = plant.model_copy(update={"object_id": "plant"})

    depth = np.full((H, W), 2.4, np.float32)
    depth[40:160, 80:120] = 2.0  # the plant, 400 mm nearer than the shelves

    result = occlusion.run(
        ctx,
        SegmentResult(masks=[shelf, plant], image_size_px=(W, H)),
        _labels(shelf=None, plant="shelf"),
        _depth(ctx, depth),
    )

    assert result.merged == [], "a 400 mm standoff is not contact"


def test_a_covering_in_contact_is_still_absorbed(ctx):
    """The other side of the same threshold: a throw lies on the chair, so the two
    are continuous across the seam and the merge goes ahead."""
    chair, throw = _split_chair(ctx)
    depth = np.full((H, W), 2.4, np.float32)
    depth[40:160, 80:120] = 2.38  # 20 mm proud, the thickness of folded cloth

    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx, depth),
    )

    assert [m.occluder_id for m in result.merged] == ["throw"]
    assert result.merged[0].seam_depth_step_m == pytest.approx(0.02, abs=1e-3)


def test_the_step_is_measured_at_the_seam_not_across_the_whole_mask(ctx):
    """A throw hanging off the front of a chair is far from its backrest, and that
    distance is not evidence of anything. Only where the two actually meet counts."""
    chair, throw = _split_chair(ctx)
    depth = np.full((H, W), 2.4, np.float32)
    depth[150:190, :] = 1.2  # the near end of the scene, far from the shared boundary

    result = occlusion.run(
        ctx,
        SegmentResult(masks=[chair, throw], image_size_px=(W, H)),
        _labels(chair=None, throw="chair"),
        _depth(ctx, depth),
    )

    assert [m.occluder_id for m in result.merged] == ["throw"]
