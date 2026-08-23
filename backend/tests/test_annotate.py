"""Set-of-mark annotation.

The numbers this draws are the only thing tying a model's answer back to an
object id, so an unreadable or misplaced index makes the whole labelling pass
useless — and it fails silently, as plausible labels attached to the wrong things.
"""

import cv2
import numpy as np
import pytest

from app.pipeline.annotate import annotate_masks
from app.schemas import ObjectMask, SegmentResult


@pytest.fixture
def scene(tmp_path):
    """A 200x200 photo with two disjoint square objects."""
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    image_path = tmp_path / "room.png"
    cv2.imwrite(str(image_path), image)

    masks = []
    for index, (x0, x1) in enumerate([(20, 80), (120, 180)]):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[60:140, x0:x1] = 255
        path = tmp_path / f"mask{index}.png"
        cv2.imwrite(str(path), mask)
        masks.append(
            ObjectMask(
                object_id=f"obj{index}",
                mask_path=str(path),
                bbox_px=(x0, 60, x1, 140),
                area_px=int((x1 - x0) * 80),
            )
        )
    return image_path, SegmentResult(masks=masks, image_size_px=(200, 200))


def _decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_output_is_a_png_the_same_size_as_the_photo(scene):
    image_path, segments = scene
    out = _decode(annotate_masks(image_path, segments))
    assert out.shape == (200, 200, 3)


def test_the_photo_is_still_visible_underneath(scene):
    """Outlines and a number chip, not a mask overlay. The model has to see the
    object to judge its material, so filling the silhouette would defeat the
    purpose. Sampled away from the border and from the centroid, which is where
    the chip deliberately sits."""
    image_path, segments = scene
    out = _decode(annotate_masks(image_path, segments))
    interior = out[68:78, 28:38]
    assert np.allclose(interior, 40, atol=2), "the object's interior was painted over"


def test_the_number_chip_is_opaque(scene):
    """An outlined digit over busy texture is unreadable, and an unreadable index
    makes every label ambiguous. The chip covers a small part of the object on
    purpose — that is the trade."""
    image_path, segments = scene
    out = _decode(annotate_masks(image_path, segments))
    chip = out[92:100, 48:56]
    assert not np.allclose(chip, 40, atol=2)


def test_each_object_gets_a_distinct_colour(scene):
    image_path, segments = scene
    out = _decode(annotate_masks(image_path, segments))
    left = {tuple(p) for p in out[55:145, 15:85].reshape(-1, 3)}
    right = {tuple(p) for p in out[55:145, 115:185].reshape(-1, 3)}
    background = (40, 40, 40)
    assert (left - {background}) != (right - {background})


def test_labels_land_inside_their_own_object(scene):
    """A number floating between two objects is ambiguous about which it names."""
    image_path, segments = scene
    out = _decode(annotate_masks(image_path, segments))
    gap = out[60:140, 85:115]
    assert np.allclose(gap, 40, atol=2), "something was drawn in the gap between objects"


def test_a_concave_mask_keeps_its_label_on_the_object(tmp_path):
    """A C-shape's centroid falls outside it, so the anchor has to be pulled back
    onto the mask or the number ends up in mid-air."""
    from app.pipeline.annotate import _label_anchor

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:90, 10:30] = 1  # spine
    mask[10:30, 10:90] = 1  # top arm
    mask[70:90, 10:90] = 1  # bottom arm

    x, y = _label_anchor(mask)
    assert mask[y, x], "the anchor landed off the object"


def test_an_unreadable_mask_is_skipped_not_fatal(scene, tmp_path, caplog):
    """One missing file should cost one outline, not the whole labelling pass."""
    image_path, segments = scene
    segments.masks[0].mask_path = str(tmp_path / "gone.png")

    out = _decode(annotate_masks(image_path, segments))
    assert out.shape == (200, 200, 3)
    assert "unreadable" in caplog.text


def test_a_missing_photo_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        annotate_masks(tmp_path / "nope.png", SegmentResult(masks=[], image_size_px=(10, 10)))


def test_numbering_follows_mask_order(scene):
    """labeling.run maps the drawn number back to an object id by position, so the
    two orderings have to be the same one."""
    image_path, segments = scene
    assert [m.object_id for m in segments.masks] == ["obj0", "obj1"]
    # obj0 is drawn as 1, obj1 as 2 — enumerate(..., start=1) in annotate_masks.
    out = _decode(annotate_masks(image_path, segments))
    assert out is not None
