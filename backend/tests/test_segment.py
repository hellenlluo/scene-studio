"""Segmentation logic that does not need the API.

The fal calls are untestable here, but the parts most likely to be wrong are not
the calls — they are what happens to the results. Deduplication in particular:
the inventory pass over-generates on purpose, so several concepts routinely find
the same object, and a missed merge means a duplicate object in the scene that
interpenetrates itself.
"""

import numpy as np
import pytest

from app.pipeline.segment import _deduplicate, _Detection, _iou, _object_id


def _mask(x0: int, x1: int, size: int = 100) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[20:80, x0:x1] = True
    return mask


# --- object ids ---------------------------------------------------------------


def test_ids_come_from_position_not_from_the_concept():
    """Naming after the concept that found it is unstable and misleading: the
    inventory returns "couch" one run and "sofa" the next, and on one real image
    the floor lamp came out as `table_lamp_5` because "table lamp" won the dedup on
    score. Position does not move when the vocabulary does."""
    mask = _mask(0, 50)
    assert _object_id(mask, set()).startswith("obj_")
    # The same geometry found under a different word gets the same id.
    assert _object_id(mask, set()) == _object_id(_mask(0, 50), set())


def test_the_id_encodes_where_the_object_is():
    left = _object_id(_mask(0, 20), set())
    right = _object_id(_mask(80, 100), set())
    assert left != right
    # Permille of image width: a mask at x 0-20 of 100 has centroid ~9.5%.
    assert left == "obj_095_495"
    assert right == "obj_895_495"


def test_ids_survive_a_gltf_loader():
    """These become MJCF body names and then glTF node names, and GLTFLoader strips
    the characters reserved for animation paths."""
    stripped = set("[]./: ")
    assert not (set(_object_id(_mask(0, 50), set())) & stripped)


def test_a_centroid_collision_still_yields_distinct_ids():
    """Two objects can share a centroid — one nested inside another."""
    taken: set[str] = set()
    first = _object_id(_mask(0, 50), taken)
    taken.add(first)
    second = _object_id(_mask(0, 50), taken)
    assert first != second


def test_ids_are_unaffected_by_other_objects_appearing():
    """Unlike a positional index, which shifts when anything before it is added or
    removed — so a viewer could not tell it was still looking at the same object."""
    target = _mask(60, 90)
    alone = _object_id(target, set())
    crowded = _object_id(target, {_object_id(_mask(0, 20), set())})
    assert alone == crowded


# --- overlap ------------------------------------------------------------------


def test_iou_of_identical_masks_is_one():
    assert _iou(_mask(0, 50), _mask(0, 50)) == pytest.approx(1.0)


def test_iou_of_disjoint_masks_is_zero():
    assert _iou(_mask(0, 20), _mask(60, 80)) == 0.0


def test_iou_of_empty_masks_does_not_divide_by_zero():
    empty = np.zeros((10, 10), dtype=bool)
    assert _iou(empty, empty) == 0.0


def test_iou_is_intersection_over_union():
    # Half overlapping: 20 wide each, 10 shared -> 10 / 30.
    assert _iou(_mask(0, 20), _mask(10, 30)) == pytest.approx(1 / 3)


# --- deduplication ------------------------------------------------------------


def test_the_same_object_found_twice_is_merged():
    """ "sofa" and "couch" are one piece of furniture. Keeping both would put two
    interpenetrating objects in the scene."""
    kept = _deduplicate(
        [
            _Detection(concept="sofa", score=0.9, mask=_mask(0, 50)),
            _Detection(concept="couch", score=0.7, mask=_mask(0, 50)),
        ]
    )
    assert len(kept) == 1


def test_the_higher_scoring_name_survives():
    """The surviving concept becomes the object id, so it should be the word the
    segmenter was most confident about."""
    kept = _deduplicate(
        [
            _Detection(concept="couch", score=0.6, mask=_mask(0, 50)),
            _Detection(concept="sofa", score=0.95, mask=_mask(0, 50)),
        ]
    )
    assert [d.concept for d in kept] == ["sofa"]


def test_distinct_objects_are_both_kept():
    kept = _deduplicate(
        [
            _Detection(concept="sofa", score=0.9, mask=_mask(0, 30)),
            _Detection(concept="side table", score=0.8, mask=_mask(60, 90)),
        ]
    )
    assert {d.concept for d in kept} == {"sofa", "side table"}


def test_partial_overlap_below_the_threshold_is_not_merged():
    """A lamp standing on a table overlaps it in the image without being it."""
    kept = _deduplicate(
        [
            _Detection(concept="table", score=0.9, mask=_mask(0, 60)),
            _Detection(concept="lamp", score=0.8, mask=_mask(50, 100)),
        ]
    )
    assert len(kept) == 2


def test_deduplication_is_order_independent():
    """Sorted by score, not by arrival — the calls complete concurrently and
    as_completed order is arbitrary."""
    a = _Detection(concept="sofa", score=0.95, mask=_mask(0, 50))
    b = _Detection(concept="couch", score=0.6, mask=_mask(0, 50))
    assert [d.concept for d in _deduplicate([a, b])] == ["sofa"]
    assert [d.concept for d in _deduplicate([b, a])] == ["sofa"]


def test_nothing_in_nothing_out():
    assert _deduplicate([]) == []


# --- a part is not a second object ---------------------------------------------


def _det(concept: str, score: float, box):
    """A detection whose mask is the given (x0, y0, x1, y1) rectangle."""
    mask = np.zeros((100, 100), bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return _Detection(concept=concept, score=score, mask=mask)


def test_a_detection_inside_another_is_dropped_as_a_part():
    """The observed failure: an open vocabulary names a whole and its part, and the
    segmenter returns both, nested.

    Measured on `room2.png` — the "ceramic vase" mask sat 99% inside the "potted
    plant" mask, but the plant is three times the area so IoU was only 0.38 and the
    scene carried the vase twice, once alone and once with flowers in it.
    """
    plant = _det("potted plant", 0.8, (10, 10, 60, 90))
    vase = _det("ceramic vase", 0.9, (20, 55, 50, 88))  # higher score, wholly inside
    kept = {d.concept for d in _deduplicate([plant, vase])}
    assert kept == {"potted plant"}, "the whole survives, not the higher-scoring part"


def test_two_objects_that_merely_overlap_both_survive():
    """Containment, not overlap. Adjacent or partly overlapping detections are two
    objects and both belong in the scene."""
    a = _det("book", 0.9, (10, 10, 50, 50))
    b = _det("mug", 0.8, (40, 40, 80, 80))  # ~6% of b is inside a
    kept = {d.concept for d in _deduplicate([a, b])}
    assert kept == {"book", "mug"}


def test_the_iou_rule_still_keeps_the_higher_score():
    """The two rules disagree about who survives and must not be merged: IoU keeps
    the better score, containment keeps the larger."""
    sofa = _det("sofa", 0.9, (10, 10, 60, 60))
    couch = _det("couch", 0.7, (11, 11, 60, 60))  # near-identical, IoU well over 0.7
    kept = {d.concept for d in _deduplicate([sofa, couch])}
    assert kept == {"sofa"}


def test_the_surviving_phrase_is_recorded_on_the_mask():
    """The phrase that found a mask is the strongest evidence about what the object
    is, and stage 3's second pass used to re-derive it from pixels having never been
    told it — a bookshelf came back as `book`, an armchair as `towel`.

    `_object_id` argues at length against naming *ids* after the phrase, and it is
    right; this carries the phrase alongside the positional id rather than instead
    of it.
    """
    det = _det("bookshelf", 0.9, (10, 10, 60, 90))
    kept = _deduplicate([det])
    assert kept[0].concept == "bookshelf"
