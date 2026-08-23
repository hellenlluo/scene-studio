"""Draw numbered outlines on a photo, so a VLM can refer to objects by index.

Set-of-mark prompting. The alternative — sending one cropped image per object —
loses the very context the question depends on: a crop of a mug and a crop of a
bucket are the same picture, and what separates them is the room around them. It is
also one API call instead of N.

The numbers are 1-based because they are read by a human-facing model, and callers
map them back to `object_id` by position in `SegmentResult.masks`.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from app.schemas import SegmentResult

__all__ = ["annotate_masks"]

log = logging.getLogger(__name__)

# Bright, distinguishable, and none of them a colour a room is likely to be.
_PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 32),
    (220, 64, 220),
    (32, 220, 220),
    (255, 128, 32),
    (160, 96, 255),
]
_OUTLINE_PX = 3
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label_anchor(mask: np.ndarray) -> tuple[int, int]:
    """A point inside the mask to put the number on.

    The centroid, pulled back onto the mask when the shape is concave enough that
    its centroid falls outside — a number floating in space next to a C-shaped
    object is ambiguous about which object it names.
    """
    ys, xs = np.nonzero(mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    if mask[cy, cx]:
        return cx, cy
    nearest = np.argmin((xs - cx) ** 2 + (ys - cy) ** 2)
    return int(xs[nearest]), int(ys[nearest])


def annotate_masks(image_path: Path, segments: SegmentResult) -> bytes:
    """Return PNG bytes of the photo with each mask outlined and numbered."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"cannot read image at {image_path}")

    canvas = image.copy()
    scale = max(0.5, min(image.shape[:2]) / 900.0)

    for index, entry in enumerate(segments.masks, start=1):
        mask = cv2.imread(entry.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            log.warning("mask %s is unreadable; not drawn", entry.mask_path)
            continue
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)

        binary = (mask > 127).astype(np.uint8)
        if not binary.any():
            continue

        # BGR, because that is what cv2 writes.
        colour = _PALETTE[(index - 1) % len(_PALETTE)][::-1]
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, _OUTLINE_PX)

        x, y = _label_anchor(binary)
        text = str(index)
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 2)
        # A filled chip behind the digits: an outline number over busy texture is
        # unreadable, and an unreadable index makes the whole labelling pass useless.
        cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + 6), colour, -1)
        cv2.putText(canvas, text, (x, y), _FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)

    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("failed to encode the annotated image")
    return buffer.tobytes()
