"""Stage 2: metric monocular depth plus camera intrinsics.

Runs Depth Anything V2 (metric, indoor) locally through `transformers`. Indoor
because that is the domain, and *metric* because `E_depth` does two jobs and only
one of them survives relative depth:

* **Position** — where object A sits relative to object B. Any depth model gives
  this, and it is the use that appears in the literature.
* **Size** — how large an object actually is, in metres. This is the role no prior
  method in this space uses depth for, and it needs real units. With a
  relative-depth checkpoint the backprojected cloud is a layout cue and nothing
  more, which is why `depth_is_metric` has to be set honestly rather than
  optimistically.

Intrinsics are read from EXIF where the photo has them and assumed from a default
field of view otherwise, because this model does not predict them. That matters
more here than it usually would: backprojection puts lateral extent at
`(u - cx) * Z / fx`, so a wrong focal length leaves depth correct and scales
width and height — an anisotropic error in exactly the quantity stage 6 estimates.
`intrinsics_source` records which of the two happened.
"""

import contextlib
import logging
import math
from typing import Any

import numpy as np
from PIL import ExifTags, Image

from app.config import Settings
from app.pipeline.base import PipelineContext
from app.schemas import DepthResult, Intrinsics, IntrinsicsSource

__all__ = ["DepthResult", "run"]

log = logging.getLogger(__name__)

# The 35 mm still frame is 36 mm wide. EXIF reports focal length normalised to it.
FRAME_35MM_WIDTH_MM = 36.0

# (model_id, device) -> (processor, model). Loading weights costs a second or two
# and the stage cache does not help across different images, so hold them.
_MODELS: dict[tuple[str, str], tuple[Any, Any]] = {}


def _select_device(settings: Settings) -> str:
    import torch

    if settings.depth_device:
        return settings.depth_device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load(settings: Settings) -> tuple[Any, Any, str]:
    """Import torch and load the checkpoint on first use.

    Imported here rather than at module scope so that `import app.main` — and
    therefore every `--reload` and every test that never touches depth — does not
    pay torch's one-to-three second import.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    device = _select_device(settings)
    key = (settings.depth_model, device)
    if key not in _MODELS:
        log.info("loading %s on %s (first run downloads weights)", settings.depth_model, device)
        processor = AutoImageProcessor.from_pretrained(settings.depth_model)
        model = AutoModelForDepthEstimation.from_pretrained(settings.depth_model)
        model.to(device).eval()
        _MODELS[key] = (processor, model)
    processor, model = _MODELS[key]
    return processor, model, device


def _exif_focal_35mm(image: Image.Image) -> float | None:
    """FocalLengthIn35mmFilm, which phones populate and which needs no sensor size.

    The alternative — FocalLength plus FocalPlaneXResolution — needs the sensor
    dimensions to be useful and is missing or wrong more often than not.
    """
    try:
        exif = image.getexif()
    except Exception:  # a truncated or non-conformant EXIF block is not a stage failure
        return None
    if not exif:
        return None

    # FocalLengthIn35mmFilm lives in the Exif sub-IFD on most cameras, but a few
    # write it into IFD0, so check both.
    tables = [exif]
    with contextlib.suppress(Exception):
        tables.append(exif.get_ifd(ExifTags.IFD.Exif))

    for table in tables:
        value = table.get(ExifTags.Base.FocalLengthIn35mmFilm)
        if value:
            try:
                focal = float(value)
            except (TypeError, ValueError):
                continue
            if focal > 0:
                return focal
    return None


def _intrinsics(image: Image.Image, settings: Settings) -> tuple[Intrinsics, IntrinsicsSource]:
    """Returns the intrinsics and how much they can be trusted."""
    width, height = image.size
    # The 35 mm equivalent is defined against the long edge, so a portrait photo
    # normalises against its height. Using width unconditionally would report a
    # portrait phone shot as far wider-angle than it is.
    long_edge = max(width, height)

    focal_35mm = _exif_focal_35mm(image)
    if focal_35mm is not None:
        fx = focal_35mm * long_edge / FRAME_35MM_WIDTH_MM
        source = IntrinsicsSource.EXIF
    else:
        fx = (width / 2.0) / math.tan(math.radians(settings.assumed_hfov_deg) / 2.0)
        source = IntrinsicsSource.ASSUMED

    # Square pixels. True for every consumer camera, and the alternative is a
    # second unknown that nothing in the photo constrains.
    return Intrinsics(fx=fx, fy=fx, cx=width / 2.0, cy=height / 2.0), source


def run(ctx: PipelineContext) -> DepthResult:
    import torch

    settings = ctx.settings
    processor, model, device = _load(settings)

    image = Image.open(ctx.image_path).convert("RGB")
    intrinsics, source = _intrinsics(image, settings)
    if source is IntrinsicsSource.ASSUMED:
        log.info(
            "no EXIF focal length on %s; assuming %.1f deg HFOV",
            ctx.image_path.name,
            settings.assumed_hfov_deg,
        )

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Go through the model and post-process rather than the depth-estimation
    # pipeline: the pipeline's `depth` field is a PIL image normalised to 0-255
    # for display, which throws away the metric scale this model was chosen for.
    # target_sizes returns metres at *input* resolution, which is what
    # backprojecting against the masks requires.
    depth = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )[0]["predicted_depth"]

    depth_m = depth.detach().to("cpu").numpy().astype(np.float32)
    if depth_m.shape != (image.height, image.width):
        raise ValueError(
            f"depth is {depth_m.shape}, expected {(image.height, image.width)} — "
            "the mask backprojection assumes they are pixel-aligned"
        )

    path = ctx.workdir() / "depth.npy"
    np.save(path, depth_m)

    finite = depth_m[np.isfinite(depth_m)]
    log.info(
        "depth %s: %.2f-%.2f m, fx=%.1f (%s)",
        depth_m.shape,
        finite.min() if finite.size else float("nan"),
        finite.max() if finite.size else float("nan"),
        intrinsics.fx,
        source.value,
    )

    return DepthResult(
        depth_path=str(path),
        intrinsics=intrinsics,
        is_metric=settings.depth_is_metric,
        intrinsics_source=source,
    )
