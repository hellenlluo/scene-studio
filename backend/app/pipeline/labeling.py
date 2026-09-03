"""Stage 3: what is in the photo, and what is it made of.

Two passes, because segmentation and labelling need each other:

    inventory(ctx)          photo -> noun phrases        (before segment)
    segment.run(ctx, nouns) nouns -> masks with ids
    run(ctx, segments)      photo + ids -> labels        (after segment)

SAM 3 is concept-prompted — its endpoint default prompt is literally "car" — so it
cannot segment without a noun list. But a label has to reference an `object_id`,
and ids only exist once masks do. One pass cannot satisfy both, hence two.

**What the model is asked for, and what it is not.** It picks from closed
vocabularies — a category and a material — and names which numbered object each one
rests on. It is never asked for metres or kilograms. Those are measurements, and a
model reading them off a photo is guessing; a guess dressed as a prior is worse than
no prior at all, because `E_prior` weights by `1/sigma^2` and an invented sigma is an
invented weight. Absolute scale comes from metric depth instead.

**Set-of-mark, not crops.** The second pass sees the whole photo with each mask
outlined and numbered, rather than isolated cutouts. A crop of a mug and a crop of a
bucket are the same picture; what separates them is the room around them.

**Wall- and ceiling-mounted objects are excluded at the inventory step**, which is a
scope limitation rather than an oversight. `supported_by=None` means *the floor*, so
a painting on a wall reads as an object floating 1.5 m up: the scale axis fails it,
stability drops it, and repair "corrects" it by snapping the painting onto the
carpet. Dropping them from the inventory is one prompt line against a schema change
touching five modules, and nothing in this project manipulates a wall fixture. The
cost is that a reconstructed scene is missing its pictures, and anything resting on
a wall shelf loses its support.
"""

import logging

from pydantic import BaseModel, Field

from app.pipeline import vlm
from app.pipeline.base import PipelineContext
from app.schemas import LabelResult, Material, ObjectLabel, SegmentResult

__all__ = ["LabelResult", "inventory", "run"]

log = logging.getLogger(__name__)

CATEGORIES = [
    "dining_table",
    "side_table",
    "desk",
    "chair",
    "stool",
    "sofa",
    "armchair",
    "bed",
    "cabinet",
    "bookshelf",
    "counter",
    "sink",
    "refrigerator",
    "oven",
    "microwave",
    "lamp",
    "monitor",
    "laptop",
    "keyboard",
    "mug",
    "cup",
    "bottle",
    "bowl",
    "plate",
    "pot",
    "pan",
    "book",
    "box",
    "basket",
    "vase",
    "plant",
    "pillow",
    "rug",
    "picture_frame",
    "telephone",
    "remote",
    "clock",
    "toy",
    "bag",
    "shoe",
    "towel",
    "cushion",
    "tray",
    "candle",
    "speaker",
    "fan",
    "heater",
    "other",
]

# Containers, which come out roughly 4x heavy: reconstruction returns an outer
# shell and the defining feature of a container is the hole. Recorded here so the
# error is attributable rather than mysterious. See app.pipeline.inertia.
HOLLOW_CATEGORIES = {"mug", "cup", "bottle", "bowl", "pot", "pan", "basket", "vase", "box"}

INVENTORY_PROMPT = """\
List every distinct physical object in this indoor photo, as short noun phrases \
suitable for prompting an open-vocabulary image segmenter.

One per line, lowercase, singular. Include furniture, appliances, and smaller \
objects resting on surfaces. Name each distinct instance separately even when two \
are the same kind of thing.

If two things are physically joined into one continuous piece — a plant growing \
out of its pot, water in a glass — name them together as the one object \
("potted plant", not "plant" and "pot" separately). Otherwise, name each distinct \
instance on its own, including smaller objects resting on furniture.

Exclude walls, floor, ceiling, windows, doorways, shadows and reflections.

Also exclude anything mounted on a wall or hanging from the ceiling — pictures, \
mirrors, wall shelves, mounted screens, pendant lights. Only list objects that rest \
on the floor or on another object."""

LABEL_PROMPT = """\
Each object in this photo has been outlined and labelled with a number.

For every numbered object, decide:

1. `category` — the closest match from CATEGORIES. Pick the closest even if it is \
imperfect; use "other" only when nothing is close.
2. `material` — the dominant material of its surface, from MATERIALS.
3. `resting_on` — the number of the object it is physically supported by, or null \
if it stands on the floor. Judge actual contact, not proximity: a lamp beside a \
table is on the floor, a lamp on top of it is not.
4. `notes` — one short phrase on what you saw, for debugging.

Do not estimate dimensions, volume, weight or density. Those are measured elsewhere \
from depth and geometry; your job is to pick the right bucket, not to measure.

CATEGORIES: {categories}
MATERIALS: {materials}"""


class _Inventory(BaseModel):
    objects: list[str] = Field(description="Short noun phrases, one per object instance.")


class _LabelledObject(BaseModel):
    index: int = Field(description="The number drawn on this object in the image.")
    category: str
    material: Material
    resting_on: int | None = Field(
        default=None, description="Index of the supporting object, or null for the floor."
    )
    notes: str = ""


class _Labels(BaseModel):
    objects: list[_LabelledObject]


def inventory(ctx: PipelineContext) -> list[str]:
    """Pass one: noun phrases for the segmenter.

    Deliberately unstructured beyond "a list of nouns". Constraining the vocabulary
    here would constrain what can be *found*, and the closed category list belongs
    in the second pass where it keys a density lookup — not in the first, where it
    would silently make anything unlisted invisible to the whole pipeline.
    """
    nouns = vlm.parse(
        ctx, "label", [ctx.image_path.read_bytes()], INVENTORY_PROMPT, _Inventory
    ).objects
    # Duplicates are wasted segmentation prompts; SAM 3 finds every instance of a
    # concept from one mention.
    unique = list(dict.fromkeys(n.strip().lower() for n in nouns if n.strip()))
    log.info("inventory: %d nouns — %s", len(unique), ", ".join(unique))
    return unique


def run(ctx: PipelineContext, segments: SegmentResult) -> LabelResult:
    """Pass two: category, material and support for each segmented object."""
    from app.pipeline.annotate import annotate_masks

    if not segments.masks:
        return LabelResult(labels=[])

    annotated = annotate_masks(ctx.image_path, segments)
    prompt = LABEL_PROMPT.format(
        categories=", ".join(CATEGORIES),
        materials=", ".join(m.value for m in Material),
    )
    parsed = vlm.parse(ctx, "label", [annotated], prompt, _Labels)

    # The model works in 1-based indices drawn on the image; everything downstream
    # works in object ids.
    by_index = {index + 1: mask.object_id for index, mask in enumerate(segments.masks)}

    labels = []
    for item in parsed.objects:
        object_id = by_index.get(item.index)
        if object_id is None:
            log.warning("label references index %s, which was not drawn", item.index)
            continue

        category = item.category if item.category in CATEGORIES else "other"
        if category != item.category:
            log.info("category %r not in the vocabulary; recorded as other", item.category)

        # Self-support is a model slip, not a scene: it would make a cycle with no
        # floor to settle against.
        support = by_index.get(item.resting_on) if item.resting_on is not None else None
        if support == object_id:
            log.warning("%s was reported as resting on itself; treating as floor", object_id)
            support = None

        labels.append(
            ObjectLabel(
                object_id=object_id,
                category=category,
                material=item.material,
                # No prior: see the module docstring. Scale comes from depth.
                prior=None,
                support_parent=support,
            )
        )

    missing = {m.object_id for m in segments.masks} - {label.object_id for label in labels}
    if missing:
        # An unlabelled object still reconstructs and still certifies; it just has
        # no category, so it falls back to a default density.
        log.warning("no label returned for %s", sorted(missing))
        labels.extend(
            ObjectLabel(object_id=object_id, category="other", material=Material.OTHER)
            for object_id in sorted(missing)
        )

    return LabelResult(labels=labels)
