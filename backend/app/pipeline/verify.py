"""Stage 5.5: does anything reconstruction produced not belong in this scene?

Segmentation and reconstruction each make their own honest mistakes — a synonym
in stage 3's inventory list segments something already covered by another mask,
a single-view model completes an occluded shape into something that never had a
photo behind it. Nothing upstream can catch either: each stage does its own job
correctly and the failure only exists in what the *combination* produced. This
stage is the check for that, run against the one thing nothing upstream is
compared to — the photo itself.

**A vision model, not a geometry heuristic.** The tempting alternative — flag two
objects whose meshes overlap in world space — was tried against this exact
failure and false: a decorative object resting *on* another (a mug on a table)
overlaps nothing, and a genuine duplicate can end up placed beside its original
rather than inside it. What actually distinguishes "this belongs" from "this
does not" is whether it is visible in the photo, which is a question about the
photo, and answering it needs something that can look at the photo.

**Two questions, asked separately, because one call could not answer both.** The
first — "is this object in the photo at all" — is a whole-scene comparison against
the photo, and works. The second — "is this object already modelled into another
object's shape" — was tried the same way three times and failed every time: shown
the whole scene and asked to find duplicates, the model matches each numbered
cushion against a real cushion in the photo, says yes, and moves on. It never
thinks to check whether the *sofa's own geometry* also contains that cushion.

Asked in isolation the same model answers correctly and immediately. Given three
views of one reconstructed sofa on a blank background and asked what detachable
items are already part of its shape, it says throw pillows; given the coffee
table, it says none. So the duplicate check renders each supporting object alone
and asks only about that object, and a child resting on a parent whose own mesh
already contains that kind of thing is the duplicate.

Isolation means *each image shows one object*, which is why several parents can
share a call: the failure above came from showing the assembled scene, not from
sharing a request. `BUILTIN_BATCH` parents go in at once against a roster naming
which images belong to which — a room where a sofa, a table and a shelf each hold
something costs one call rather than three.

**Geometry cannot answer the second question, and three attempts confirmed it.**
Mask overlap: the pillow masks and the sofa mask are adjacent, not overlapping —
segmentation cut a pillow-shaped notch out of the sofa, so the intersection is
0.1%. Mesh volume overlap: also zero, because the loose pillow is posed resting
*on* the sofa rather than inside it. Silhouette coverage in image space: dominated
by the fact that placement is still wrong at this stage, which reported a side
table as 99.8% "covered" by a sofa it is merely in front of. What actually
distinguishes the cases is semantic — whether a sofa is the kind of thing that
comes with cushions — and that needs a model that knows what a sofa is.

Runs after RECONCILE, before INERTIA: objects are posed in world space by then,
which a render needs, but nothing has been solved, certified, or paid for in
CoACD decomposition yet — a hallucinated object is cheaper to catch before any of
that runs than after.
"""

import logging

from PIL import Image
from pydantic import BaseModel, Field

from app.pipeline import snapshot, vlm
from app.pipeline.base import PipelineContext, prompt_fingerprint
from app.schemas import ObjectVerdict, ObjectVerification, SceneGraph, VerifyResult

__all__ = ["VerifyResult", "prompts_fingerprint", "run"]

log = logging.getLogger(__name__)

# Parents per duplicate-check call, at three rendered views each — so at most 12
# images in one request. The bound exists because the batch is what makes this one
# call instead of N, and an unbounded one would put a whole cluttered room's worth
# of renders in a single message; the reliability of the answer is the thing being
# economised on, not the token count.
BUILTIN_BATCH = 4

EXTERNAL_PROMPT = """\
The first image is the original photo. The next three images are a 3D \
reconstruction of the same scene: the same viewpoint as the photo, then two \
further views orbiting left and right around it. Each numbered object is the \
same physical thing across all three reconstruction images, coloured and \
outlined consistently, though it may not be visible from every angle.

The numbered objects are:

{inventory}

For every numbered object, decide whether it is `ok` — a real item genuinely \
visible somewhere in the original photo — or `external`, meaning it is not \
present in the photo at all and reconstruction invented it.

The reconstruction's shapes are rougher than the photo and its colours duller and \
flatter; that alone is never a reason to flag something. Neither is an object \
sitting at the wrong height, floating, or intersecting its neighbour — placement \
is corrected by a later physics stage and is not your concern.

**Do not use where an object sits in the reconstruction to decide where to look \
for it in the photo.** Nothing has been solved yet, so an object is routinely \
rendered on the wrong surface, in the wrong part of the room, or at the wrong \
scale. Search the whole photo for something of that shape and colour, wherever it \
may be. "There is no such thing on the table" is not a finding when the object was \
never on the table.

Every numbered object was cut from this photo by a segmenter, so the default is \
strongly that it is real. Flag `external` only for something you cannot find \
anywhere in the photo at any location. When in doubt, `ok`.

Give a one-sentence `reason` for each naming what you saw."""

BUILTIN_PROMPT = """\
These images are views of {count} reconstructed 3D objects, each one alone on a \
blank background. Every object gets three views in a row — the same object from \
three angles — and the objects follow each other in this order:

{roster}

Single-image reconstruction completes an object using what it expects that kind \
of object to have, so a reconstructed sofa often arrives with its cushions \
already modelled into it, a desk with its drawers.

For each object, and for each kind of item listed against it, say whether that \
object's own shape **already includes** one — an item a person could lift off and \
carry away, which a bare version of that object would not have. Fixed structural \
parts (legs, arms, a backrest frame, a tabletop) never count, however cushion-like \
they look.

**Answer about each object using only its own three images.** The objects here are \
unrelated to each other and were photographed separately; what one of them \
contains says nothing about any other. Do not carry an answer across objects, and \
do not compare them.

Return one entry per (object_number, category) pair asked about. If you cannot \
clearly see such an item as part of that object's shape, say it is not included."""


class _Verdict(BaseModel):
    number: int = Field(description="The number drawn on this object in the renders.")
    verdict: ObjectVerdict
    reason: str = ""


class _Verdicts(BaseModel):
    objects: list[_Verdict]


class _BuiltIn(BaseModel):
    object_number: int = Field(description="The number of the object this answer is about.")
    category: str = Field(description="One of the kinds of item asked about for that object.")
    already_included: bool
    reason: str = ""


class _BuiltIns(BaseModel):
    items: list[_BuiltIn]


def _find_external(ctx: PipelineContext, graph: SceneGraph) -> list[ObjectVerification]:
    """Whole-scene comparison against the photo: what did reconstruction invent?"""
    photo_bytes = ctx.image_path.read_bytes()
    width, height = Image.open(ctx.image_path).size
    views = snapshot.render_views(graph, ctx.settings, width / height)

    # The ids are what appear in logs, the certificate and the viewer's object
    # list, so naming them here means a verdict reads against those without a
    # second lookup. The number stays the key the model answers with: it is what
    # is actually drawn in the images.
    inventory = "\n".join(
        f"{index + 1}. {obj.object_id} — {obj.label.category}"
        for index, obj in enumerate(graph.objects)
    )
    verdicts = vlm.parse(
        ctx,
        "verify",
        [photo_bytes, views["front"], views["left"], views["right"]],
        EXTERNAL_PROMPT.format(inventory=inventory),
        _Verdicts,
    ).objects

    by_number = {index + 1: obj.object_id for index, obj in enumerate(graph.objects)}
    found = []
    for verdict in verdicts:
        object_id = by_number.get(verdict.number)
        if object_id is None:
            log.warning("verify: model referenced number %s, which was not drawn", verdict.number)
            continue
        if verdict.verdict == ObjectVerdict.EXTERNAL:
            found.append(
                ObjectVerification(
                    object_id=object_id,
                    verdict=ObjectVerdict.EXTERNAL,
                    reason=verdict.reason,
                )
            )
    return found


def _isolated_views(graph: SceneGraph, obj, settings, aspect: float) -> list[bytes]:
    """Three orbiting views of one object alone on a blank background.

    Alone is the point: the question is what *this* object's shape contains, and
    the neighbours resting on it are exactly what would confuse that.
    """
    solo = graph.model_copy(update={"objects": [obj]})
    return [
        snapshot.render(solo, settings, aspect, snapshot.orbit_camera(solo, azimuth, 20.0))
        for azimuth in (-35.0, 0.0, 35.0)
    ]


def _ask_builtin(
    ctx: PipelineContext, graph: SceneGraph, batch: list, aspect: float
) -> list[ObjectVerification]:
    """One call covering every parent in `batch`.

    Isolation is what makes this question answerable at all — see the module
    docstring — and it survives batching, because isolation is a property of each
    *image*: every one still shows a single object on a blank background. What is
    shared is the request, so the prompt has to re-establish the boundary the
    separate calls used to get for free, hence the roster and the instruction not
    to carry answers across objects.
    """
    images: list[bytes] = []
    roster_lines = []
    by_number: dict[int, tuple] = {}

    for number, (parent, resting) in enumerate(batch, start=1):
        first = len(images) + 1
        images.extend(_isolated_views(graph, parent, ctx.settings, aspect))
        categories = sorted({child.label.category for child in resting})
        roster_lines.append(
            f"  {number}. a {parent.label.category} — images {first}-{len(images)}; "
            f"items the scene places on it: {', '.join(categories)}"
        )
        by_number[number] = (parent, resting)

    answers = vlm.parse(
        ctx,
        "verify",
        images,
        BUILTIN_PROMPT.format(count=len(batch), roster="\n".join(roster_lines)),
        _BuiltIns,
    ).items

    found: list[ObjectVerification] = []
    for answer in answers:
        if not answer.already_included:
            continue
        entry = by_number.get(answer.object_number)
        if entry is None:
            log.warning(
                "verify: model referenced object %s, which was not shown", answer.object_number
            )
            continue
        parent, resting = entry
        # Every child of that category goes, not one of them. Reconstruction
        # completes the parent from the same photo, so if it modelled in a
        # cushion at all it modelled in the ones the photo showed — and there
        # is no correspondence to say which loose copy matches which bump.
        for child in resting:
            if child.label.category != answer.category:
                continue
            reason = (
                f"already part of {parent.object_id} ({parent.label.category}): {answer.reason}"
            )
            found.append(
                ObjectVerification(
                    object_id=child.object_id,
                    verdict=ObjectVerdict.DUPLICATE,
                    duplicate_of=parent.object_id,
                    reason=reason,
                )
            )
    return found


def _find_builtin_duplicates(ctx: PipelineContext, graph: SceneGraph) -> list[ObjectVerification]:
    """Children whose supporting object already has that kind of thing modelled in.

    Parents are batched rather than asked one per call: a room where a sofa, a
    table and a shelf each hold something used to cost three calls on top of the
    whole-scene check, and the question about one parent does not depend on the
    answer for any other.
    """
    children: dict[str, list] = {}
    for obj in graph.objects:
        if obj.supported_by:
            children.setdefault(obj.supported_by, []).append(obj)
    if not children:
        return []

    batchable = [
        (parent, resting)
        for parent_id, resting in children.items()
        if (parent := graph.get(parent_id)) is not None
    ]
    if not batchable:
        return []

    width, height = Image.open(ctx.image_path).size
    aspect = width / height

    found: list[ObjectVerification] = []
    for start in range(0, len(batchable), BUILTIN_BATCH):
        found.extend(_ask_builtin(ctx, graph, batchable[start : start + BUILTIN_BATCH], aspect))
    return found


def run(ctx: PipelineContext, graph: SceneGraph) -> VerifyResult:
    if not graph.objects:
        return VerifyResult(graph=graph, removed=[])

    removed = _find_external(ctx, graph) + _find_builtin_duplicates(ctx, graph)
    # An object can be reported by both checks; the first verdict wins, and the
    # order above makes that `external`, which is the stronger statement.
    seen: set[str] = set()
    unique: list[ObjectVerification] = []
    for entry in removed:
        if entry.object_id not in seen:
            seen.add(entry.object_id)
            unique.append(entry)
            log.info("verify: dropping %s (%s) — %s", entry.object_id, entry.verdict, entry.reason)

    if not seen:
        return VerifyResult(graph=graph, removed=[])

    working = graph.model_copy(deep=True)
    working.objects = [obj for obj in working.objects if obj.object_id not in seen]
    # A support parent that was just removed is a dangling reference; the floor is
    # the honest fallback, same rule `reconcile` applies when reconstruction drops
    # an object outright.
    for obj in working.objects:
        if obj.supported_by in seen:
            obj.supported_by = None

    return VerifyResult(graph=working, removed=unique)


def prompts_fingerprint() -> str:
    """Cache-key input, so editing either prompt re-runs this stage."""
    return prompt_fingerprint(EXTERNAL_PROMPT, BUILTIN_PROMPT)
