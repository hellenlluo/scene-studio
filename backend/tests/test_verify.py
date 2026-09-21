"""Stage 5.5's duplicate check, with the model stubbed.

What is under test is the batching, not the judgement: how many calls the stage
makes, whether each parent's images are the ones its roster line points at, and
whether an answer lands on that parent's children and no one else's. The verdict
itself comes from a model and is not something a test can assert.
"""

import numpy as np
import pytest

from app.pipeline import verify
from app.pipeline.base import PipelineContext
from app.schemas import CameraPose, ObjectVerdict
from tests.fixtures.scenes import kitchen


@pytest.fixture
def ctx(tmp_path):
    import cv2

    image = tmp_path / "room.png"
    cv2.imwrite(str(image), np.full((120, 160, 3), 200, np.uint8))
    return PipelineContext.create("verify-test", image)


class _Recorder:
    """Stands in for `vlm.parse`, answering `already_included` for chosen pairs."""

    def __init__(self, included: set[tuple[int, str]] | None = None):
        self.calls: list[tuple[int, str]] = []  # (image count, prompt)
        self.included = included or set()

    def __call__(self, ctx, stage, images, prompt, schema):
        self.calls.append((len(images), prompt))
        items = [
            schema.model_fields["items"].annotation.__args__[0](
                object_number=number,
                category=category,
                already_included=True,
                reason="modelled in",
            )
            for number, category in self.included
        ]
        return schema(items=items)


def _with_parents(count: int):
    """A scene with `count` tables, each holding its own mug."""
    graph = kitchen()
    graph.objects = [obj for obj in graph.objects if obj.object_id in {"table", "mug"}]
    table, mug = graph.get("table"), graph.get("mug")
    objects = []
    for index in range(count):
        t = table.model_copy(deep=True)
        m = mug.model_copy(deep=True)
        t.object_id, m.object_id = f"table{index}", f"mug{index}"
        m.supported_by = t.object_id
        objects.extend([t, m])
    graph.objects = objects
    # `_isolated_views` orbits the photo camera, which the hand-authored fixtures
    # have no reason to carry; reconcile always sets one by the time verify runs.
    graph.camera = CameraPose(position_m=(0.0, -3.0, 1.2), forward=(0.0, 1.0, 0.0))
    return graph


def test_one_call_per_batch_not_per_parent(ctx, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(verify.vlm, "parse", recorder)

    verify._find_builtin_duplicates(ctx, _with_parents(verify.BUILTIN_BATCH))

    assert len(recorder.calls) == 1
    images, prompt = recorder.calls[0]
    # Three views per parent, all in the one request.
    assert images == 3 * verify.BUILTIN_BATCH
    assert f"images {images - 2}-{images}" in prompt


def test_batches_are_bounded(ctx, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(verify.vlm, "parse", recorder)

    # One more parent than fits, so the bound has to split rather than grow the call.
    verify._find_builtin_duplicates(ctx, _with_parents(verify.BUILTIN_BATCH + 1))

    assert len(recorder.calls) == 2
    assert [images for images, _ in recorder.calls] == [3 * verify.BUILTIN_BATCH, 3]


def test_verdict_lands_only_on_that_parents_children(ctx, monkeypatch):
    # Object 2 in the roster says its mug is built in; object 1 says nothing.
    recorder = _Recorder(included={(2, "mug")})
    monkeypatch.setattr(verify.vlm, "parse", recorder)

    found = verify._find_builtin_duplicates(ctx, _with_parents(2))

    assert len(found) == 1
    assert found[0].verdict is ObjectVerdict.DUPLICATE
    assert found[0].object_id == "mug1"
    assert found[0].duplicate_of == "table1"


def test_unknown_object_number_is_dropped(ctx, monkeypatch):
    recorder = _Recorder(included={(99, "mug")})
    monkeypatch.setattr(verify.vlm, "parse", recorder)

    assert verify._find_builtin_duplicates(ctx, _with_parents(2)) == []


def test_no_children_asks_nothing(ctx, monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(verify.vlm, "parse", recorder)

    graph = _with_parents(2)
    graph.objects = [obj for obj in graph.objects if not obj.supported_by]

    assert verify._find_builtin_duplicates(ctx, graph) == []
    assert recorder.calls == []
