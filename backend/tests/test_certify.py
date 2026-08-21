"""The certificate aggregator.

What these mostly pin is the *reporting* semantics rather than the physics: which
axis status means what, and which combinations are allowed to read as certified.
The individual axes are tested in their own files.
"""

import pytest

from app.certify import certify
from app.pipeline.base import PipelineContext
from app.schemas import AxisStatus
from tests.fixtures import scenes


@pytest.fixture
def ctx(tmp_path):
    image = tmp_path / "room.png"
    image.write_bytes(b"only the bytes are hashed")
    return PipelineContext.create("certify-test", image)


# --- the four statuses --------------------------------------------------------


def test_a_sound_scene_passes_every_axis(ctx):
    cert = certify.run(ctx, scenes.kitchen())
    assert cert.scale_status is AxisStatus.PASS
    assert cert.stability_status is AxisStatus.PASS
    assert cert.inertial_status is AxisStatus.PASS
    assert cert.cost_status is AxisStatus.PASS
    assert not cert.failing_object_ids()


def test_a_sound_scene_certifies(ctx):
    """All four axes have validators now, so this is the first configuration that
    can honestly report as certified."""
    cert = certify.run(ctx, scenes.kitchen())
    assert cert.unchecked_axes == []
    assert cert.passed
    assert not cert.failing_object_ids()


def test_every_axis_is_reached(ctx):
    """A NOT_RUN from here on means a validator raised or was skipped, not that
    the check does not exist."""
    cert = certify.run(ctx, scenes.kitchen())
    assert all(s is not AxisStatus.NOT_RUN for s in cert.axes.values())


def test_a_broken_scene_fails_only_the_axes_it_broke(ctx):
    cert = certify.run(ctx, scenes.mug_sunk_into_table())
    assert cert.scale_status is AxisStatus.FAIL
    assert cert.stability_status is AxisStatus.FAIL
    # One failure does not poison the report.
    assert cert.inertial_status is AxisStatus.PASS
    assert cert.cost_status is AxisStatus.PASS
    assert not cert.passed


def test_the_two_physical_axes_disagree_about_blame(ctx):
    """Worth knowing before it looks like a bug. Stability measures overlap
    symmetrically, so a mug buried in a table fails both bodies; the scale axis
    checks each object against its own support, so it names only the mug. The
    union is what `failing_object_ids` reports."""
    cert = certify.run(ctx, scenes.mug_sunk_into_table())
    scale_failures = {c.object_id for c in cert.scale if not c.passed}
    stability_failures = {c.object_id for c in cert.stability if not c.passed}
    assert scale_failures == {"mug"}
    assert stability_failures == {"mug", "table"}
    assert cert.failing_object_ids() == {"mug", "table"}


# --- provenance of the thresholds ---------------------------------------------


def test_the_tolerance_is_recorded_on_the_certificate(ctx):
    """So a stored result stays interpretable after the setting moves, and so the
    sensitivity of the results to delta can be reported."""
    cert = certify.run(ctx, scenes.kitchen())
    assert cert.penetration_tolerance_m == ctx.settings.max_penetration_m


def test_every_axis_carries_its_measurements(ctx):
    """A bare pass/fail would throw away exactly what a user needs to fix it."""
    cert = certify.run(ctx, scenes.kitchen())
    assert len(cert.scale) == 3
    assert len(cert.stability) == 3
    assert len(cert.inertial) == 3
    assert cert.cost is not None and cert.cost.mean_step_time_ms > 0
