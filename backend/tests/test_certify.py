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


def test_a_sound_rigid_scene_passes_every_axis_that_ran(ctx):
    cert = certify.run(ctx, scenes.kitchen())
    assert cert.stability_status is AxisStatus.PASS
    assert cert.inertial_status is AxisStatus.PASS
    assert cert.cost_status is AxisStatus.PASS
    assert not cert.failing_object_ids()


def test_scale_is_not_run_so_nothing_certifies_yet(ctx):
    """app.certify.scale is still a stub. Reporting that axis as NOT_APPLICABLE
    would let a scene certify with its scale unexamined, which is the whole thing
    the per-axis contract exists to prevent."""
    cert = certify.run(ctx, scenes.kitchen())
    assert cert.scale_status is AxisStatus.NOT_RUN
    assert cert.unchecked_axes == ["scale"]
    assert not cert.passed


def test_a_broken_scene_fails_the_axis_it_broke(ctx):
    cert = certify.run(ctx, scenes.mug_sunk_into_table())
    assert cert.stability_status is AxisStatus.FAIL
    # The other axes are unaffected — one failure does not poison the report.
    assert cert.inertial_status is AxisStatus.PASS
    assert cert.cost_status is AxisStatus.PASS
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
    assert len(cert.stability) == 3
    assert len(cert.inertial) == 3
    assert cert.cost is not None and cert.cost.mean_step_time_ms > 0
