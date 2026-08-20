"""The stage contract's invariants, as opposed to its field list.

Each test here pins a claim the design rests on, so that a later refactor that
quietly breaks one fails loudly instead of producing a scene that certifies
dishonestly.
"""

import pytest
from pydantic import ValidationError

from app.schemas import (
    AssetFrame,
    AxisStatus,
    Certificate,
    DimensionPrior,
    InertialProperties,
    ObjectLabel,
    PartGeometry,
    Provenance,
    SceneObject,
    StabilityCheck,
)


def _part(part_id: str = "body") -> PartGeometry:
    return PartGeometry(part_id=part_id, name=part_id, dims_m=(0.6, 0.9, 0.6))


def _label(object_id: str = "obj0") -> ObjectLabel:
    return ObjectLabel(
        object_id=object_id,
        category="cabinet",
        prior=DimensionPrior(dims_m=(0.6, 0.9, 0.6), sigma_m=(0.02, 0.05, 0.02)),
    )


def _obj(**kwargs) -> SceneObject:
    # frame is required rather than defaulted: reconcile exists to record how each
    # asset was brought into the canonical frame, and an object that cannot say
    # where it came from is one the reconciliation cannot be audited on.
    return SceneObject(
        object_id="obj0",
        label=_label(),
        frame=AssetFrame(source="test"),
        **{"parts": [_part()], **kwargs},
    )


# --- scale is per object ------------------------------------------------------


def test_scale_is_isotropic_and_per_object():
    assert _obj(scale=2.0).dims_m == (1.2, 1.8, 1.2)


def test_prior_sigma_is_per_axis():
    """A scalar confidence cannot express that a door is pinned in width and
    height and nearly free in depth, and E_prior weights by 1/sigma^2."""
    prior = _label().prior
    assert len(prior.sigma_m) == 3
    assert prior.sigma_m[1] > prior.sigma_m[0]


# --- pinning ------------------------------------------------------------------


def test_user_edits_pin_and_other_provenances_do_not():
    obj = _obj(provenance={"scale": Provenance.USER, "position_m": Provenance.DERIVED})
    assert obj.is_pinned("scale")
    assert not obj.is_pinned("position_m")
    assert not obj.is_pinned("orientation")
    assert obj.pinned_fields() == {"scale"}


# --- structural invariants ----------------------------------------------------


def test_root_part_is_found_by_hierarchy_not_list_order():
    """Nothing orders `parts`, and a reconstruction is free to emit a shade before
    the base it sits on."""
    obj = _obj(
        scale=2.0,
        parts=[
            PartGeometry(part_id="shade", name="shade", parent_part_id="body", dims_m=(0.1,) * 3),
            PartGeometry(part_id="body", name="body", dims_m=(1.0, 2.0, 1.0)),
        ],
    )
    assert obj.root_part.part_id == "body"
    assert obj.dims_m == (2.0, 4.0, 2.0)


def test_root_part_falls_back_to_first_for_a_flat_list():
    assert _obj().root_part.part_id == "body"


def test_a_part_naming_a_nonexistent_parent_is_rejected():
    """Otherwise it surfaces as a KeyError deep inside MJCF emission, with a
    message that says nothing about which object was malformed."""
    with pytest.raises(ValidationError, match="unknown parent"):
        _obj(
            parts=[
                _part("body"),
                PartGeometry(part_id="x", name="x", parent_part_id="ghost", dims_m=(1, 1, 1)),
            ]
        )


def test_a_cycle_in_the_part_hierarchy_is_rejected():
    """The cycle sits alongside a valid root, so the root check passes and the
    walk-to-root is what has to catch it — otherwise it hangs instead of raising."""
    with pytest.raises(ValidationError, match="cycle"):
        _obj(
            parts=[
                _part("body"),
                PartGeometry(part_id="a", name="a", parent_part_id="b", dims_m=(1, 1, 1)),
                PartGeometry(part_id="b", name="b", parent_part_id="a", dims_m=(1, 1, 1)),
            ]
        )


def test_exactly_one_root_part_is_required():
    with pytest.raises(ValidationError, match="exactly one root part"):
        _obj(parts=[_part("a"), _part("b")])


def test_duplicate_part_ids_are_rejected():
    with pytest.raises(ValidationError, match="duplicate part ids"):
        _obj(parts=[_part("body"), _part("body")])


def test_an_object_needs_at_least_one_part():
    with pytest.raises(ValidationError, match="at least one part"):
        _obj(parts=[])


# --- certification ------------------------------------------------------------


def _stability(object_id: str, passed: bool) -> StabilityCheck:
    return StabilityCheck(
        object_id=object_id,
        com_displacement_m=0.0 if passed else 0.05,
        orientation_drift_deg=0.0,
        initial_penetration_m=0.0,
        passed=passed,
    )


def test_axes_default_to_not_run():
    """Not NOT_APPLICABLE: a fresh certificate has not established that there was
    nothing to check, only that nothing checked."""
    cert = Certificate()
    assert all(s is AxisStatus.NOT_RUN for s in cert.axes.values())
    assert cert.unchecked_axes == ["scale", "stability", "inertial", "cost"]


def test_a_certificate_that_tested_nothing_is_not_certified():
    """Otherwise a default-constructed certificate reports as certified, which is
    exactly the dishonesty the per-axis contract exists to prevent."""
    assert not Certificate().passed


def test_not_run_disqualifies_even_when_everything_else_passes():
    """Otherwise a scene certifies on the strength of whichever validators happen
    to be wired up."""
    cert = Certificate(
        scale_status=AxisStatus.NOT_RUN,
        stability_status=AxisStatus.PASS,
        inertial_status=AxisStatus.PASS,
        cost_status=AxisStatus.PASS,
    )
    assert not cert.passed
    assert cert.unchecked_axes == ["scale"]


def test_a_single_failing_axis_fails_the_certificate():
    cert = Certificate(
        scale_status=AxisStatus.PASS,
        stability_status=AxisStatus.FAIL,
        inertial_status=AxisStatus.PASS,
        cost_status=AxisStatus.PASS,
    )
    assert not cert.passed


def test_a_fully_checked_sound_scene_certifies():
    cert = Certificate(
        scale_status=AxisStatus.PASS,
        stability_status=AxisStatus.PASS,
        inertial_status=AxisStatus.PASS,
        cost_status=AxisStatus.PASS,
    )
    assert cert.passed
    assert cert.unchecked_axes == []


def test_a_pass_rate_over_nothing_is_none_not_one():
    """1.0 would put "100% pass" in a results table for a scene where nothing was
    ever checked."""
    assert Certificate().stability_pass_rate is None
    assert Certificate().scale_pass_rate is None


def test_failing_objects_gather_across_every_axis():
    cert = Certificate(stability=[_stability("obj0", False), _stability("obj1", True)])
    assert cert.failing_object_ids() == {"obj0"}


def test_penetration_tolerance_defaults_to_two_millimetres():
    assert Certificate().penetration_tolerance_m == 0.002


# --- inertial sanity ----------------------------------------------------------


def _inertial(diag) -> InertialProperties:
    return InertialProperties(
        density_kg_m3=600.0, volume_m3=0.1, mass_kg=60.0, inertia_diag=diag, watertight=True
    )


def test_inertia_triangle_inequality():
    assert _inertial((1.0, 1.0, 1.0)).satisfies_triangle_inequality
    assert not _inertial((1.0, 1.0, 5.0)).satisfies_triangle_inequality


def test_inertia_positive_definite():
    assert _inertial((1.0, 2.0, 2.5)).is_positive_definite
    assert not _inertial((0.0, 2.0, 2.0)).is_positive_definite


def test_mesh_is_assumed_not_watertight_until_proven():
    """trimesh hands you a non-watertight mesh and a meaningless volume without
    complaining, so the safe default has to be the pessimistic one."""
    assert not InertialProperties(
        density_kg_m3=600.0, volume_m3=0.1, mass_kg=60.0, inertia_diag=(1.0, 1.0, 1.0)
    ).watertight


# --- caching ------------------------------------------------------------------


def test_cache_key_covers_anchors_and_weights(tmp_path):
    """Keying solve on the graph alone means adding a scale anchor silently returns
    the artifact from the un-anchored run — the flagship interaction failing in the
    least visible way possible, by looking like it worked."""
    from app.pipeline.base import PipelineContext, cache_key
    from app.schemas import ScaleAnchor, SceneGraph, SolveWeights, StageName

    image = tmp_path / "room.png"
    image.write_bytes(b"not really a png, only its bytes are hashed")
    ctx = PipelineContext.create("job", image)

    graph = SceneGraph(objects=[])
    weights = SolveWeights()
    anchor = ScaleAnchor(object_id="obj0", axis=0, value_m=1.83)

    bare = cache_key(ctx, StageName.SOLVE, graph, weights, [])
    anchored = cache_key(ctx, StageName.SOLVE, graph, weights, [anchor])
    reweighted = cache_key(ctx, StageName.SOLVE, graph, SolveWeights(penetration=99.0), [])

    assert bare != anchored
    assert bare != reweighted
    assert bare == cache_key(ctx, StageName.SOLVE, graph, weights, [])


def test_certify_cache_key_covers_thresholds(tmp_path):
    from app.config import get_settings
    from app.pipeline.base import PipelineContext, cache_key
    from app.schemas import SceneGraph, StageName

    image = tmp_path / "room.png"
    image.write_bytes(b"bytes")
    ctx = PipelineContext.create("job", image)
    graph = SceneGraph(objects=[])

    loose = get_settings().certification_thresholds()
    tight = {**loose, "max_penetration_m": 0.0005}
    assert cache_key(ctx, StageName.CERTIFY, graph, loose) != cache_key(
        ctx, StageName.CERTIFY, graph, tight
    )
