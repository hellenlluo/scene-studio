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
    HypothesisSource,
    InertialProperties,
    Joint,
    JointHypothesis,
    JointLimits,
    JointType,
    ObjectLabel,
    PartGeometry,
    Provenance,
    Route,
    SceneObject,
    StabilityCheck,
    SweepCheck,
)


def _part(part_id: str = "body") -> PartGeometry:
    return PartGeometry(part_id=part_id, name=part_id, dims_m=(0.6, 0.9, 0.6))


def _label(object_id: str = "obj0", route: Route = Route.RIGID) -> ObjectLabel:
    return ObjectLabel(
        object_id=object_id,
        category="cabinet",
        route=route,
        prior=DimensionPrior(dims_m=(0.6, 0.9, 0.6), sigma_m=(0.02, 0.05, 0.02)),
    )


def _obj(**kwargs) -> SceneObject:
    # frame is required rather than defaulted: stage 5 exists to record how each
    # branch's asset was brought into the canonical frame, and an object that
    # cannot say where it came from is one the reconciliation cannot be audited on.
    return SceneObject(
        object_id="obj0",
        label=_label(),
        frame=AssetFrame(source="test"),
        **{"parts": [_part()], **kwargs},
    )


def _joint(
    joint_id: str = "j0",
    type: JointType = JointType.REVOLUTE,
    parent_part_id: str = "body",
    child_part_id: str = "door",
    **kwargs,
) -> Joint:
    return Joint(
        joint_id=joint_id,
        name="door_left",
        type=type,
        parent_part_id=parent_part_id,
        child_part_id=child_part_id,
        axis=(0.0, 0.0, 1.0),
        origin_m=(0.3, 0.0, 0.0),
        limits=JointLimits(lower=0.0, upper=1.57),
        **kwargs,
    )


# --- a rigid object is an articulated object with one part and no joints ------


def test_rigid_object_is_the_degenerate_articulated_case():
    obj = _obj()
    assert obj.parts and not obj.joints
    assert not obj.is_articulated


def _two_part_kwargs() -> dict:
    return {
        "parts": [
            _part("body"),
            PartGeometry(
                part_id="door", name="door", parent_part_id="body", dims_m=(0.6, 0.02, 0.9)
            ),
        ]
    }


def test_object_whose_joints_all_failed_to_fixed_is_not_articulated():
    """The articulated branch coming back with nothing degrades to the rigid
    result rather than to a broken one — that is what makes biasing the router
    toward articulated cheap."""
    obj = _obj(joints=[_joint(type=JointType.FIXED)], **_two_part_kwargs())
    assert not obj.is_articulated


# --- structural invariants ----------------------------------------------------


def test_a_joint_naming_a_nonexistent_part_is_rejected():
    """Otherwise it surfaces as a KeyError deep inside MJCF emission, with a
    message that says nothing about which object was malformed."""
    with pytest.raises(ValidationError, match="does not exist"):
        _obj(joints=[_joint()])


def test_a_joint_cannot_drive_the_root_part():
    """The root body carries the object pose, so a joint on it would move the whole
    object instead of articulating it."""
    with pytest.raises(ValidationError, match="cannot drive the root part"):
        _obj(
            joints=[_joint(parent_part_id="door", child_part_id="body")],
            **_two_part_kwargs(),
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


# --- degradation --------------------------------------------------------------


def test_weld_makes_an_object_rigid_and_records_why():
    """A single OBB box cannot articulate — a joint needs two parts — so the
    emergency rung degrades to rigid structurally rather than by policy. What
    `weld` adds is that the loss is recorded instead of silent."""
    obj = _obj(joints=[_joint()], **_two_part_kwargs())
    assert obj.is_articulated

    obj.weld("articulated branch returned no usable joints")

    assert not obj.is_articulated
    assert obj.route_taken is Route.RIGID
    assert obj.degradation_reason
    assert obj.joints[0].provenance["type"] is Provenance.FALLBACK
    assert all(p.welded for p in obj.parts if p.parent_part_id)


def test_root_part_is_found_by_hierarchy_not_list_order():
    """Nothing orders `parts`, and a URDF branch is free to emit a door before the
    body it hangs on."""
    obj = _obj(
        scale=2.0,
        parts=[
            PartGeometry(
                part_id="door", name="door", parent_part_id="body", dims_m=(0.1, 0.1, 0.1)
            ),
            PartGeometry(part_id="body", name="body", dims_m=(1.0, 2.0, 1.0)),
        ],
    )
    assert obj.root_part.part_id == "body"
    assert obj.dims_m == (2.0, 4.0, 2.0)


def test_root_part_falls_back_to_first_for_a_flat_list():
    assert _obj().root_part.part_id == "body"


# --- scale is per object ------------------------------------------------------


def test_scale_is_isotropic_and_per_object():
    obj = _obj(scale=2.0)
    assert obj.dims_m == (1.2, 1.8, 1.2)


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


# --- joint hypotheses ---------------------------------------------------------


def _hyp(score: float) -> JointHypothesis:
    return JointHypothesis(
        type=JointType.REVOLUTE,
        parent_part_id="body",
        child_part_id="door",
        axis=(0.0, 0.0, 1.0),
        origin_m=(0.3, 0.0, 0.0),
        limits=JointLimits(lower=0.0, upper=1.57),
        score=score,
        source=HypothesisSource.BRANCH,
    )


def test_score_margin_needs_two_candidates():
    assert _joint(hypotheses=[_hyp(0.9)]).score_margin is None
    assert _joint(hypotheses=[]).score_margin is None


def test_score_margin_is_top1_minus_top2():
    joint = _joint(hypotheses=[_hyp(0.9), _hyp(0.4), _hyp(0.1)])
    assert joint.score_margin == 0.5


# --- certification ------------------------------------------------------------


def _stability(object_id: str, passed: bool) -> StabilityCheck:
    return StabilityCheck(
        object_id=object_id,
        com_displacement_m=0.0 if passed else 0.05,
        orientation_drift_deg=0.0,
        initial_penetration_m=0.0,
        passed=passed,
    )


def _sweep(joint_id: str, passed: bool) -> SweepCheck:
    return SweepCheck(
        joint_id=joint_id,
        object_id="obj0",
        steps=50,
        max_parent_penetration_m=0.0 if passed else 0.01,
        passed=passed,
    )


def test_untested_axis_is_not_applicable_not_pass():
    """A scene of jointless objects has not passed the kinematic axis; it has
    not been tested on it."""
    cert = Certificate()
    assert cert.kinematic_status is AxisStatus.NOT_APPLICABLE


def test_a_certificate_that_tested_nothing_is_not_certified():
    """Otherwise a default-constructed certificate reports as certified, which is
    exactly the dishonesty the per-axis contract exists to prevent."""
    assert not Certificate().passed


def test_a_jointless_scene_still_certifies_on_the_other_four_axes():
    cert = Certificate(
        scale_status=AxisStatus.PASS,
        stability_status=AxisStatus.PASS,
        kinematic_status=AxisStatus.NOT_APPLICABLE,
        inertial_status=AxisStatus.PASS,
        cost_status=AxisStatus.PASS,
    )
    assert cert.passed


def test_a_single_failing_axis_fails_the_certificate():
    cert = Certificate(stability_status=AxisStatus.FAIL, scale_status=AxisStatus.PASS)
    assert not cert.passed


def test_jointless_objects_are_excluded_from_kinematic_pass_rate():
    """A cabinet welded shut is trivially collision-free. Crediting it would make
    the pass rate a lie, so it is reported separately instead."""
    cert = Certificate(
        sweeps=[_sweep("j0", True), _sweep("j1", False)],
        jointless_objects=["obj7", "obj8"],
    )
    assert cert.kinematic_pass_rate == 0.5
    assert cert.jointless_objects == ["obj7", "obj8"]


def test_failing_objects_gather_across_every_axis():
    cert = Certificate(
        stability=[_stability("obj0", False), _stability("obj1", True)],
        sweeps=[_sweep("j0", False)],
    )
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


# --- routing ------------------------------------------------------------------


def test_routing_defaults_to_articulated():
    """Routing errors are asymmetric: a misrouted cabinet loses its articulation
    unrecoverably, a misrouted armchair costs one wasted model call."""
    label = ObjectLabel(
        object_id="obj0",
        category="whatever",
        prior=DimensionPrior(dims_m=(1.0, 1.0, 1.0), sigma_m=(0.1, 0.1, 0.1)),
    )
    assert label.route is Route.ARTICULATED


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
    reweighted = cache_key(ctx, StageName.SOLVE, graph, SolveWeights(sweep=99.0), [])

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
