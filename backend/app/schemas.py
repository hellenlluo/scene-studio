"""Typed contracts flowing between pipeline stages.

Every stage consumes and returns one of these, so stages are independently
testable and swappable without running the rest of the pipeline (or a GPU).

Three commitments shape this file:

* **Scale is per object, not global.** Stage 6 optimises one isotropic scale per
  object jointly with pose and joint parameters. A single scene-wide multiplier
  cannot express the couplings the approach rests on — a drawer that has to fit
  inside its cabinet constrains two objects against each other, not the scene
  against the camera.
* **A rigid object is an articulated object with one part and no joints.** The
  articulated branch is a superset of the rigid one, so both routes emit the
  same structure and every degradation path becomes a move within one type
  rather than a fall between two.
* **Certification is a five-axis contract, not a boolean.** Each axis carries
  the measurements it passed or failed on, because "simulation-ready" has to be
  checkable, and a bare pass/fail is not.

Anisotropic scale is deliberately absent: it breaks inertia and it looks wrong.
"""

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z — MuJoCo order


class StageName(StrEnum):
    SEGMENT = "segment"
    DEPTH = "depth"
    LABEL = "label"
    RIGID = "rigid"  # 4a
    ARTICULATED = "articulated"  # 4b
    RECONCILE = "reconcile"
    # Runs before SOLVE, not after CERTIFY: the solver's physics block steps
    # MuJoCo, and MuJoCo cannot settle a body with no mass. This stage assigns
    # the density, damping and friction priors; mass and the inertia tensor are
    # derived from density x scaled volume and so are recomputed whenever the
    # solver moves a scale.
    INERTIA = "inertia"
    SOLVE = "solve"
    CERTIFY = "certify"
    REPAIR = "repair"
    EXPORT = "export"


class Provenance(StrEnum):
    """Where a value came from. Reported per field, per object and per joint.

    USER is the load-bearing one: a user-pinned value is held fixed while
    everything else re-solves around it, so the solver reads this to decide
    which variables are free.
    """

    MODEL = "model"  # predicted by an upstream reconstruction model
    DERIVED = "derived"  # produced by the solver
    USER = "user"  # pinned by a user edit
    FALLBACK = "fallback"  # a degradation path produced this


class Pinnable(BaseModel):
    """Mixin for anything carrying per-field provenance."""

    provenance: dict[str, Provenance] = Field(default_factory=dict)

    def is_pinned(self, field: str) -> bool:
        return self.provenance.get(field) is Provenance.USER

    def pinned_fields(self) -> set[str]:
        return {k for k, v in self.provenance.items() if v is Provenance.USER}


# --- stage 1: segment ---------------------------------------------------------


class PartMask(BaseModel):
    part_id: str
    mask_path: str
    bbox_px: tuple[int, int, int, int]
    area_px: int


class ObjectMask(BaseModel):
    object_id: str
    mask_path: str  # single-channel PNG, image resolution
    bbox_px: tuple[int, int, int, int]
    area_px: int
    # Only populated for objects stage 3 routed to the articulated branch.
    # Fewer part masks than the VLM's expected joint inventory is a v2
    # uncertainty signal, so the count is worth keeping even when unused.
    parts: list[PartMask] = Field(default_factory=list)


class SegmentResult(BaseModel):
    masks: list[ObjectMask]
    image_size_px: tuple[int, int]


# --- stage 2: depth -----------------------------------------------------------


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class IntrinsicsSource(StrEnum):
    """Where the focal length came from, in descending order of trust.

    Three states rather than a boolean because backprojection puts lateral extent
    at (u - cx) * Z / fx: a wrong focal length leaves depth correct and scales
    width and height, which is an anisotropic error in the exact quantity stage 6
    estimates. ASSUMED means the solver should trust depth-derived width less than
    depth-derived distance; MODEL and EXIF do not carry that caveat.
    """

    MODEL = "model"  # the depth model predicted them (UniDepth, MoGe)
    EXIF = "exif"  # read off the photo
    ASSUMED = "assumed"  # a default field of view, i.e. a guess


class DepthResult(BaseModel):
    depth_path: str  # float32 .npy, metres, image resolution
    intrinsics: Intrinsics
    is_metric: bool = Field(
        description="False if the model output is only up-to-scale, in which case "
        "E_depth contributes to position but carries no size information."
    )
    intrinsics_source: IntrinsicsSource = IntrinsicsSource.ASSUMED


# --- stage 3: label and route -------------------------------------------------


class Route(StrEnum):
    RIGID = "rigid"
    ARTICULATED = "articulated"


class DimensionPrior(BaseModel):
    """Class-conditioned real-world size, with the uncertainty that makes it usable.

    E_prior is a Mahalanobis term weighted by 1/sigma^2, so the per-axis sigma is
    not decoration: it is what lets "dishwasher: 60 cm +/- 2 cm" outweigh
    "chair: 50 cm +/- 20 cm" instead of both pulling equally. A scalar
    confidence cannot express that, because the tightness is per axis — an
    interior door is pinned in width and height and nearly free in depth.
    """

    dims_m: Vec3
    sigma_m: Vec3 = Field(description="Per-axis stddev. Feeds E_prior as 1/sigma^2.")


class ObjectLabel(BaseModel):
    object_id: str
    category: str

    route: Route = Field(
        default=Route.ARTICULATED,
        description="Biased toward ARTICULATED on purpose. Routing errors are "
        "asymmetric: a misrouted cabinet loses its articulation silently and "
        "unrecoverably, while a misrouted armchair costs one wasted model call "
        "and degrades to exactly the rigid result.",
    )
    route_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    expected_joints: list[str] = Field(
        default_factory=list,
        description="Joint inventory the VLM expects, e.g. ['door_left', 'drawer_top']. "
        "Compared against the count actually fitted.",
    )

    prior: DimensionPrior
    support_parent: str | None = Field(
        default=None, description="object_id of the supporting body, or None for the floor."
    )


class LabelResult(BaseModel):
    labels: list[ObjectLabel]


# --- stages 4a / 4b: the two reconstruction branches --------------------------


class ProxyTier(StrEnum):
    """Collision-geometry fidelity, and implicitly whether a mesh exists at all.

    The floor is a whole-object mesh, not a box. OBB is the emergency rung — it
    means reconstruction produced nothing, so the object is one part and
    therefore, structurally, cannot articulate.

    OBB and CONVEX_HULL are both *solid and convex*, so neither can represent a
    hollow container: a drawer inside an OBB carcass interpenetrates it at every
    joint value even when the joint is perfect. Hollowness is a non-convex
    property, so an articulated container needs DECOMPOSED — or a carcass built
    from separate panel parts, which is the same thing expressed through the part
    hierarchy instead of through convex pieces.
    """

    OBB = "obb"  # no mesh; reconstruction failed
    CONVEX_HULL = "convex_hull"
    DECOMPOSED = "decomposed"


class Axis(StrEnum):
    X_POS = "+x"
    X_NEG = "-x"
    Y_POS = "+y"
    Y_NEG = "-y"
    Z_POS = "+z"
    Z_NEG = "-z"


class AssetFrame(BaseModel):
    """How a branch's raw output was brought into the canonical frame.

    Mesh models and URDF models emit assets in different up-axes, different
    front-faces, and different scale conventions — most URDF models normalise to
    a unit box. Recording the transform that was stripped is what lets stage 6
    optimise one scale variable per object instead of two incompatible ones, and
    keeps the reconciliation auditable when an object comes out sideways.
    """

    source: str = Field(description="Model that produced the asset, e.g. 'trellis', 'spark'.")
    up_axis: Axis = Axis.Z_POS
    front_axis: Axis = Axis.Y_NEG
    rotation: Quat = (1.0, 0.0, 0.0, 0.0)
    normalization_scale: float = Field(
        default=1.0,
        description="The branch's own internal normalisation, divided out here so it "
        "never gets confused with the physical scale the solver estimates.",
    )


class InertialProperties(BaseModel):
    """Assigned from a class-conditioned density prior; mass and inertia derived.

    mass_kg and inertia_diag are functions of density and the *scaled* mesh
    volume, so they are recomputed whenever the solver moves a scale rather than
    being fitted independently. volume_m3 is retained because the inertial
    certification axis checks mass, density and volume against each other, and
    it cannot do that if only two of the three survive.
    """

    density_kg_m3: float
    volume_m3: float
    mass_kg: float
    com_m: Vec3 = (0.0, 0.0, 0.0)
    inertia_diag: Vec3 = Field(description="Principal moments, part frame.")
    principal_axes: Quat = Field(
        default=(1.0, 0.0, 0.0, 0.0), description="Part frame -> principal frame (MuJoCo iquat)."
    )
    watertight: bool = Field(
        default=False,
        description="False means volume, and therefore mass, is not trustworthy — "
        "trimesh will hand you a non-watertight mesh and a meaningless volume "
        "without complaining.",
    )

    @property
    def satisfies_triangle_inequality(self) -> bool:
        a, b, c = sorted(self.inertia_diag)
        return a + b >= c

    @property
    def is_positive_definite(self) -> bool:
        return all(i > 0.0 for i in self.inertia_diag)


class PartGeometry(BaseModel):
    part_id: str
    name: str = Field(description="Semantic role, e.g. 'body', 'door_left', 'drawer_top'.")
    parent_part_id: str | None = None

    visual_mesh_path: str | None = None  # None when the OBB fallback is in use
    collision_mesh_paths: list[str] = Field(default_factory=list)
    proxy_tier: ProxyTier = ProxyTier.OBB

    dims_m: Vec3 = Field(description="Extent at unit object scale; multiply by SceneObject.scale.")
    origin_m: Vec3 = (0.0, 0.0, 0.0)  # part frame origin, object canonical frame

    inertial: InertialProperties | None = None  # populated by stage 9

    welded: bool = Field(
        default=False,
        description="Collapsed into its parent as rigid geometry after a part or "
        "joint failure. The scene stays exportable; only actuability is lost.",
    )
    visible_surface_fraction: float | None = Field(
        default=None,
        description="Observed fraction of the part's extent. Single-view depth sees "
        "the front shell only, so the rest is amodal inference. v2 uncertainty signal.",
    )


class JointType(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class JointLimits(BaseModel):
    """Radians for revolute, metres for prismatic."""

    lower: float
    upper: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


class HypothesisSource(StrEnum):
    BRANCH = "branch"  # 4b: SPARK, Articulate-Anything, URDF-Anything
    GEOMETRY = "geometry"  # stage 5 fallback, derived from part extents and gaps
    PRIOR = "prior"  # manufacturing convention for the category
    USER_DRAG = "user_drag"  # fitted from a drag in the viewer


class JointHypothesis(BaseModel):
    """One candidate articulation. The set is what user drags choose between.

    A sloppy 200 ms drag through 15 degrees is ample to disambiguate three
    discrete candidates and nowhere near enough for unconstrained 6-DoF fitting,
    so keeping the candidate set around is what makes the interaction work.
    """

    type: JointType
    parent_part_id: str
    child_part_id: str
    axis: Vec3 = Field(description="Unit direction, object canonical frame.")
    origin_m: Vec3 = Field(description="Pivot for revolute, a point on the slide for prismatic.")
    limits: JointLimits
    score: float = Field(description="Higher is better. Only the ranking is meaningful.")
    source: HypothesisSource = HypothesisSource.BRANCH


class JointDynamics(BaseModel):
    """Assigned in stage 9. Actuation behaviour depends on these more than on mesh fidelity."""

    damping: float = 0.0
    friction_loss: float = 0.0
    armature: float = 0.0


class Joint(Pinnable):
    joint_id: str
    name: str

    # The fitted values. Not an index into `hypotheses` — the solver refines
    # axis, origin and limits continuously from the winning candidate.
    type: JointType
    parent_part_id: str
    child_part_id: str
    axis: Vec3
    origin_m: Vec3
    limits: JointLimits

    dynamics: JointDynamics = Field(default_factory=JointDynamics)
    hypotheses: list[JointHypothesis] = Field(
        default_factory=list, description="Ranked, best first."
    )

    @property
    def score_margin(self) -> float | None:
        """Top-1 minus top-2 hypothesis score. A thin margin is a v2 amber signal."""
        if len(self.hypotheses) < 2:
            return None
        return self.hypotheses[0].score - self.hypotheses[1].score


class ObjectAssets(BaseModel):
    """Output of either branch, in the branch's own frame and units.

    One type for both branches, because the rigid case is the degenerate one:
    a single part and no joint hypotheses. That keeps the superset claim
    structural rather than aspirational.
    """

    object_id: str
    frame: AssetFrame
    parts: list[PartGeometry]
    joint_hypotheses: list[JointHypothesis] = Field(default_factory=list)

    failed: bool = False
    failure_reason: str | None = None


class ReconstructionResult(BaseModel):
    objects: list[ObjectAssets]


# --- stage 5: reconcile -------------------------------------------------------


class SceneObject(Pinnable):
    """One object in the canonical, gravity-aligned scene graph.

    scale, position_m, orientation and the joint parameters are exactly the
    stage-6 decision variables. Provenance decides which of them are free.
    """

    object_id: str
    label: ObjectLabel
    frame: AssetFrame

    parts: list[PartGeometry]
    joints: list[Joint] = Field(default_factory=list)

    # --- solve variables ---
    scale: float = Field(default=1.0, gt=0.0, description="Isotropic. The s_i of stage 6.")
    position_m: Vec3 = (0.0, 0.0, 0.0)
    orientation: Quat = (1.0, 0.0, 0.0, 0.0)

    supported_by: str | None = Field(
        default=None, description="object_id of the supporting body, or None for the floor."
    )

    # A misrouted object that fell back is a routing error we can count, but only
    # if both the request and the outcome are on the record.
    route_taken: Route = Route.RIGID
    degradation_reason: str | None = Field(
        default=None,
        description="Why this object lost its articulation, when it did. Shown in "
        "the viewer: 'could not be articulated' is a different message to the user "
        "than 'articulates but fails certification', and conflating them hides "
        "which objects are worth re-routing by hand.",
    )

    @property
    def is_articulated(self) -> bool:
        return any(j.type is not JointType.FIXED for j in self.joints)

    @model_validator(mode="after")
    def _check_structure(self) -> "SceneObject":
        """Referential integrity of the part tree and its joints.

        None of this is defensive programming: a joint naming a part that does not
        exist currently sails through reconcile and surfaces as a KeyError deep
        inside MJCF emission or the kinematic sweep, where the message says
        nothing about which object was malformed.
        """
        if not self.parts:
            raise ValueError(f"{self.object_id}: an object needs at least one part")

        ids = [p.part_id for p in self.parts]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.object_id}: duplicate part ids")
        known = set(ids)

        roots = [p for p in self.parts if p.parent_part_id is None]
        if len(roots) != 1:
            raise ValueError(
                f"{self.object_id}: expected exactly one root part, found {len(roots)}"
            )

        for part in self.parts:
            if part.parent_part_id is not None and part.parent_part_id not in known:
                raise ValueError(
                    f"{self.object_id}: part {part.part_id!r} names unknown parent "
                    f"{part.parent_part_id!r}"
                )

        # Walk to the root from every part. Bounded by the part count, so a cycle
        # shows up as failure to arrive rather than as a hang.
        parents = {p.part_id: p.parent_part_id for p in self.parts}
        for start in ids:
            node, steps = start, 0
            while node is not None and steps <= len(ids):
                node, steps = parents[node], steps + 1
            if node is not None:
                raise ValueError(f"{self.object_id}: cycle in the part hierarchy at {start!r}")

        root_id = roots[0].part_id
        for joint in self.joints:
            for role, part_id in (("parent", joint.parent_part_id), ("child", joint.child_part_id)):
                if part_id not in known:
                    raise ValueError(
                        f"{self.object_id}/{joint.joint_id}: {role} part {part_id!r} does not exist"
                    )
            if joint.child_part_id == root_id:
                # The root body carries the object pose; a joint driving it would
                # move the whole object rather than articulate it.
                raise ValueError(f"{self.object_id}/{joint.joint_id}: cannot drive the root part")
            if joint.child_part_id == joint.parent_part_id:
                raise ValueError(f"{self.object_id}/{joint.joint_id}: joins a part to itself")

        return self

    def weld(self, reason: str) -> "SceneObject":
        """Collapse to a rigid object, keeping the geometry and losing the motion.

        The degradation path when the articulated branch fails or reconstruction
        falls back to a single box. Deliberately explicit rather than something the
        model coerces silently: an object that quietly stopped articulating is
        exactly the failure the router's bias exists to avoid, so the decision gets
        recorded as FALLBACK provenance and shows up in the certificate as
        jointless rather than as a kinematic failure.
        """
        for joint in self.joints:
            joint.type = JointType.FIXED
            joint.provenance["type"] = Provenance.FALLBACK
        for part in self.parts:
            if part.parent_part_id is not None:
                part.welded = True
        self.provenance["joints"] = Provenance.FALLBACK
        self.label.route = Route.RIGID
        self.route_taken = Route.RIGID
        self.degradation_reason = reason
        return self

    @property
    def root_part(self) -> PartGeometry:
        """The part with no parent. Falls back to the first only for a flat list.

        Not `parts[0]`: nothing orders this list, and a URDF branch is free to
        emit a door before the body it hangs on.
        """
        return next((p for p in self.parts if p.parent_part_id is None), self.parts[0])

    @property
    def dims_m(self) -> Vec3:
        """Root part extent at the current scale."""
        w, h, d = self.root_part.dims_m
        return (w * self.scale, h * self.scale, d * self.scale)


class SceneGraph(BaseModel):
    objects: list[SceneObject]
    floor_height_m: float = 0.0
    gravity_rotation: Quat = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="Rotation applied in stage 5 to bring the fitted floor plane "
        "world-down. Everything after this assumes gravity along -Z.",
    )

    def get(self, object_id: str) -> SceneObject | None:
        return next((o for o in self.objects if o.object_id == object_id), None)

    @property
    def joint_count(self) -> int:
        return sum(len(o.joints) for o in self.objects)


# --- stage 6: solve -----------------------------------------------------------


class SolveWeights(BaseModel):
    """The lambdas of stage 6's objective."""

    depth: float = 1.0
    prior: float = 1.0
    silhouette: float = 0.5
    support: float = 2.0
    penetration: float = 5.0
    sweep: float = 5.0


class ScaleAnchor(BaseModel):
    """A user-supplied true dimension: E_prior with sigma -> 0, i.e. a hard constraint.

    One anchor usually settles the whole scene, which is a consequence of the
    coupled formulation rather than a coincidence — absolute scale is a single
    global degree of freedom, and support and contact constraints propagate it
    outward. Propagation reaches only as far as the constraint graph is
    connected; an object with no contacts and no support parent gains nothing.
    """

    object_id: str
    axis: int = Field(ge=0, le=2, description="0 = width, 1 = height, 2 = depth.")
    value_m: float = Field(gt=0.0)


class SolveDiagnostics(BaseModel):
    iterations: int = 0
    converged: bool = False
    residual_by_term: dict[str, float] = Field(
        default_factory=dict, description="Final E_depth, E_prior, E_sil, E_supp, E_pen, E_sweep."
    )
    sweep_samples_per_joint: int = 0

    scale_variance: dict[str, float] = Field(
        default_factory=dict,
        description="Inverse-Hessian diagonal at convergence, per object. Falls out "
        "of choosing Gauss-Newton for the smooth block; it is v2's primary "
        "confidence signal and the answer to which objects an anchor never reached.",
    )


class SolveResult(BaseModel):
    graph: SceneGraph
    diagnostics: SolveDiagnostics = Field(default_factory=SolveDiagnostics)


# --- stage 7: certify ---------------------------------------------------------


class AxisStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class ScaleCheck(BaseModel):
    object_id: str
    deviation_sigma: Vec3 = Field(description="Per-axis (fitted - prior) / sigma.")
    support_gap_m: float = Field(
        default=0.0, description="Signed gap on the support contact. Negative is penetration."
    )
    base_inside_parent: bool = True
    passed: bool = True


class StabilityCheck(BaseModel):
    object_id: str
    com_displacement_m: float
    orientation_drift_deg: float
    initial_penetration_m: float = Field(
        description="Measured at t=0. Once the solver starts pushing bodies apart the "
        "overlap is gone and the reconstruction error that caused it is unobservable."
    )
    passed: bool


class SweepCheck(BaseModel):
    """One joint driven across its full fitted range.

    Parameter accuracy and functional success are different quantities: 3 degrees
    of axis error is nothing on a 20 cm drawer and 9 cm of swept displacement at
    the edge of a 1.8 m wardrobe door. This measures the second thing.
    """

    joint_id: str
    object_id: str
    steps: int
    max_parent_penetration_m: float
    sibling_contacts: list[str] = Field(default_factory=list)
    world_contact: bool = False
    blocked_at_q: float | None = Field(
        default=None, description="First joint coordinate that violated. None if clear."
    )
    feasible_limits: JointLimits | None = Field(
        default=None,
        description="Largest sub-range that sweeps clean. Trimming to this is the "
        "minimal repair; the gap against the fitted limits is range genuinely lost.",
    )
    passed: bool = True


class InertialCheck(BaseModel):
    object_id: str
    mass_density_volume_consistent: bool
    positive_definite: bool
    triangle_inequality: bool
    passed: bool


class CostCheck(BaseModel):
    proxy_tier: ProxyTier
    mean_step_time_ms: float
    budget_ms: float
    passed: bool


class Certificate(BaseModel):
    """ "Simulation-ready" as a checkable multi-axis contract rather than a vibe."""

    scale_status: AxisStatus = AxisStatus.NOT_APPLICABLE
    stability_status: AxisStatus = AxisStatus.NOT_APPLICABLE
    kinematic_status: AxisStatus = AxisStatus.NOT_APPLICABLE
    inertial_status: AxisStatus = AxisStatus.NOT_APPLICABLE
    cost_status: AxisStatus = AxisStatus.NOT_APPLICABLE

    scale: list[ScaleCheck] = Field(default_factory=list)
    stability: list[StabilityCheck] = Field(default_factory=list)
    sweeps: list[SweepCheck] = Field(default_factory=list)
    inertial: list[InertialCheck] = Field(default_factory=list)
    cost: CostCheck | None = None

    jointless_objects: list[str] = Field(
        default_factory=list,
        description="Objects with zero fitted joints, reported separately and never "
        "counted as kinematically passing. A cabinet welded shut is trivially "
        "collision-free, and folding that into a pass rate would be dishonest.",
    )

    penetration_tolerance_m: float = Field(
        default=0.002,
        description="Populated from Settings.max_penetration_m by the certify stage; "
        "recorded on the certificate so a stored result stays interpretable after "
        "the setting changes. Non-zero because mesh discretisation produces sub-millimetre "
        "contacts on surfaces flush by design, and a zero-tolerance check would "
        "fail every well-modelled drawer. Results are reported against this value "
        "so their sensitivity to it is visible.",
    )

    @property
    def axes(self) -> dict[str, AxisStatus]:
        return {
            "scale": self.scale_status,
            "stability": self.stability_status,
            "kinematic": self.kinematic_status,
            "inertial": self.inertial_status,
            "cost": self.cost_status,
        }

    @property
    def passed(self) -> bool:
        """No axis failed, and at least one was actually evaluated.

        The second clause is not pedantry. Without it a default-constructed
        certificate — every axis NOT_APPLICABLE because nothing ran — reports as
        certified, which is precisely the dishonesty the per-axis contract exists
        to prevent. Untested is not passed.
        """
        statuses = self.axes.values()
        return all(s is not AxisStatus.FAIL for s in statuses) and any(
            s is AxisStatus.PASS for s in statuses
        )

    @property
    def stability_pass_rate(self) -> float:
        if not self.stability:
            return 1.0
        return sum(c.passed for c in self.stability) / len(self.stability)

    @property
    def kinematic_pass_rate(self) -> float:
        """Over fitted joints only. Jointless objects are excluded, not credited."""
        if not self.sweeps:
            return 1.0
        return sum(c.passed for c in self.sweeps) / len(self.sweeps)

    def failing_object_ids(self) -> set[str]:
        """Certified failures — the red channel. Not uncertainty, which is amber."""
        failing = {c.object_id for c in self.scale if not c.passed}
        failing |= {c.object_id for c in self.stability if not c.passed}
        failing |= {c.object_id for c in self.sweeps if not c.passed}
        failing |= {c.object_id for c in self.inertial if not c.passed}
        return failing


# --- stage 8: repair ----------------------------------------------------------


class RepairKind(StrEnum):
    SNAP_TO_SUPPORT = "snap_to_support"
    RESOLVE_PENETRATION = "resolve_penetration"
    RESCALE = "rescale"
    TRIM_JOINT_LIMITS = "trim_joint_limits"
    MOVE_JOINT_ORIGIN = "move_joint_origin"
    ROTATE_JOINT_AXIS = "rotate_joint_axis"
    WELD_JOINT = "weld_joint"
    UPGRADE_PROXY_TIER = "upgrade_proxy_tier"


class RepairAction(BaseModel):
    """Given an interfering joint the fix could be the axis, the origin, the limits
    or the scale. Choosing badly silently discards range the object really has, so
    the magnitude and the axis repaired are both recorded.
    """

    kind: RepairKind
    target_id: str = Field(description="object_id, or joint_id for the joint repairs.")
    axis_repaired: str

    delta_position_m: Vec3 = (0.0, 0.0, 0.0)
    delta_scale: float = 1.0
    delta_limits: JointLimits | None = None

    magnitude: float = Field(default=0.0, description="Size of the correction, for minimality.")
    improved: bool = Field(
        default=True,
        description="Whether revalidation actually got better. A repair that made "
        "things worse is discarded rather than reported.",
    )


class RepairResult(BaseModel):
    graph: SceneGraph
    certificate: Certificate
    actions: list[RepairAction] = Field(default_factory=list)


# --- stage 10: export ---------------------------------------------------------


class ExportResult(BaseModel):
    mjcf_path: str | None = None
    urdf_path: str | None = None
    gltf_path: str | None = None


# --- the edit loop ------------------------------------------------------------


class ObjectEdit(BaseModel):
    """A user edit. Every field is optional; whatever is set becomes a pinned
    constraint held fixed while the rest of the scene re-solves around it.

    Re-routing sends the object back through the other branch of stage 4, so it
    is an edit like any other rather than a separate operation.
    """

    object_id: str
    scale: float | None = None
    position_m: Vec3 | None = None
    orientation: Quat | None = None
    supported_by: str | None = None
    mass_kg: float | None = None
    route: Route | None = None


class JointEdit(BaseModel):
    joint_id: str
    type: JointType | None = None
    axis: Vec3 | None = None
    origin_m: Vec3 | None = None
    limits: JointLimits | None = None
    dynamics: JointDynamics | None = None
    weld: bool = False


class SceneEditRequest(BaseModel):
    objects: list[ObjectEdit] = Field(default_factory=list)
    joints: list[JointEdit] = Field(default_factory=list)
    anchors: list[ScaleAnchor] = Field(default_factory=list)
    resolve: bool = Field(
        default=True, description="Re-run stage 6 with the edits pinned, then re-certify."
    )


# --- v2: the amber channel ----------------------------------------------------


class ObjectUncertainty(BaseModel):
    """High uncertainty is not failure. An object can be confidently wrong or
    uncertainly fine, and those warrant different user responses — which is why
    this is a separate channel from Certificate rather than a field on it.

    Every field is optional because reporting confidence obligates calibrating
    it, and an uncalibrated signal is better absent than shown.
    """

    object_id: str
    scale_variance: float | None = None  # inverse-Hessian diagonal
    depth_prior_disagreement_m: float | None = None
    joint_score_margin: float | None = None
    expected_vs_fitted_joints: tuple[int, int] | None = None
    used_obb_fallback: bool | None = None
    visible_surface_fraction: float | None = None


# --- the thing the API returns ------------------------------------------------


class SceneSpec(BaseModel):
    scene_id: str
    image_path: str
    intrinsics: Intrinsics

    graph: SceneGraph
    certificate: Certificate = Field(default_factory=Certificate)

    anchors: list[ScaleAnchor] = Field(default_factory=list)
    weights: SolveWeights = Field(default_factory=SolveWeights)
    diagnostics: SolveDiagnostics = Field(default_factory=SolveDiagnostics)
    repairs_applied: list[RepairAction] = Field(default_factory=list)

    exports: ExportResult = Field(default_factory=ExportResult)
    uncertainty: list[ObjectUncertainty] = Field(default_factory=list)
