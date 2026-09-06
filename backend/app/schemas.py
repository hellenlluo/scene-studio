"""Typed contracts flowing between pipeline stages.

Every stage consumes and returns one of these, so stages are independently
testable and swappable without running the rest of the pipeline (or a GPU).

Two commitments shape this file:

* **Scale is per object, not global.** Stage 6 optimises one isotropic scale per
  object jointly with pose. A single scene-wide multiplier cannot express the
  couplings the approach rests on — a mug whose base has to sit inside a tabletop
  constrains two objects against each other, not the scene against the camera.
* **Certification is a four-axis contract, not a boolean.** Each axis carries the
  measurements it passed or failed on, because "simulation-ready" has to be
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
    RECONSTRUCT = "reconstruct"
    RECONCILE = "reconcile"
    # Runs after RECONCILE, before INERTIA: objects are posed in world space by
    # then, which a render needs, but nothing has been solved or certified yet, so
    # a hallucinated or duplicated object is dropped before any of that work is
    # spent on it.
    VERIFY = "verify"
    # Runs before SOLVE, not after CERTIFY: the solver's physics block steps
    # MuJoCo, and MuJoCo cannot settle a body with no mass. This stage assigns
    # the density priors; mass and the inertia tensor are derived from
    # density x scaled volume and so are recomputed whenever a scale moves.
    INERTIA = "inertia"
    SOLVE = "solve"
    CERTIFY = "certify"
    REPAIR = "repair"
    EXPORT = "export"


class Provenance(StrEnum):
    """Where a value came from. Reported per field.

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


class ObjectMask(BaseModel):
    object_id: str
    mask_path: str  # single-channel PNG, image resolution
    bbox_px: tuple[int, int, int, int]
    area_px: int


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


# --- stage 3: label -----------------------------------------------------------


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


class Material(StrEnum):
    """Dominant surface material, which is what density is keyed on.

    Coarse on purpose. A closed vocabulary is something a VLM judges reliably from
    a photo, whereas a number in kg/m3 is not — so the model picks a bucket and
    the density comes from a table.
    """

    WOOD = "wood"
    METAL = "metal"
    PLASTIC = "plastic"
    GLASS = "glass"
    CERAMIC = "ceramic"
    FABRIC = "fabric"
    STONE = "stone"
    PAPER = "paper"
    OTHER = "other"


class ObjectLabel(BaseModel):
    object_id: str
    category: str
    material: Material = Field(
        default=Material.OTHER,
        description="Feeds the density prior. Only meaningful against a *mesh* "
        "volume: material density times a bounding-box volume overestimates open "
        "shapes by an order of magnitude. See app.pipeline.inertia.",
    )
    prior: DimensionPrior | None = Field(
        default=None,
        description="Absent until there is a measured priors table to draw on. A VLM "
        "asked for metres is guessing, and a fabricated prior with an invented sigma "
        "is worse than none: E_prior weights by 1/sigma^2, so a made-up sigma is a "
        "made-up weight. While this is None, absolute scale rests entirely on metric "
        "depth — which is row 1 of the ablation, not a workaround.",
    )
    support_parent: str | None = Field(
        default=None,
        description="object_id of the supporting body, or None for the floor. The VLM "
        "supplies an initial guess; reconcile overrides it with measured contact once "
        "geometry exists, since by then it is observable rather than inferred.",
    )


class LabelResult(BaseModel):
    labels: list[ObjectLabel]


# --- stage 4: reconstruct -----------------------------------------------------


class ProxyTier(StrEnum):
    """Collision-geometry fidelity.

    Describes the *collision* representation specifically, which stays OBB until the
    inertia stage runs CoACD over the visual mesh. Whether reconstruction succeeded
    is a separate question, answered by `PartGeometry.visual_mesh_path` being set —
    an object can have a real mesh and still be colliding as a box.
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
    """How a reconstruction model's raw output was brought into the canonical frame.

    Models emit assets in different up-axes, different front-faces, and different
    scale conventions — most normalise to a unit box. Recording the transform that
    was stripped is what lets stage 6 optimise one scale variable per object
    instead of two incompatible ones, and keeps the reconciliation auditable when
    an object comes out sideways.
    """

    source: str = Field(description="Model that produced the asset, e.g. 'sam3d'.")
    up_axis: Axis = Axis.Z_POS
    front_axis: Axis = Axis.Y_NEG
    rotation: Quat = (1.0, 0.0, 0.0, 0.0)
    normalization_scale: float = Field(
        default=1.0,
        description="The model's own internal normalisation, divided out here so it "
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
    volume_m3: float = Field(
        description="Enclosed volume of the **visual** mesh, not the collision proxy "
        "and not the bounding box. Measured on a dining table: the mesh encloses "
        "0.064 m3, its convex hull 0.962 m3, its bounding box 1.013 m3 — so taking "
        "volume from the proxy gives a 15x mass error on any open shape. The two "
        "meshes exist for different purposes and this is the one that means mass."
    )
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
    """One rigid piece of an object.

    Most objects are a single part. The hierarchy exists because a reconstruction
    can come back as several rigidly-attached pieces — a lamp base and its shade —
    and keeping them separate gives the collision proxy something better than one
    box around the union.
    """

    part_id: str
    name: str
    parent_part_id: str | None = None

    visual_mesh_path: str | None = None  # None when the OBB fallback is in use
    collision_mesh_paths: list[str] = Field(default_factory=list)
    proxy_tier: ProxyTier = ProxyTier.OBB

    dims_m: Vec3 = Field(description="Extent at unit object scale; multiply by SceneObject.scale.")
    origin_m: Vec3 = (0.0, 0.0, 0.0)  # part frame origin, object canonical frame

    source_volume_m3: float | None = Field(
        default=None,
        description="Enclosed volume measured on the mesh **as reconstructed**, "
        "before decimation. Mass is computed from this rather than from the stored "
        "file: decimation is what makes a 22 MB mesh shippable, but it breaks "
        "watertightness, and trimesh's volume is only meaningful on a closed "
        "surface. Measured on one object — 1.1M faces to 20k changed the volume by "
        "0.18%, so the loss is in the guarantee, not the number.",
    )
    source_watertight: bool = Field(
        default=False,
        description="Whether the mesh was closed *before* decimation, and so whether "
        "`source_volume_m3` can be trusted. Pessimistic by default.",
    )

    inertial: InertialProperties | None = None  # populated by the inertia stage

    visible_surface_fraction: float | None = Field(
        default=None,
        description="Observed fraction of the part's extent. Single-view depth sees "
        "the front shell only, so the rest is amodal inference. v2 uncertainty signal.",
    )


class ObjectAssets(BaseModel):
    """Reconstruction output for one object, in the model's own frame and units."""

    object_id: str
    frame: AssetFrame
    parts: list[PartGeometry]

    camera_translation: Vec3 | None = Field(
        default=None,
        description="Where the reconstruction model put this object, in the camera "
        "frame and in its own units. A placement *hypothesis*, not a measurement — "
        "reconcile has metric depth and should prefer it. Kept because discarding "
        "the model's opinion of the layout throws away information that is free.",
    )

    failed: bool = False
    failure_reason: str | None = None


class ReconstructionResult(BaseModel):
    objects: list[ObjectAssets]


# --- stage 5: reconcile -------------------------------------------------------


class DepthObservation(BaseModel):
    """What stage 2 measured about one object, kept so stage 6 can be re-run.

    Six numbers, and they are the entire depth input to the solve: `_residuals`
    scores the object's world AABB centre and extent against these and reads
    nothing else from the depth map. Storing them rather than the 4 MB map is what
    makes a re-solve stateless — a user edit can be committed and the scene
    re-fitted without the depth `.npy` and every mask PNG still being on disk under
    the key that produced them.

    In the **recentred world frame**, not the camera frame `reconcile.observe`
    reports. Applying `world_offset_m` once here means a re-solve does not have to
    know whether it has been applied already, which is the sort of thing that is
    silently wrong in one direction.
    """

    centre_m: Vec3
    extent_m: Vec3


class SceneObject(Pinnable):
    """One object in the canonical, gravity-aligned scene graph.

    scale, position_m and orientation are exactly the stage-6 decision variables.
    Provenance decides which of them are free.
    """

    object_id: str
    label: ObjectLabel
    frame: AssetFrame

    parts: list[PartGeometry]

    # --- solve variables ---
    scale: float = Field(default=1.0, gt=0.0, description="Isotropic. The s_i of stage 6.")
    position_m: Vec3 = (0.0, 0.0, 0.0)
    orientation: Quat = (1.0, 0.0, 0.0, 0.0)

    supported_by: str | None = Field(
        default=None, description="object_id of the supporting body, or None for the floor."
    )

    observation: DepthObservation | None = Field(
        default=None,
        description="What depth measured for this object, carried so stage 6 can be "
        "re-run without the depth map. Absent on hand-authored scenes, which have no "
        "measurement behind them — the solver then has no `E_depth` term for the "
        "object and relies on contact alone.",
    )

    degradation_reason: str | None = Field(
        default=None,
        description="Why this object came back worse than intended, when it did — "
        "reconstruction failed and it fell back to a box, say. Shown in the viewer, "
        "because 'we could not reconstruct this' is a different message to the user "
        "than 'this reconstructed but fails certification'.",
    )

    @model_validator(mode="after")
    def _check_structure(self) -> "SceneObject":
        """Referential integrity of the part tree.

        Not defensive programming: a part naming a parent that does not exist
        surfaces as a KeyError deep inside MJCF emission, where the message says
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

        return self

    @property
    def root_part(self) -> PartGeometry:
        """The part with no parent. Falls back to the first only for a flat list.

        Not `parts[0]`: nothing orders this list, and a reconstruction is free to
        emit a shade before the base it sits on.
        """
        return next((p for p in self.parts if p.parent_part_id is None), self.parts[0])

    @property
    def dims_m(self) -> Vec3:
        """Root part extent at the current scale."""
        w, h, d = self.root_part.dims_m
        return (w * self.scale, h * self.scale, d * self.scale)


class CameraPose(BaseModel):
    """Where the photo was taken from, in world coordinates.

    Recoverable exactly rather than guessed: the reconstruction is built in camera
    coordinates, so the camera sits at the origin of the frame reconcile rotates and
    shifts. Its height above the floor falls out of the floor fit — on one room,
    1.16 m, which is a plausible standing eye level and a free sanity check on the
    plane.

    The viewer opens from here so a reconstruction can be compared against the photo
    it came from without the user hunting for the angle.
    """

    position_m: Vec3 = (0.0, 0.0, 0.0)
    forward: Vec3 = Field(default=(0.0, 1.0, 0.0), description="Unit view direction.")
    up: Vec3 = (0.0, 0.0, 1.0)
    vertical_fov_deg: float = Field(
        default=60.0, description="Derived from the intrinsics, so the framing matches."
    )


class SceneGraph(BaseModel):
    objects: list[SceneObject]
    camera: CameraPose | None = None
    floor_height_m: float = 0.0
    world_offset_m: Vec3 = Field(
        default=(0.0, 0.0, 0.0),
        description="Translation already applied to bring the scene's horizontal "
        "centre onto the origin.\n\n"
        "Reconstruction happens in camera coordinates, so the untranslated origin is "
        "wherever the photographer stood and a scene lands somewhere different for "
        "every photo. That is correct and unhelpful: a viewer with one fixed camera, "
        "and a user comparing two reconstructions, both want the scene in the same "
        "place. Recording the offset rather than discarding it keeps the frames "
        "relatable — `solve` measures against depth observations that are still in "
        "camera coordinates, and has to undo this to compare like with like.\n\n"
        "Horizontal only. Z is the floor, fixed by the plane fit and shared with "
        "MJCF's ground plane, so shifting it would put objects above a floor that "
        "does not move.",
    )
    gravity_rotation: Quat = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description="Rotation applied in stage 5 to bring the fitted floor plane "
        "world-down. Everything after this assumes gravity along -Z.",
    )

    def get(self, object_id: str) -> SceneObject | None:
        return next((o for o in self.objects if o.object_id == object_id), None)


class SupportSurface(BaseModel):
    """An upward-facing face something could rest on.

    Geometric only: this says a placement is *possible*, never that it is
    sensible. An apple fits perfectly well on a toilet cistern. Deciding which
    possible placements make sense is the VLM's job, and the point of computing
    these is to hand it a short list that is at least physically achievable.
    """

    object_id: str
    part_id: str
    height_m: float = Field(
        description="World z of the face centroid. A face tilted within tolerance "
        "has no single height, and the solver refines the contact anyway."
    )
    polygon_xy: list[tuple[float, float]] = Field(description="World XY, in ring order.")
    normal: Vec3
    area_m2: float
    free_area_m2: float = Field(
        description="Area minus the footprint of whatever already rests here."
    )


# --- stage 5.5: verify ---------------------------------------------------------


class ObjectVerdict(StrEnum):
    """What the reconstruction-verification pass decided about one object,
    comparing a render of the posed scene against the original photo."""

    OK = "ok"
    DUPLICATE = "duplicate"
    EXTERNAL = "external"


class ObjectVerification(BaseModel):
    """One object's verdict, kept whether or not it survives — a removal with no
    visible reason is not something anyone could check later."""

    object_id: str
    verdict: ObjectVerdict
    duplicate_of: str | None = Field(
        default=None, description="The object_id it duplicates, when verdict is DUPLICATE."
    )
    reason: str = Field(default="", description="What the model saw, for debugging.")


class VerifyResult(BaseModel):
    graph: SceneGraph
    removed: list[ObjectVerification] = Field(
        default_factory=list, description="Only the non-OK verdicts; OK is not worth persisting."
    )


# --- stage 6: solve -----------------------------------------------------------


class SolveWeights(BaseModel):
    """The lambdas of stage 6's objective."""

    depth: float = 1.0
    prior: float = 1.0
    silhouette: float = 0.5
    support: float = Field(
        default=8.0,
        description="Weight on the signed support gap.\n\n"
        "Has to outweigh depth on the vertical axis, and by more than it looks: "
        "depth contributes six residuals per object — three for the centre, three "
        "for the extent — against one for support, so parity of *weights* is a "
        "six-to-one loss for contact. At 2.0 a reconstructed side table settled "
        "17.8 mm clear of the rug it rests on, against a 5 mm tolerance, because the "
        "depth measurement wanted it higher and outvoted the contact.\n\n"
        "Raised to 8.0 on the measurement: table gap 17.8 mm -> 5.1 mm, and the "
        "depth term's own cost rose only 0.374 -> 0.406, about 10%. Contact is "
        "bought cheaply because the two measurements disagree along one axis only. "
        "30.0 buys a further 5.1 -> 3.1 mm for no extra depth cost and 100.0 buys "
        "nothing more, so this sits at the knee rather than at the floor — a contact "
        "constraint that overrode depth entirely would be discarding the "
        "measurement the whole pipeline exists to make.",
    )
    penetration: float = Field(
        default=20.0,
        description="Weight on pairwise overlap depth.\n\n"
        "Raised from 5.0 on measurement. At 5.0 it was outvoted the same way "
        "`support` was before its own raise — depth contributes six residuals per "
        "object against one per *pair* here — and the overlap it left was the thing "
        "physics could not survive: a 14.5 mm book-on-book overlap ejected a 0.158 kg "
        "book at 16.7 m/s. Swept on `room2.png`, worst overlap and settle drift both "
        "improve to 20 and both get worse past it:\n\n"
        "    weight   worst overlap   settle drift\n"
        "         5         7.5 mm        2601 mm\n"
        "        20         4.5 mm        1768 mm\n"
        "        60        43.5 mm        2554 mm\n\n"
        "60 overshoots — pushing one pair apart drives each of them into something "
        "else — so this sits at the minimum rather than at the top of the range. "
        "`room1.png` certifies on all four axes at both 5 and 20 with identical "
        "settle drift, so the raise costs nothing on a scene that was already sound.",
    )
    containment: float = Field(
        default=0.0,
        description="Weight on how far a supported object's footprint centre sits "
        "*outside* its support.\n\n"
        "The support term is signed but vertical — it closes the gap to whatever an "
        "object rests on and says nothing about staying over it. So nothing in the "
        "objective resisted lateral drift, and depth is free to slide a child off "
        "its parent at no cost. Measured on `room2.png`: between reconcile and "
        "solve, four objects on the side table moved 314, 904, 1007 and 1225 mm "
        "sideways while keeping the same `supported_by`, and seven of nine scale "
        "failures were objects at the right height (gap under 3 mm against a 5 mm "
        "bar) that were no longer over anything. Floor-standing objects moved "
        "0.0-0.8 mm over the same solve, which is what a term acting only on "
        "supported objects looks like when it is missing.\n\n"
        "One-sided and zero inside, exactly like penetration: an object over its "
        "support pays nothing, so this cannot distort a scene that was already "
        "right. Measured on the footprint *centre* against the parent's AABB, "
        "because that is the criterion `certify.scale` reports — a laptop "
        "overhanging a side table by one corner is resting on it, and requiring "
        "full containment would fail it.\n\n"
        "**Off by default, because on a concave parent it does more harm than good.** "
        "The target is the parent\'s AABB, and a bookshelf\'s AABB is mostly solid "
        "shelf — so pulling a vase\'s centre \'inside\' it drives the vase into the "
        "structure. Swept on `room2.png`, against the scene entering solve with no "
        "overlap at all:\n\n"
        "    weight   vase buried   not-over-support   settle drift\n"
        "         0       0.00 mm           6/15           1237 mm\n"
        "         8       4.95 mm           4/15           1568 mm\n"
        "        20      20.62 mm           3/15           1731 mm\n"
        "        60      34.03 mm           1/15           2358 mm\n\n"
        "It buys the scale axis by paying the stability axis, monotonically, and "
        "settling degrades at every step — the 34 mm burial is what MuJoCo ejects at "
        "15.9 m/s before diverging. The term itself is right and its unit tests hold "
        "on a flat support; what is wrong is the target. It should pull toward the "
        "parent\'s *supporting surface*, not its bounding box, and today\'s "
        "`SupportHeights` cannot express that — a top-down heightfield keeps one "
        "height per column, so it cannot represent a shelf with anything under it. "
        "Turn this on once support is a surface rather than a heightfield.\n\n"
        "For the record, on a flat support the weight sweep was clean: 60.0 is the "
        "floor rather than the knee, which is the opposite of how "
        "`support` is tuned, and for a reason that does not apply there. Swept "
        "against depth wrong by 0.8 m, the residual offset goes 311 mm at 0, then "
        "10.8, 2.8, 0.7, 0.2 and 0.0 mm at 4, 8, 16, 30 and 60 — while the depth "
        "term\'s own cost sits at 0.043 from 4 upward and never moves again. "
        "`base_inside_parent` is a boolean, so a term that leaves 0.7 mm outside "
        "has not fixed the check it exists to fix, and there is no repair strategy "
        "for containment to finish the job — `_snap_to_support` only translates in "
        "z. Weighting `support` this hard would override depth on every object "
        "because it is signed and always active; this one is inert until an object "
        "has already left its support, so buying the boolean outright costs "
        "nothing on a scene that was placed correctly.",
    )


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
    """What stage 6's block-coordinate loop did, and how far it got.

    `converged` and `settled` are separate because they answer different questions
    and one used to mask the other. A single flag reported `False` on a scene where
    the optimiser was converging cleanly — 199, 40, 16, 7 mm per round — purely
    because one floor lamp fell over, which reads as "the solve failed" when the
    solve was working and the *reconstruction* was unstable.
    """

    iterations: int = 0
    converged: bool = Field(
        default=False,
        description="The optimiser reached a fixed point: a round moved every "
        "variable by less than the round tolerance. About the solve, not the scene.",
    )
    settled: bool = Field(
        default=False,
        description="The solved pose is already the equilibrium — settling the scene "
        "under gravity moves nothing by more than the round tolerance.\n\n"
        "Stricter than 'the objects stop moving', and deliberately. A toppled lamp "
        "lying still has stopped moving: measured, it is motionless from t=3.5 s and "
        "844 mm from where the solver put it. That scene is *stable*; it is not "
        "*correct*. Simulation-ready means the MJCF loads and nothing jumps, which "
        "is the same thing the stability axis asks with `max_com_displacement_m`.",
    )
    max_settle_drift_m: float = Field(
        default=0.0,
        description="Furthest any object moves between its solved and settled pose. "
        "The scene-wide maximum, so one unstable object dominates it — which is why "
        "it is reported as a number rather than only as the flag above.",
    )
    residual_by_term: dict[str, float] = Field(
        default_factory=dict, description="Final E_depth, E_prior, E_sil, E_supp, E_pen."
    )

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
    NOT_APPLICABLE = "not_applicable"  # nothing to check
    NOT_RUN = "not_run"  # the validator did not execute

    """NOT_APPLICABLE and NOT_RUN are deliberately distinct.

    "There was nothing of this kind to check" and "nobody checked whether this
    scene is correctly scaled" are different claims, and collapsing them lets a
    scene report as certified on the strength of the axes that happened to be
    wired up. `Certificate.passed` treats NOT_RUN as disqualifying and
    NOT_APPLICABLE as fine.
    """


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
    penetration_against: str | None = Field(
        default=None,
        description="What the deepest overlap is with: another object's id, or "
        '`"floor"`. None when there is no overlap.\n\n'
        "Recorded because the depth alone does not say what to do about it. Sunk "
        "into the ground and interpenetrating a neighbour are different "
        "reconstruction errors with different fixes, and a reader told only "
        '"overlaps another object" will go looking through the object list for a '
        "collision that is with the floor.",
    )
    passed: bool


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

    scale_status: AxisStatus = AxisStatus.NOT_RUN
    stability_status: AxisStatus = AxisStatus.NOT_RUN
    inertial_status: AxisStatus = AxisStatus.NOT_RUN
    cost_status: AxisStatus = AxisStatus.NOT_RUN

    scale: list[ScaleCheck] = Field(default_factory=list)
    stability: list[StabilityCheck] = Field(default_factory=list)
    inertial: list[InertialCheck] = Field(default_factory=list)
    cost: CostCheck | None = None

    penetration_tolerance_m: float = Field(
        default=0.002,
        description="Populated from Settings.max_penetration_m by the certify stage; "
        "recorded on the certificate so a stored result stays interpretable after "
        "the setting changes. Non-zero because mesh discretisation produces "
        "sub-millimetre contacts on surfaces flush by design, and a zero-tolerance "
        "check would fail every well-modelled object. Results are reported against "
        "this value so their sensitivity to it is visible.",
    )
    com_displacement_tolerance_m: float = Field(
        default=0.01,
        description="From Settings.max_com_displacement_m, recorded for the same "
        "reason as the penetration tolerance.",
    )
    support_gap_tolerance_m: float = Field(
        default=0.005,
        description="From Settings.max_support_gap_m — how far an object may sit from "
        "what it rests on before the scale axis calls it floating or sunk.\n\n"
        "Added after a client reported an object as 'floating above its support by "
        "0.9 mm (tolerance 2 mm)' when it passed the gap test outright at a 5 mm "
        "bar and had failed on containment instead. With no gap tolerance published "
        "the client had reached for `penetration_tolerance_m`, which is the closest "
        "number on the certificate and the wrong one.",
    )
    prior_deviation_tolerance_sigma: float = Field(
        default=3.0,
        description="From Settings.max_prior_deviation_sigma. Published for the same "
        "reason: a client with no threshold to read invents one, and an invented "
        "cutoff disagrees with the axis it claims to explain.",
    )
    orientation_drift_tolerance_deg: float = Field(
        default=2.0,
        description="From Settings.max_orientation_drift_deg.\n\n"
        "All five thresholds are on the certificate so a reader can say *which* "
        "criterion a check failed on. Without them a client has to invent its own "
        "cutoffs, and one did: a floor lamp failing on 295 mm of displacement was "
        "also reported as overlapping its neighbour, because the client tested "
        "penetration against zero and the lamp had 4.7 micrometres of contact "
        "noise — rendered, after rounding, as 'overlaps another object by 0 mm'.",
    )

    @property
    def axes(self) -> dict[str, AxisStatus]:
        return {
            "scale": self.scale_status,
            "stability": self.stability_status,
            "inertial": self.inertial_status,
            "cost": self.cost_status,
        }

    @property
    def passed(self) -> bool:
        """Every axis was reached, none failed, and at least one had work to do.

        None of the three clauses is pedantry. Without the NOT_RUN clause a scene
        certifies on the strength of whichever validators happen to be wired up,
        which is the same dishonesty the per-axis contract exists to prevent, just
        one level higher. Untested is not passed.
        """
        statuses = list(self.axes.values())
        return (
            all(s is not AxisStatus.FAIL for s in statuses)
            and all(s is not AxisStatus.NOT_RUN for s in statuses)
            and any(s is AxisStatus.PASS for s in statuses)
        )

    @property
    def unchecked_axes(self) -> list[str]:
        """Axes no validator reached. Non-empty means `passed` cannot be True."""
        return [name for name, status in self.axes.items() if status is AxisStatus.NOT_RUN]

    @property
    def stability_pass_rate(self) -> float | None:
        """None when nothing was settled. There is no rate over an empty set."""
        if not self.stability:
            return None
        return sum(c.passed for c in self.stability) / len(self.stability)

    @property
    def scale_pass_rate(self) -> float | None:
        if not self.scale:
            return None
        return sum(c.passed for c in self.scale) / len(self.scale)

    def failing_object_ids(self) -> set[str]:
        """Certified failures — the red channel. Not uncertainty, which is amber."""
        failing = {c.object_id for c in self.scale if not c.passed}
        failing |= {c.object_id for c in self.stability if not c.passed}
        failing |= {c.object_id for c in self.inertial if not c.passed}
        return failing


# --- stage 8: repair ----------------------------------------------------------


class RepairKind(StrEnum):
    SNAP_TO_SUPPORT = "snap_to_support"
    RESOLVE_PENETRATION = "resolve_penetration"
    RESCALE = "rescale"
    UPGRADE_PROXY_TIER = "upgrade_proxy_tier"


class RepairAction(BaseModel):
    """The magnitude and the axis repaired are both recorded, because choosing a
    correction badly discards information silently and that is the failure mode
    worth engineering against.
    """

    kind: RepairKind
    target_id: str
    axis_repaired: str

    delta_position_m: Vec3 = (0.0, 0.0, 0.0)
    delta_scale: float = 1.0

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

    rounds_used: int = 0
    converged: bool = Field(
        default=True,
        description="False means repair stopped because it ran out of rounds, not "
        "because it was finished. The scene might still certify with a larger "
        "budget, and a caller that cannot tell the difference would report a "
        "truncated repair as a genuine failure.",
    )


# --- stage 10: export ---------------------------------------------------------


class ExportResult(BaseModel):
    """Paths relative to the storage directory, not absolute.

    These are what a client fetches through the `/storage` mount, so it builds the
    URL as `/storage/{path}`. Storing the absolute form would be unusable to a
    browser and would break the moment the storage directory moves — which it does
    between a developer's machine and a test's temp dir.
    """

    mjcf_path: str | None = None
    gltf_path: str | None = None


# --- the edit loop ------------------------------------------------------------


class ObjectEdit(BaseModel):
    """A user edit. Every field is optional; whatever is set becomes a pinned
    constraint held fixed while the rest of the scene re-solves around it.
    """

    object_id: str
    scale: float | None = None
    position_m: Vec3 | None = None
    orientation: Quat | None = None
    supported_by: str | None = None
    mass_kg: float | None = None


class SceneEditRequest(BaseModel):
    objects: list[ObjectEdit] = Field(default_factory=list)
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
    used_obb_fallback: bool | None = None
    visible_surface_fraction: float | None = None


# --- the thing the API returns ------------------------------------------------


class SceneSpec(BaseModel):
    scene_id: str
    image_path: str = Field(
        description="Relative to the storage directory, like ExportResult's paths — "
        "a client builds the URL as /storage/{image_path}."
    )
    intrinsics: Intrinsics

    graph: SceneGraph
    certificate: Certificate = Field(default_factory=Certificate)

    anchors: list[ScaleAnchor] = Field(default_factory=list)
    weights: SolveWeights = Field(default_factory=SolveWeights)
    diagnostics: SolveDiagnostics = Field(default_factory=SolveDiagnostics)
    repairs_applied: list[RepairAction] = Field(default_factory=list)
    removed_objects: list[ObjectVerification] = Field(
        default_factory=list,
        description="Objects the verification stage dropped as external or duplicated, "
        "with the reasoning — the auditable record of what reconstruction produced "
        "that this scene does not.",
    )

    exports: ExportResult = Field(default_factory=ExportResult)
    uncertainty: list[ObjectUncertainty] = Field(default_factory=list)
