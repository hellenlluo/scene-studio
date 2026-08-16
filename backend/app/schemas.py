"""Typed contracts flowing between pipeline stages.

Every stage consumes and returns one of these, so stages are independently
testable and swappable without running the rest of the pipeline (or a GPU).
"""

from enum import StrEnum

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # w, x, y, z — MuJoCo order


class StageName(StrEnum):
    SEGMENT = "segment"
    DEPTH = "depth"
    LABEL = "label"
    SCALE = "scale"
    GEOMETRY = "geometry"
    COLLISION = "collision"
    ASSEMBLE = "assemble"
    VALIDATE = "validate"
    REPAIR = "repair"
    EXPORT = "export"


class ProxyTier(StrEnum):
    """Collision-geometry fidelity. Every object is guaranteed at least an OBB."""

    OBB = "obb"
    CONVEX_HULL = "convex_hull"
    DECOMPOSED = "decomposed"


# --- stage 1: segment ---------------------------------------------------------


class ObjectMask(BaseModel):
    object_id: str
    mask_path: str  # single-channel PNG, image resolution
    bbox_px: tuple[int, int, int, int]
    area_px: int


class SegmentResult(BaseModel):
    masks: list[ObjectMask]


# --- stage 2: depth -----------------------------------------------------------


class Intrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class DepthResult(BaseModel):
    depth_path: str  # float32 .npy, metres, image resolution
    intrinsics: Intrinsics
    is_metric: bool = Field(description="False if the model output is only up-to-scale.")


# --- stage 3: label -----------------------------------------------------------


class ObjectLabel(BaseModel):
    object_id: str
    category: str
    # VLM prior on real-world size. Coarse but roughly unbiased, which is what
    # makes it useful against depth error that is biased per object.
    prior_dims_m: Vec3
    prior_confidence: float = Field(ge=0.0, le=1.0)
    likely_articulated: bool = False


class LabelResult(BaseModel):
    labels: list[ObjectLabel]


# --- stage 4: scale -----------------------------------------------------------


class ScaleResult(BaseModel):
    """Reconciliation of metric depth against per-object class priors."""

    scene_scale: float = Field(description="Multiplier applied to depth to reach metres.")
    per_object_dims_m: dict[str, Vec3]
    residuals_m: dict[str, float] = Field(
        default_factory=dict,
        description="Post-fit disagreement between depth and prior, per object. "
        "Large residuals are the signal for which objects need user correction.",
    )


# --- stages 5-6: geometry, collision -----------------------------------------


class ObjectGeometry(BaseModel):
    object_id: str
    visual_mesh_path: str | None = None  # None when the OBB fallback is in use
    collision_mesh_paths: list[str] = Field(default_factory=list)
    proxy_tier: ProxyTier
    dims_m: Vec3
    mass_kg: float
    inertia_diag: Vec3
    density_kg_m3: float


class GeometryResult(BaseModel):
    geometries: list[ObjectGeometry]


# --- stage 7: assemble --------------------------------------------------------


class PlacedObject(BaseModel):
    object_id: str
    position_m: Vec3
    orientation: Quat
    supported_by: str | None = Field(
        default=None, description="object_id of the supporting body, or None for the floor."
    )


class SceneLayout(BaseModel):
    objects: list[PlacedObject]
    floor_height_m: float = 0.0


# --- stages 8-9: validate, repair --------------------------------------------


class ObjectValidation(BaseModel):
    object_id: str
    passed: bool
    com_displacement_m: float
    orientation_drift_deg: float
    penetration_m: float
    reasons: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    objects: list[ObjectValidation]

    @property
    def pass_rate(self) -> float:
        if not self.objects:
            return 1.0
        return sum(o.passed for o in self.objects) / len(self.objects)


class RepairAction(BaseModel):
    object_id: str
    kind: str  # "snap_to_support" | "resolve_penetration" | "rescale"
    delta_position_m: Vec3 = (0.0, 0.0, 0.0)
    delta_scale: float = 1.0


# --- the thing the API returns ------------------------------------------------


class SceneSpec(BaseModel):
    scene_id: str
    image_path: str
    intrinsics: Intrinsics
    scene_scale: float
    labels: list[ObjectLabel]
    geometries: list[ObjectGeometry]
    layout: SceneLayout
    validation: ValidationResult
    repairs_applied: list[RepairAction] = Field(default_factory=list)
    mjcf_path: str | None = None
    gltf_path: str | None = None
