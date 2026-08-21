from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class MissingCredential(RuntimeError):
    """Raised at the point of use, naming the variable and the stage that wanted it.

    Better than the SDK's own error, which surfaces three frames deep into a
    background job and says nothing about which stage was running.
    """


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SCENESTUDIO_")

    database_url: str = f"sqlite:///{BACKEND_ROOT / 'storage' / 'scenestudio.db'}"
    storage_dir: Path = BACKEND_ROOT / "storage"
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- provider credentials ---
    # `validation_alias` bypasses env_prefix: these are the names the vendor SDKs
    # expect, so a developer who already has FAL_KEY exported for the fal CLI does
    # not need to re-export it as SCENESTUDIO_FAL_KEY. Reading them through Settings
    # rather than os.environ is what makes `backend/.env` work at all —
    # pydantic-settings loads that file into this object and never touches the
    # process environment, so an SDK constructed with no arguments would not see it.
    openai_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    fal_key: str | None = Field(default=None, validation_alias=AliasChoices("FAL_KEY"))
    replicate_api_token: str | None = Field(
        default=None, validation_alias=AliasChoices("REPLICATE_API_TOKEN")
    )

    # --- model selection ---
    # Endpoints and model ids are settings, not literals, so swapping a provider is
    # a config change and an ablation over model choice does not need a code edit.
    openai_model: str = Field(
        default="gpt-5",
        description="Vision-capable model for stage 3. Verify against the current "
        "model list before relying on this default.",
    )
    fal_segment_endpoint: str = "fal-ai/sam-3/image"
    fal_mesh_endpoint: str = "fal-ai/sam-3/3d-objects"
    replicate_depth_model: str = Field(
        default="",
        description="MoGe-2 or UniDepth. Must be metric — Marigold and MiDaS are "
        "relative-depth only and cannot carry the size role of E_depth.",
    )

    # --- stage 2: local depth ---
    depth_model: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf"
    depth_device: str | None = Field(
        default=None, description="Force a torch device. None auto-selects mps/cuda/cpu."
    )
    depth_is_metric: bool = Field(
        default=True,
        description="Set False if depth_model points at a relative-depth checkpoint "
        "(MiDaS, Marigold, plain Depth Anything). E_depth then contributes to "
        "position only and carries no size information, so say so honestly rather "
        "than letting the solver treat the output as metres.",
    )
    assumed_hfov_deg: float = Field(
        default=60.0,
        description="Fallback horizontal field of view when EXIF carries no focal "
        "length. Backprojection puts lateral extent at (u - cx) * Z / fx, so an "
        "error here distorts estimated width and height while leaving depth correct.",
    )

    def require(self, field: str, stage: str) -> str:
        value = getattr(self, field)
        if not value:
            alias = {
                "openai_api_key": "OPENAI_API_KEY",
                "fal_key": "FAL_KEY",
                "replicate_api_token": "REPLICATE_API_TOKEN",
            }.get(field, field.upper())
            raise MissingCredential(f"{stage} needs {alias}; set it in backend/.env")
        return str(value)

    # --- certification thresholds ---
    # Stability axis.
    settle_seconds: float = 2.0
    max_com_displacement_m: float = 0.01
    max_orientation_drift_deg: float = 2.0

    # Non-zero because mesh
    # discretisation produces sub-millimetre contacts on surfaces flush by
    # design, and a zero-tolerance check would fail every well-modelled drawer.
    # Sensitivity of the reported results to this value is itself a result.
    max_penetration_m: float = 0.002

    # Scale axis.
    max_support_gap_m: float = 0.005
    max_prior_deviation_sigma: float = 3.0

    # Inertial axis. Relative, because the absolute error scales with the object:
    # 1 g on a mug means something quite different to 1 g on a wardrobe.
    max_inertia_rel_error: float = 0.01

    # Cost axis.
    step_time_budget_ms: float = 2.0

    # Repair. A *cost* budget, not a safety net: every accepted repair strictly
    # decreases a violation score bounded below by zero, so the loop terminates on
    # its own and cannot oscillate. This only binds when repairs cascade — fixing
    # one object creating a failure in another — and each round costs one
    # re-certification per proposal, roughly 20 ms on a small scene.
    max_repair_rounds: int = 10

    @property
    def uploads_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def artifacts_dir(self) -> Path:
        """Per-stage cached outputs, keyed by input hash."""
        return self.storage_dir / "artifacts"

    @property
    def scenes_dir(self) -> Path:
        return self.storage_dir / "scenes"

    def storage_relative(self, path: Path | str) -> str:
        """A path under storage_dir, expressed relative to it.

        Everything the browser fetches goes through the `/storage` static mount, so
        a stored absolute filesystem path is unusable to a client — and it also
        breaks the moment the storage directory moves, which it does between a
        developer's machine and a test's temp dir. Storing the relative form and
        letting the client prepend `/storage/` keeps the record portable and the
        URL derivable.

        Paths outside storage_dir are returned unchanged rather than raising: an
        odd path in a record is a smaller problem than a stage that cannot finish.
        """
        resolved = Path(path).resolve()
        try:
            return resolved.relative_to(self.storage_dir.resolve()).as_posix()
        except ValueError:
            return str(path)

    def certification_thresholds(self) -> dict[str, float | int]:
        """The subset of settings a certification result depends on.

        Used as part of the certify stage's cache key, so tightening a threshold
        re-certifies instead of returning the artifact from the looser run. Kept
        explicit rather than dumping the whole Settings: storage paths and CORS
        origins have no bearing on whether a scene passes.
        """
        return {
            "settle_seconds": self.settle_seconds,
            "max_com_displacement_m": self.max_com_displacement_m,
            "max_orientation_drift_deg": self.max_orientation_drift_deg,
            "max_penetration_m": self.max_penetration_m,
            "max_support_gap_m": self.max_support_gap_m,
            "max_prior_deviation_sigma": self.max_prior_deviation_sigma,
            "max_inertia_rel_error": self.max_inertia_rel_error,
            "step_time_budget_ms": self.step_time_budget_ms,
        }

    def ensure_dirs(self) -> None:
        for path in (self.storage_dir, self.uploads_dir, self.artifacts_dir, self.scenes_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
