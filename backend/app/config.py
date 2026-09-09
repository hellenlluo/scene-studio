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

    # --- stage 4: reconstruction ---
    max_occluder_depth_step_m: float = Field(
        default=0.05,
        description="How far apart two surfaces may be at their shared boundary and "
        "still count as touching, in `occlusion`.\n\n"
        "This is the check that separates a covering from a foreground object. Both "
        "sever the mask behind them and both can be labelled as resting on it, so "
        "neither geometry nor the support relation can tell them apart: a small plant "
        "standing on a bookshelf shelf is `supported_by` that bookshelf and splits it "
        "exactly as a throw splits an armchair. What differs is depth. A throw lies "
        "*on* the chair, so the two are continuous across the seam; a plant stands "
        "clear of the shelves behind it, so depth steps by the standoff distance.\n\n"
        "5 cm is generous for contact — it has to absorb the thickness of a folded "
        "textile and depth noise at a silhouette edge, where every model is least "
        "reliable — and still far below any real standoff, which is tens of "
        "centimetres.",
    )

    max_mesh_faces: int = Field(
        default=120_000,
        description="Decimation target for reconstructed meshes. SAM 3D returns "
        "hundreds of thousands of faces — one side table came back at 1.1M faces "
        "and 22 MB, which is unusable in a browser and slow in MuJoCo.\n\n"
        "The binding constraint is not file size but `max_penetration_m`: mesh error "
        "and the certification tolerance are the same kind of quantity, so a mesh "
        "wrong by more than delta can cause spurious penetration failures or hide "
        "real ones. Measured on a ~1.2 m object, mean surface deviation is 0.12 mm "
        "at 20k faces, 0.36 mm at 5k, 0.96 mm at 1k, and 2.2 mm at 500 — where it "
        "crosses the 2 mm tolerance and volume error jumps to 5%. 5k leaves a "
        "comfortable margin at 0.09 MB per object.\n\n"
        "Raised from 5k to 40k when textured reconstruction was turned on, because "
        "decimation discards UVs: any mesh crossing this threshold loses its material, "
        "so the number stopped being a pure size/accuracy trade-off and became the "
        "line between a textured object and a grey one.\n\n"
        "The headroom is real rather than generous. At ProxyTier.OBB collision uses "
        "boxes, so face count does not enter the physics at all — measured 0.022 ms "
        "per step against a 2 ms budget. And textured output is small to begin with: "
        "SAM 3D returns a low-poly mesh plus a 1024x1024 base colour map instead of "
        "baking detail into geometry, so the same objects that came back at 138k and "
        "415k faces untextured arrive at 7k and 12k. On room.jpg, 8 of 9 objects fit "
        "under 20k and the ninth needed 31k; 40k covers the observed spread with room "
        "to spare, at about 1.2 MB per object.\n\n"
        "Raised again from 40k to 120k after `room2.png`, where it stopped being a "
        "fidelity setting and started breaking physics. One object — an armchair — "
        "crossed 40k and was decimated, and as `_load_glb` records, decimation opens "
        "the surface: the mesh arrived watertight and came out with 3209 boundary "
        "loops. CoACD then produced no hull for the lower 27.9% of it, so the chair "
        "had no legs to stand on, fell 589 mm, took the pillow resting on it down "
        "992 mm, and left the scene unable to settle. Every other object in either "
        "scene has full collision coverage to its base. A cap that silently deletes "
        "an object\'s contact surface is worse than a large file.\n\n"
        "The proper fix is to decompose collision from the *undecimated* mesh, so "
        "visual budget and physical correctness stop being the same number. This "
        "raise buys headroom rather than fixing that coupling.\n\n"
        "This will need revisiting when a mesh collision tier exists, at which point "
        "the max_penetration_m argument above binds again.",
    )

    # --- stage 5.5: reconstruction verification ---
    snapshot_max_dim_px: int = Field(
        default=900,
        description="Longest edge of the rendered comparison snapshot. Matched to "
        "`annotate.py`'s own scale reference rather than chosen independently — "
        "both exist to be read by the same family of vision models, so there is no "
        "reason for them to disagree about how much resolution that needs.",
    )

    # --- stage 9: inertia and collision proxies ---
    coacd_threshold: float = Field(
        default=0.05,
        description="CoACD concavity threshold: the residual concavity it stops "
        "refining at. Lower means more, tighter pieces and a slower `mean_step_time_ms` "
        "on the cost axis, which is the trade this number exists to expose. 0.05 is "
        "CoACD's own default and the reference point for the proxy-tier table.",
    )
    max_convex_pieces: int = Field(
        default=24,
        description="Cap on hulls per part. Uncapped, a decomposition of a "
        "reconstructed sofa runs to hundreds of pieces and the cost axis fails for "
        "a reason that has nothing to do with the object.",
    )
    max_collision_faces: int = Field(
        default=10_000,
        description="Decimation target for the mesh handed to CoACD, separate from "
        "`max_mesh_faces` because the two are spent on different things. The visual "
        "cap is high to keep UVs; collision geometry has no UVs and CoACD's runtime "
        "grows with the input, so this one stays low. The output pieces are convex "
        "hulls either way, and a hull does not remember the faces it came from.",
    )

    # --- stage 6: support surfaces ---
    support_burial_slack_m: float = Field(
        default=0.10,
        description="How far above a child's base a surface may sit and still count "
        "as the one it rests on.\n\n"
        "`SupportHeights.under` picks the highest surface at or below the child, "
        "which is what makes a shelf work: the topmost surface in a bookshelf's "
        "column is the top of the unit, and a book on the middle shelf rests on "
        "neither that nor the shelf above it. But reconstruction routinely buries "
        "an object in its support by a few centimetres before the solver runs, and "
        "a strict cutoff would then skip past the shelf it is buried in and answer "
        "with the one below — turning a 20 mm error into a whole shelf of error. "
        "10 cm is far more burial than reconstruction produces and far less than "
        "any shelf spacing, so it separates the two cases cleanly.",
    )

    support_grid_cell_m: float = Field(
        default=0.02,
        description="Cell size of the top-surface height grid the solver reads an "
        "object's support height out of. 2 cm is chosen against "
        "`max_support_gap_m` (5 mm) rather than for looks: the grid answers *where* "
        "a surface is, and the solver then closes the last millimetres itself, so "
        "the cell only has to be small enough not to miss a surface feature an "
        "object could rest on. Finer costs rays quadratically.",
    )
    support_grid_max_cells: int = Field(
        default=128,
        description="Cap per axis, so a large support degrades to a coarser cell "
        "rather than to a quarter-million rays. A 2.5 m rug at 2 cm is 125 cells, "
        "so this binds only on something bigger than the room.",
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
    support_claim_margin_m: float = Field(
        default=0.05,
        description="How much closer a geometric candidate must be than the labelled "
        "one before `reconcile` overrules the label about what an object rests on.\n\n"
        "Stage 3 sees the photo and answers semantically; this stage sees geometry "
        "and answers by nearest surface. Nearest-surface alone is wrong on a near "
        "tie, because heights collide: measured on `room2.png`, the VLM correctly "
        "put a book on the side table and reconcile moved it onto the *teacup* "
        "beside it, whose rim happens to sit at the book's base height. A book "
        "resting on a teacup is not a thing.\n\n"
        "So geometry vetoes rather than replaces — it overrules the label only when "
        "clearly better, and a label that geometry merely fails to confirm is kept. "
        "5 cm is well above the disagreement between two surfaces that are really "
        "the same contact and well below a genuine mistake, where the labelled "
        "parent is usually the wrong height entirely.",
    )

    max_repair_rounds: int = 10

    # Beyond this, the support *relation* is more likely wrong than the position,
    # and moving the object would destroy information rather than correct it. The
    # motivating case: a wall-mounted picture read as floor-supported, which repair
    # would otherwise "fix" by snapping it 1.5 m down onto the carpet.
    max_snap_m: float = 0.5

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

    def support_settings(self) -> dict[str, float | int]:
        """The settings that change where a support surface is measured to be.

        A cache-key input for every stage that measures one — reconcile, solve and
        certify all call `support.build`, and all three would otherwise return an
        artifact computed under a different definition of "the surface under this
        object". That is not hypothetical: `support_burial_slack_m` was added to fix
        books resting on the wrong shelf, and without this the fix changed nothing
        on a rerun because every stage downstream of segmentation was cached.
        """
        return {
            "support_grid_cell_m": self.support_grid_cell_m,
            "support_grid_max_cells": self.support_grid_max_cells,
            "support_burial_slack_m": self.support_burial_slack_m,
        }

    def certification_thresholds(self) -> dict[str, float | int]:
        """The subset of settings a certification result depends on.

        Used as part of the certify stage's cache key, so tightening a threshold
        re-certifies instead of returning the artifact from the looser run. Kept
        explicit rather than dumping the whole Settings: storage paths and CORS
        origins have no bearing on whether a scene passes.
        """
        return {
            **self.support_settings(),
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
