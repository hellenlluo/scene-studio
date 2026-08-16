from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SCENESTUDIO_")

    database_url: str = f"sqlite:///{BACKEND_ROOT / 'storage' / 'scenestudio.db'}"
    storage_dir: Path = BACKEND_ROOT / "storage"
    cors_origins: list[str] = ["http://localhost:5173"]

    # Physics validation thresholds. See scratch/idea.md.
    settle_seconds: float = 2.0
    max_com_displacement_m: float = 0.01
    max_orientation_drift_deg: float = 2.0
    max_penetration_m: float = 0.002

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

    def ensure_dirs(self) -> None:
        for path in (self.storage_dir, self.uploads_dir, self.artifacts_dir, self.scenes_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
