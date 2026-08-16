"""Stage framework: typed I/O plus content-addressed artifact caching.

The tail of this pipeline gets re-run constantly during development. Caching each
stage on the hash of its inputs means iterating on `validate` does not re-run
depth estimation every time.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.schemas import StageName


@dataclass
class PipelineContext:
    job_id: str
    image_path: Path
    settings: Settings

    @classmethod
    def create(cls, job_id: str, image_path: Path) -> "PipelineContext":
        settings = get_settings()
        settings.ensure_dirs()
        return cls(job_id=job_id, image_path=image_path, settings=settings)

    def workdir(self) -> Path:
        path = self.settings.scenes_dir / self.job_id
        path.mkdir(parents=True, exist_ok=True)
        return path


class Stage[In: BaseModel | None, Out: BaseModel](Protocol):
    name: StageName

    def run(self, ctx: PipelineContext, data: In) -> Out: ...


def _image_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_key(ctx: PipelineContext, stage: StageName, data: BaseModel | None) -> str:
    """Hash of (image, stage, upstream result) — changes whenever any input does."""
    payload = json.dumps(
        {
            "image": _image_fingerprint(ctx.image_path),
            "stage": str(stage),
            "input": data.model_dump(mode="json") if data is not None else None,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cached[T: BaseModel](ctx: PipelineContext, key: str, model: type[T]) -> T | None:
    path = ctx.settings.artifacts_dir / f"{key}.json"
    if not path.exists():
        return None
    return model.model_validate_json(path.read_text())


def store_cached(ctx: PipelineContext, key: str, result: BaseModel) -> None:
    path = ctx.settings.artifacts_dir / f"{key}.json"
    path.write_text(result.model_dump_json(indent=2))
