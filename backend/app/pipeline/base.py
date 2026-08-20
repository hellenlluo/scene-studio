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


def _dump(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_dump(v) for v in value]
    return value


def cache_key(ctx: PipelineContext, stage: StageName, *inputs: object) -> str:
    """Hash of (image, stage, every input) — changes whenever any of them does.

    Variadic and not just "the upstream result" because several stages depend on
    more than their predecessor: solve reads the weights and the anchors, certify
    reads the thresholds. Keying only on the upstream model means a stage silently
    returns a stale artifact when one of those changes — which would break the
    scale-anchor interaction in the least visible way possible, by looking like
    it worked.
    """
    payload = json.dumps(
        {
            "image": _image_fingerprint(ctx.image_path),
            "stage": str(stage),
            "inputs": [_dump(i) for i in inputs],
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
