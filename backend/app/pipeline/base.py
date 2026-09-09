"""Stage framework: typed I/O plus content-addressed artifact caching.

The tail of this pipeline gets re-run constantly during development. Caching each
stage on the hash of its inputs means iterating on `validate` does not re-run
depth estimation every time.
"""

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
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


def prompt_fingerprint(*texts: str) -> str:
    """A short hash of the prompts a stage sends, for use as a cache-key input.

    A VLM stage's output depends on its prompt as surely as on its upstream model,
    and a prompt is not one of its arguments — so without this, editing one changes
    nothing until somebody deletes the artifact by hand. That is not hypothetical:
    the inventory, labelling and verification prompts were each edited during one
    session and each silently served the previous answer, the last one after the
    edit had already been shown to fix the bug it was written for.
    """
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:12]


def source_fingerprint(*modules: ModuleType) -> str:
    """A hash of these modules' source, for use as a cache-key input.

    Settings, prompts and thresholds can be enumerated and keyed on; the code that
    consumes them cannot. Six times in one session a stage was edited, re-run, and
    silently served its previous artifact — twice after the fix had already been
    demonstrated correct by calling the stage directly. The failure mode is
    expensive precisely because it looks like the fix not working.

    Only local stages carry this. `segment` and `reconstruct` are paid, they change
    rarely, and re-running them on a comment edit would cost real money; they stay
    keyed on their explicit inputs.
    """
    payload = "\x00".join(Path(inspect.getfile(m)).read_text() for m in modules)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


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
