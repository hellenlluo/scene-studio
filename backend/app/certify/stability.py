from app.config import Settings
from app.schemas import SceneGraph, StabilityCheck

__all__ = ["run"]


def run(graph: SceneGraph, settings: Settings) -> list[StabilityCheck]:
    """Settle the scene under gravity in MuJoCo and score each object.

    Build MJCF via app.export.mjcf so validation and export can never drift
    apart, record initial poses, step for settle_seconds, then compare. An object
    passes when COM displacement, orientation drift and t=0 penetration are all
    inside the configured thresholds.

    Measure penetration before stepping. Once the solver starts pushing bodies
    apart the initial overlap is gone and the reconstruction error that caused it
    is no longer observable.
    """
    raise NotImplementedError("certify stability")
