from app.pipeline.base import PipelineContext
from app.schemas import GeometryResult, SceneLayout, ValidationResult

__all__ = ["ValidationResult", "run"]


def run(ctx: PipelineContext, geometry: GeometryResult, layout: SceneLayout) -> ValidationResult:
    """Settle the scene under gravity in MuJoCo and score each object.

    Build MJCF, record initial poses, step for settings.settle_seconds, then
    compare. An object passes when its centre-of-mass displacement, orientation
    drift, and t=0 penetration depth are all within the configured thresholds.

    Measure penetration before stepping — once the solver starts pushing bodies
    apart, the initial overlap is gone and the reconstruction error that caused
    it is no longer observable.
    """
    raise NotImplementedError("validate")
