from app.pipeline.base import PipelineContext
from app.schemas import GeometryResult, RepairAction, SceneLayout, ValidationResult

__all__ = ["run"]


def run(
    ctx: PipelineContext,
    geometry: GeometryResult,
    layout: SceneLayout,
    validation: ValidationResult,
) -> tuple[SceneLayout, ValidationResult, list[RepairAction]]:
    """Compute the minimal correction that makes a failing scene valid.

    Cheapest corrections first — snap to the inferred support plane, then push
    apart residual penetrations, then rescale only if an object's scale residual
    was already large. Revalidate after applying, and keep a repair only if it
    actually improved that object's result; a repair that makes things worse
    should be discarded rather than reported.

    Returns the repaired layout, its revalidation, and the actions applied.
    """
    raise NotImplementedError("repair")
