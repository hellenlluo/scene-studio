from app.schemas import ScaleCheck, SceneGraph

__all__ = ["run"]


def run(graph: SceneGraph) -> list[ScaleCheck]:
    """Do the fitted dimensions agree with the priors, and do contacts close?

    Passes when each axis is within its stated tolerance in sigma and every
    support contact closes within 5 mm with the child's base polygon inside the
    parent's. The base-inside-parent test is the part that constrains *relative*
    scale even where absolute scale is still free: a mug whose base overhangs the
    tabletop is a scale error, not just a placement one.
    """
    raise NotImplementedError("certify scale")
