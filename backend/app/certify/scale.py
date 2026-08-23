"""Scale certification: are the fitted sizes plausible, and do things rest on things?

Three checks, all computed from the scene graph alone. No depth map, no MuJoCo,
no I/O — which makes this the cheapest axis and the only one that can run on a
graph with no geometry files at all.

**Depth is deliberately absent.** It is an *input* to the thing being checked: the
solver minimises E_depth, so a certifier that also scored against depth would let
a solve which overfit it certify beautifully while being wrong.

Being straight about what this is, though: every check here corresponds to a term
the solver was already minimising, so it is not validation against outside truth.
It is a **feasibility check** — did the optimiser actually achieve the constraints
it was given, or did it trade one away to satisfy another? An object that ended up
four sigma from its class prior because depth pulled hard is a genuine warning,
and that is what this catches. Independent validation needs a tape measure, and
that is evaluation rather than certification.
"""

import numpy as np

from app.geometry import world_aabb
from app.schemas import ScaleCheck, SceneGraph, Vec3

__all__ = ["run"]


def _deviation_sigma(fitted: Vec3, prior_dims: Vec3, prior_sigma: Vec3) -> Vec3:
    """(fitted - prior) / sigma, per axis.

    A sigma of zero is a hard constraint — a scale anchor, or a prior the VLM
    claimed certainty about — so any mismatch has to read as a large deviation
    rather than a division by zero.
    """
    out = []
    for f, p, s in zip(fitted, prior_dims, prior_sigma, strict=True):
        out.append((f - p) / max(s, 1e-9))
    return (out[0], out[1], out[2])


def run(
    graph: SceneGraph,
    max_deviation_sigma: float,
    max_support_gap_m: float,
) -> list[ScaleCheck]:
    bounds = {obj.object_id: world_aabb(obj) for obj in graph.objects}

    checks: list[ScaleCheck] = []
    for obj in graph.objects:
        prior = obj.label.prior
        # No prior means there is nothing to deviate from. Reporting zeros would
        # claim the object matched a prior it never had, which reads as a pass.
        # Zeros are correct here for a different reason: the deviation *component*
        # is vacuously satisfied, and the support checks below still apply.
        deviation = (
            _deviation_sigma(obj.dims_m, prior.dims_m, prior.sigma_m)
            if prior is not None
            else (0.0, 0.0, 0.0)
        )

        low, high = bounds[obj.object_id]
        parent = graph.get(obj.supported_by) if obj.supported_by else None

        if parent is None:
            # Supported by the floor, which is unbounded — so containment is
            # trivially true and only the gap is informative. An object naming a
            # support that is not in the graph lands here too; that is a
            # reconcile bug, and reporting it as a floor contact at least makes
            # the object's height visible rather than skipping it silently.
            support_top = graph.floor_height_m
            inside = True
        else:
            parent_low, parent_high = bounds[parent.object_id]
            support_top = float(parent_high[2])
            # Containment is tested on the footprint *centre*, not the whole
            # footprint. The centre-over-support test is the toppling condition
            # and it is what a physically wrong placement violates; requiring
            # full containment would fail a laptop legitimately overhanging a
            # side table, and gross size mismatch is already caught by the
            # deviation term above.
            centre_xy = (low[:2] + high[:2]) / 2.0
            inside = bool(
                np.all(centre_xy >= parent_low[:2]) and np.all(centre_xy <= parent_high[:2])
            )

        # Positive means floating, negative means sunk into the support.
        gap = float(low[2]) - support_top

        checks.append(
            ScaleCheck(
                object_id=obj.object_id,
                deviation_sigma=deviation,
                support_gap_m=gap,
                base_inside_parent=inside,
                passed=(
                    all(abs(d) <= max_deviation_sigma for d in deviation)
                    and abs(gap) <= max_support_gap_m
                    and inside
                ),
            )
        )
    return checks
