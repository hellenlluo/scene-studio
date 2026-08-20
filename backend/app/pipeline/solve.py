from app.pipeline.base import PipelineContext
from app.schemas import (
    DepthResult,
    ScaleAnchor,
    SceneGraph,
    SegmentResult,
    SolveResult,
    SolveWeights,
)

__all__ = ["SolveResult", "run"]


def run(
    ctx: PipelineContext,
    graph: SceneGraph,
    depth: DepthResult,
    segments: SegmentResult,
    weights: SolveWeights,
    anchors: list[ScaleAnchor] | None = None,
) -> SolveResult:
    """Coupled optimisation over scale and pose. The core stage.

    Variables: per-object isotropic scale s_i and pose T_i in SE(3). Objective:

        E = w_depth * E_depth    point-to-mesh, backprojected cloud vs posed mesh
          + w_prior * E_prior    Mahalanobis (d_i(s_i) - mu)^2 / sigma^2
          + w_sil   * E_sil      rendered-vs-observed mask IoU
          + w_supp  * E_supp     signed support gap + base-inside-parent-polygon
          + w_pen   * E_pen      sum max(0, penetration)^2 over object pairs

    E_depth and E_prior are the two terms carrying absolute scale, and they are
    fused because their errors are complementary rather than redundant: depth is
    a direct per-object measurement but biased by material and distance, the
    prior is coarse but roughly unbiased since it never sees these pixels. Depth
    alone is wrong-but-precise; the prior alone is right-on-average-but-vague.

    Solve as block coordinate descent, not one monolithic optimisation:

      (a) Gauss-Newton on the smooth terms for scale and pose, via
          scipy.optimize.least_squares. Keep the Jacobian: J^T J is the Hessian
          approximation and its inverse diagonal is per-object scale variance,
          which is what SolveDiagnostics.scale_variance reports.
      (b) evaluate the physics terms in MuJoCo, after inertia.recompute.
      (c) feed the violations back as residuals; iterate to a fixed point.

    A coupled solve that fails to converge is worse than two stages that do, and
    this ordering degrades gracefully to sequential-with-feedback.

    Anchors enter as E_prior terms with sigma -> 0. So do user-pinned values:
    respect SceneObject.is_pinned by holding those variables out of the free set
    rather than penalising them heavily.

    Record the iteration count. If physics feedback measurably improves scale
    over one pass, the coupling thesis is demonstrated rather than asserted, and
    that number is the evidence.
    """
    raise NotImplementedError("solve")
