"""Cost certification: is the scene fast enough to actually simulate?

An axis rather than a footnote, because a convex decomposition that certifies on
every other axis but steps too slowly to train in is not simulation-ready. It is
also the measurement behind the proxy-tier trade: fidelity buys accuracy and costs
step time, and the exchange rate is a result worth reporting rather than a
parameter to guess at.

Timing is taken on a **settled** scene. At t=0 objects are barely touching and the
contact set is at its smallest, so timing there would report the cheapest moment
the scene ever has. The steady state — everything resting, contacts established —
is what a training run actually pays, so the warm-up doubles as the settle.
"""

import time

from app.schemas import CostCheck, ProxyTier, SceneGraph

__all__ = ["run"]

WARMUP_STEPS = 200
MEASURE_STEPS = 200

# Coarsest to finest. The most expensive tier present is what drives step time,
# so that is the one the certificate reports for the scene.
_TIER_ORDER = [ProxyTier.OBB, ProxyTier.CONVEX_HULL, ProxyTier.DECOMPOSED]


def scene_tier(graph: SceneGraph) -> ProxyTier:
    tiers = [part.proxy_tier for obj in graph.objects for part in obj.parts]
    if not tiers:
        return ProxyTier.OBB
    return max(tiers, key=_TIER_ORDER.index)


def run(
    graph: SceneGraph,
    budget_ms: float,
    warmup_steps: int = WARMUP_STEPS,
    measure_steps: int = MEASURE_STEPS,
) -> CostCheck:
    import mujoco

    from app.export import mjcf

    model = mujoco.MjModel.from_xml_string(mjcf.build_xml(graph))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    # Settle first, so the measurement covers the steady-state contact set rather
    # than the sparse one the scene starts with.
    for _ in range(warmup_steps):
        mujoco.mj_step(model, data)

    started = time.perf_counter()
    for _ in range(measure_steps):
        mujoco.mj_step(model, data)
    elapsed = time.perf_counter() - started

    mean_ms = (elapsed / measure_steps) * 1e3
    return CostCheck(
        proxy_tier=scene_tier(graph),
        mean_step_time_ms=mean_ms,
        budget_ms=budget_ms,
        passed=mean_ms <= budget_ms,
    )
