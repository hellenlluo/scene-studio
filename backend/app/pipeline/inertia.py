"""Collision proxies and mass properties.

Mass is `density x volume`, and both halves have a trap in them.

**Volume comes from the visual mesh, never the collision proxy.** Measured on a
dining table modelled as a slab and four legs:

| volume source | volume | mass at wood 700 |
|---|---|---|
| bounding box | 1.013 m3 | 709 kg |
| convex hull | 0.962 m3 | 674 kg |
| **visual mesh** | **0.064 m3** | **45 kg** |

A real dining table weighs about 40 kg. The hull is 95% of the bounding box —
taking volume from the proxy is a 15x error on any open shape, and open shapes are
most of the furniture in a room. The two meshes exist for different purposes and
this is the one that means mass. MuJoCo would otherwise derive inertia from the
collision geoms, which is why `app.export.mjcf` emits an explicit `<inertial>`
block whenever these properties are populated.

**Density is keyed on material, not category.** Once volume is a real mesh volume,
plain material density lands within about 12% without calibration. That only holds
against mesh volume — against a bounding box it overestimates by an order of
magnitude, which is what makes the rule above load-bearing rather than tidy.

**Known limitation: containers come out heavy.** Single-view reconstruction
returns an outer shell, so a mug is a solid ceramic cylinder — 1.5 kg against a
real 0.35 kg, roughly 4x. The defining feature of a container is the hole, and the
hole is unobservable from a photo of it. Accepted for now rather than papered over
with a fudge factor: the error is systematic, it only affects container categories,
and nothing in the current certification depends on mass being right. It starts
mattering when actuation does.

Until stage 4 produces meshes every part falls back to an OBB, so *every* mass is
currently the bounding-box number and is too heavy. That is the same limitation,
temporarily applied to everything.
"""

from app.pipeline.base import PipelineContext
from app.schemas import Material, SceneGraph

__all__ = ["MATERIAL_DENSITY_KG_M3", "SceneGraph", "recompute", "run"]

# Solid-material densities. Correct against a *mesh* volume; see the module
# docstring for why they are badly wrong against a bounding box.
#
# `metal` and `wood` span a wide real range (aluminium 2700 to steel 7800; balsa
# 150 to oak 900). The values here are the common case for household objects, and
# a finer vocabulary would be asking the VLM to judge something it cannot see.
MATERIAL_DENSITY_KG_M3: dict[Material, float] = {
    Material.WOOD: 650.0,
    Material.METAL: 7800.0,
    Material.PLASTIC: 1100.0,
    Material.GLASS: 2500.0,
    Material.CERAMIC: 2400.0,
    Material.FABRIC: 300.0,
    Material.STONE: 2700.0,
    Material.PAPER: 800.0,
    # Water, which is also MuJoCo's own default for a geom with no density.
    Material.OTHER: 1000.0,
}


def run(ctx: PipelineContext, graph: SceneGraph) -> SceneGraph:
    """Assign collision proxies and the inertial priors.

    Convex decomposition via CoACD over each visual mesh, preceded by
    watertightness repair — trimesh will hand you a non-watertight mesh and a
    meaningless volume without complaining, and mass computed from that volume is
    meaningless too. Record InertialProperties.watertight either way, because the
    inertial certification axis treats an untrustworthy mass as a failure rather
    than a caveat.

    Compute `volume_m3` from the **visual** mesh even though the collision geometry
    is the proxy. See the module docstring.

    This runs before solve, not after certify: the solver's physics block steps
    MuJoCo, and MuJoCo cannot settle a body with no mass.
    """
    raise NotImplementedError("inertia")


def recompute(graph: SceneGraph) -> SceneGraph:
    """Refresh mass and inertia from density and the current scale.

    Volume goes as scale^3, so every mass in the scene is stale the moment the
    solver moves a scale. Pure function, called inside the solve loop; the
    density assigned in run() is what stays fixed.
    """
    raise NotImplementedError("inertia recompute")
