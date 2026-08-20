"""Inertial certification: are the mass properties physically possible?

Three cheap checks that each catch a real failure, and all three are easy to get
wrong in ways nothing else notices. MuJoCo will happily simulate a body with a
nonsensical inertia tensor; it just behaves strangely under actuation, which is
the one thing this project is trying to guarantee.

Two judgement calls worth knowing about:

**A non-watertight mesh fails the consistency check outright.** trimesh returns a
volume for an open mesh without complaining, and that volume is meaningless — so
a mass derived from it is meaningless too, however plausible the number looks.
Since every actuation result downstream depends on that mass, "we cannot trust
this" is a failure rather than a caveat. It is also repairable: watertight repair
via manifold3d is exactly the fix.

**A part with no inertial properties at all is skipped, not failed.** That is an
object which has not been through stage 9 yet. Absent is not the same as
inconsistent, and reporting it either way would be a lie — so it produces no
check, and an object where nothing is checkable produces none at all, leaving the
axis NOT_APPLICABLE.
"""

from app.schemas import InertialCheck, InertialProperties, SceneGraph

__all__ = ["run"]


def _consistent(inertial: InertialProperties, rel_tol: float) -> bool:
    """mass == density * volume, to within a relative tolerance."""
    if not inertial.watertight:
        return False
    if inertial.mass_kg <= 0.0 or inertial.volume_m3 <= 0.0:
        # A zero-mass body is a hard model error in MuJoCo, and a zero-volume one
        # means the mesh collapsed somewhere upstream.
        return False
    expected = inertial.density_kg_m3 * inertial.volume_m3
    return abs(inertial.mass_kg - expected) <= rel_tol * inertial.mass_kg


def run(graph: SceneGraph, rel_tol: float = 0.01) -> list[InertialCheck]:
    checks: list[InertialCheck] = []
    for obj in graph.objects:
        present = [p.inertial for p in obj.parts if p.inertial is not None]
        if not present:
            continue  # stage 9 has not run for this object; not checkable

        consistent = all(_consistent(i, rel_tol) for i in present)
        definite = all(i.is_positive_definite for i in present)
        triangle = all(i.satisfies_triangle_inequality for i in present)

        checks.append(
            InertialCheck(
                object_id=obj.object_id,
                mass_density_volume_consistent=consistent,
                positive_definite=definite,
                triangle_inequality=triangle,
                passed=consistent and definite and triangle,
            )
        )
    return checks
