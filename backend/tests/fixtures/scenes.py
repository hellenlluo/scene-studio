"""Hand-authored scenes, in metres, z up, gravity along -Z.

These exist so the back half of the pipeline — MJCF, certification, repair,
export — is testable before any perception stage runs. No GPU, no API key, no
mesh files: every part falls back to its OBB box, which is the tier the pipeline
guarantees anyway.

Two of the three are deliberately broken, and they model what a bad
reconstruction actually produces rather than an abstract failure. A validator
tested only on scenes that pass is one you have no evidence detects anything.
"""

from app.schemas import (
    AssetFrame,
    DimensionPrior,
    InertialProperties,
    ObjectLabel,
    PartGeometry,
    SceneGraph,
    SceneObject,
    Vec3,
)

__all__ = [
    "cabinet_overlapping_table",
    "kitchen",
    "mug_floating_above_table",
    "mug_overhanging_its_support",
    "mug_sunk_into_table",
]


def _box_inertial(dims: Vec3, density: float) -> InertialProperties:
    """Uniform-density box: the closed form MuJoCo would compute for the same shape."""
    x, y, z = dims
    volume = x * y * z
    mass = volume * density
    k = mass / 12.0
    return InertialProperties(
        density_kg_m3=density,
        volume_m3=volume,
        mass_kg=mass,
        inertia_diag=(k * (y * y + z * z), k * (x * x + z * z), k * (x * x + y * y)),
        watertight=True,
    )


def _part(
    part_id: str,
    dims: Vec3,
    origin: Vec3 = (0.0, 0.0, 0.0),
    parent: str | None = None,
    density: float = 600.0,
) -> PartGeometry:
    return PartGeometry(
        part_id=part_id,
        name=part_id,
        parent_part_id=parent,
        dims_m=dims,
        origin_m=origin,
        inertial=_box_inertial(dims, density),
    )


def _object(
    object_id: str,
    category: str,
    parts: list[PartGeometry],
    position: Vec3,
    prior_dims: Vec3,
    supported_by: str | None = None,
) -> SceneObject:
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(
            object_id=object_id,
            category=category,
            prior=DimensionPrior(dims_m=prior_dims, sigma_m=(0.05, 0.05, 0.05)),
            support_parent=supported_by,
        ),
        frame=AssetFrame(source="authored"),
        parts=parts,
        position_m=position,
        supported_by=supported_by,
    )


def kitchen() -> SceneGraph:
    """A scene that should certify on every axis that has a validator.

    A table, a mug resting on it, and a cabinet on the floor. Every object is a
    single solid part, which is what reconstruction produces.
    """
    table = _object(
        "table",
        "table",
        [_part("top", (1.2, 0.75, 0.75), density=700.0)],
        position=(0.0, 0.0, 0.375),
        prior_dims=(1.2, 0.75, 0.75),
    )
    # Base at z = 0.75, the tabletop, so centre sits at 0.75 + 0.10/2.
    mug = _object(
        "mug",
        "mug",
        [_part("body", (0.09, 0.09, 0.10), density=400.0)],
        position=(0.3, 0.0, 0.80),
        prior_dims=(0.09, 0.09, 0.10),
        supported_by="table",
    )
    cabinet = _object(
        "cabinet",
        "cabinet",
        [_part("body", (0.6, 0.6, 0.92), density=550.0)],
        position=(1.2, 0.0, 0.46),
        prior_dims=(0.6, 0.6, 0.9),
    )
    return SceneGraph(objects=[table, mug, cabinet])


def mug_floating_above_table() -> SceneGraph:
    """A mug hovering 25 cm over the table — what an over-estimated depth looks
    like once it reaches physics.

    It falls, so COM displacement fails while initial penetration stays clean. The
    two signals are independent and this pins that.
    """
    graph = kitchen()
    mug = graph.get("mug")
    x, y, z = mug.position_m
    mug.position_m = (x, y, z + 0.25)
    return graph


def mug_sunk_into_table() -> SceneGraph:
    """A mug buried 5 cm inside the tabletop — under-estimated depth, or two
    objects whose scales disagree.

    The interpenetration case, and the reason overlap is measured at t=0: the
    solver ejects the mug within a few steps, so by the end of settling the
    evidence that diagnosed the error no longer exists.
    """
    graph = kitchen()
    mug = graph.get("mug")
    x, y, z = mug.position_m
    mug.position_m = (x, y, z - 0.05)
    return graph


def cabinet_overlapping_table() -> SceneGraph:
    """Two floor-standing objects occupying the same space — what disagreeing
    per-object scales look like once both are placed.

    Neither supports the other, so snapping to a support plane cannot help. The
    fix has to be a lateral push, and choosing *which* body to move is the part
    that makes repair underdetermined.
    """
    graph = kitchen()
    cabinet = graph.get("cabinet")
    _, y, z = cabinet.position_m
    cabinet.position_m = (0.7, y, z)  # 0.2 m into the table
    return graph


def mug_overhanging_its_support() -> SceneGraph:
    """A sound object whose *bounding box* centre hangs off its support.

    Not a broken scene — the third one that is not. A heavy body sitting well
    inside the tabletop with a light handle reaching 480 mm past its edge is
    resting, stable, and ordinary, and the only thing wrong with it is what a
    bounding box says about it: the box centre lands 193 mm beyond the table while
    the centre of mass stays 50 mm inside it.

    This is `room.png`'s lamp in miniature — shade overhanging base — and
    `room2.png`'s armchair in reverse, where the box centre is over the rug and the
    mass is not. The toppling condition is about weight, so one of the two centres
    is the right one to test and it is not the box.
    """
    graph = kitchen()
    mug = graph.get("mug")
    mug.position_m = (0.55, 0.0, 0.80)
    mug.parts = [
        _part("body", (0.09, 0.09, 0.10), density=8000.0),
        _part("handle", (0.50, 0.04, 0.02), origin=(0.28, 0.0, 0.0), parent="body", density=30.0),
    ]
    return graph


def mug_adrift_from_its_support() -> SceneGraph:
    """Off its support laterally, and touching nothing at all.

    The gap between repair's two strategies, and `room2.png`'s plate in miniature:
    that plate sits 138 mm outside the side table it is recorded as resting on and
    53 mm below its top, so it touches neither the table nor anything else. With no
    contact patch there is no hull for `_slide_onto_support` to aim at, and
    `_snap_to_support` translates in z alone and so cannot bring it back over the
    table. Measured before the fallback existed, five repair rounds proposed not one
    action for it.

    Distinct from `mug_overhanging_its_support`, which is a *sound* scene the box
    test misreads. This one is genuinely broken.
    """
    graph = kitchen()
    mug = graph.get("mug")
    # Table top spans x [-0.6, 0.6] at z = 0.75. Clear of it in x, level with it in
    # z, so nothing is under the mug and nothing is beside it either.
    mug.position_m = (0.9, 0.0, 0.80)
    return graph
