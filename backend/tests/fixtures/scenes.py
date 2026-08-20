"""Hand-authored scenes, in metres, z up, gravity along -Z.

These exist so the back half of the pipeline — MJCF, certification, repair,
export — is testable before any perception stage runs. No GPU, no API key, no
mesh files: every part falls back to its OBB box, which is the tier the pipeline
guarantees anyway.

Two of the three scenes are deliberately broken. A validator tested only on
scenes that pass is a validator you have no evidence detects anything, and the
kinematic axis in particular fails silently and completely when it is
misconfigured — so the broken fixtures are the ones carrying their weight.

**Cabinets are modelled as five welded panels, not one solid box.** A solid box
carcass would contain its own drawer at every joint value, so the sweep would
report a permanent penetration and the scene could never certify. That is not an
artefact of the fixture: `ProxyTier.OBB` genuinely cannot represent a hollow
container, so any real cabinet that never gets past OBB will fail its kinematic
axis for reasons that have nothing to do with its joints. Panels are how a convex
decomposition would represent it, and welded child parts are already in the
schema for exactly this shape of problem.
"""

from app.schemas import (
    AssetFrame,
    DimensionPrior,
    InertialProperties,
    Joint,
    JointDynamics,
    JointLimits,
    JointType,
    ObjectLabel,
    PartGeometry,
    Route,
    SceneGraph,
    SceneObject,
    Vec3,
)

__all__ = [
    "cabinet_door_into_table",
    "cabinet_with_overlong_drawer",
    "kitchen",
    "rigid_kitchen",
]

PANEL = 0.02  # carcass panel thickness


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
    welded: bool = False,
) -> PartGeometry:
    return PartGeometry(
        part_id=part_id,
        name=part_id,
        parent_part_id=parent,
        dims_m=dims,
        origin_m=origin,
        inertial=_box_inertial(dims, density),
        welded=welded,
    )


def _object(
    object_id: str,
    category: str,
    parts: list[PartGeometry],
    position: Vec3,
    prior_dims: Vec3,
    joints: list[Joint] | None = None,
    supported_by: str | None = None,
) -> SceneObject:
    joints = joints or []
    return SceneObject(
        object_id=object_id,
        label=ObjectLabel(
            object_id=object_id,
            category=category,
            route=Route.ARTICULATED if joints else Route.RIGID,
            prior=DimensionPrior(dims_m=prior_dims, sigma_m=(0.05, 0.05, 0.05)),
            support_parent=supported_by,
        ),
        frame=AssetFrame(source="authored"),
        parts=parts,
        joints=joints,
        position_m=position,
        supported_by=supported_by,
        route_taken=Route.ARTICULATED if joints else Route.RIGID,
    )


def _table() -> SceneObject:
    """1.2 x 0.75 x 0.75 slab. Its top surface sits at z = 0.75."""
    return _object(
        "table",
        "table",
        [_part("top", (1.2, 0.75, 0.75), density=700.0)],
        position=(0.0, 0.0, 0.375),
        prior_dims=(1.2, 0.75, 0.75),
    )


def _mug() -> SceneObject:
    """Resting on the table: base at z = 0.75, so centre at 0.75 + 0.10/2."""
    return _object(
        "mug",
        "mug",
        [_part("body", (0.09, 0.09, 0.10), density=400.0)],
        position=(0.3, 0.0, 0.80),
        prior_dims=(0.09, 0.09, 0.10),
        supported_by="table",
    )


def _carcass(depth: float = 0.6) -> list[PartGeometry]:
    """Five welded panels enclosing a 0.6 x `depth` x 0.9 volume, open toward +y.

    The bottom panel is the root; the rest hang off it as welded children, which
    is both how a decomposition would express it and how the schema represents a
    part that has collapsed into its parent.
    """
    w, h = 0.6, 0.9
    half_d, half_h = depth / 2.0, h / 2.0
    return [
        _part("bottom", (w, depth, PANEL), (0.0, 0.0, -half_h)),
        _part("top", (w, depth, PANEL), (0.0, 0.0, half_h), parent="bottom", welded=True),
        _part("back", (w, PANEL, h), (0.0, -half_d, 0.0), parent="bottom", welded=True),
        _part("left", (PANEL, depth, h), (-w / 2, 0.0, 0.0), parent="bottom", welded=True),
        _part("right", (PANEL, depth, h), (w / 2, 0.0, 0.0), parent="bottom", welded=True),
    ]


def _drawer_joint(joint_id: str, child: str, travel: float) -> Joint:
    """Prismatic, sliding out along +y (the open face)."""
    return Joint(
        joint_id=joint_id,
        name=joint_id,
        type=JointType.PRISMATIC,
        parent_part_id="bottom",
        child_part_id=child,
        axis=(0.0, 1.0, 0.0),
        origin_m=(0.0, 0.0, 0.0),
        limits=JointLimits(lower=0.0, upper=travel),
        dynamics=JointDynamics(damping=5.0, friction_loss=2.0),
    )


def kitchen() -> SceneGraph:
    """A scene that should certify on every axis.

    Table, a mug resting on it, and a two-drawer cabinet whose drawers fit inside
    their carcass and slide their full travel without touching anything.
    """
    drawer_dims = (0.5, 0.5, 0.3)
    cabinet = _object(
        "cabinet",
        "cabinet",
        [
            *_carcass(),
            _part("drawer_top", drawer_dims, (0.0, 0.0, 0.2), parent="bottom"),
            _part("drawer_bottom", drawer_dims, (0.0, 0.0, -0.15), parent="bottom"),
        ],
        position=(1.2, 0.0, 0.46),
        prior_dims=(0.6, 0.6, 0.9),
        joints=[
            _drawer_joint("drawer_top_slide", "drawer_top", 0.4),
            _drawer_joint("drawer_bottom_slide", "drawer_bottom", 0.4),
        ],
    )
    return SceneGraph(objects=[_table(), _mug(), cabinet])


def rigid_kitchen() -> SceneGraph:
    """The same room with articulation switched off — what the pipeline builds today.

    Every object is a single solid part with no joints, which is what the rigid
    branch produces. Use this for the stability, inertial, scale and cost axes;
    the articulated fixtures exist for the kinematic axis, which is dormant while
    `enable_articulation` is False.

    Note the cabinet is one solid box here and that is *fine*. A solid carcass is
    only a problem when something has to move inside it, so the hollow-container
    question does not arise in a rigid scene at all.
    """
    cabinet = _object(
        "cabinet",
        "cabinet",
        [_part("body", (0.6, 0.6, 0.92), density=550.0)],
        position=(1.2, 0.0, 0.46),
        prior_dims=(0.6, 0.6, 0.9),
    )
    return SceneGraph(objects=[_table(), _mug(), cabinet])


def cabinet_with_overlong_drawer() -> SceneGraph:
    """A 0.9 m drawer in a 0.6 m carcass — the ideation doc's own example.

    Fails at *every* joint value including fully closed, because the drawer is
    longer than the box it lives in. There is no clean sub-range, so a correct
    validator reports `feasible_limits=None` rather than inventing one.
    """
    cabinet = _object(
        "cabinet",
        "cabinet",
        [
            *_carcass(depth=0.6),
            _part("drawer_top", (0.5, 0.9, 0.3), (0.0, 0.0, 0.2), parent="bottom"),
        ],
        position=(0.0, 0.0, 0.46),
        prior_dims=(0.6, 0.6, 0.9),
        joints=[_drawer_joint("drawer_top_slide", "drawer_top", 0.4)],
    )
    return SceneGraph(objects=[cabinet])


def cabinet_door_into_table() -> SceneGraph:
    """A door that swings freely for part of its range and then hits a table.

    The interesting case, and the one AxisErr cannot see: the joint parameters are
    perfectly good, the door simply has nowhere to go past a certain angle. A
    correct validator reports a `blocked_at_q` partway through and
    `feasible_limits` covering the sub-range that does work — which is the range
    the object genuinely has and that a careless repair would throw away.
    """
    door = _part("door", (0.6, 0.02, 0.9), (0.0, 0.31, 0.0), parent="bottom", density=500.0)
    hinge = Joint(
        joint_id="door_hinge",
        name="door_left",
        type=JointType.REVOLUTE,
        parent_part_id="bottom",
        child_part_id="door",
        axis=(0.0, 0.0, 1.0),
        origin_m=(-0.3, 0.31, 0.0),  # hinged on the left edge of the open face
        limits=JointLimits(lower=0.0, upper=1.5708),  # 0-90 degrees, radians
        dynamics=JointDynamics(damping=2.0),
    )
    cabinet = _object(
        "cabinet",
        "cabinet",
        [*_carcass(), door],
        position=(0.0, 0.0, 0.46),
        prior_dims=(0.6, 0.6, 0.9),
        joints=[hinge],
    )
    # Parked just off the cabinet's open face, in the door's swing path.
    blocker = _object(
        "table",
        "table",
        [_part("top", (1.2, 0.75, 0.75), density=700.0)],
        position=(-0.75, 0.8, 0.375),
        prior_dims=(1.2, 0.75, 0.75),
    )
    return SceneGraph(objects=[cabinet, blocker])
