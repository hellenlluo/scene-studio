"""MJCF emission.

Used three times over: as the export format, as the input every physics
certification axis steps, and as what the browser's MuJoCo WASM build loads.
Keeping it one pure function of the graph is what stops those from drifting — an
exported scene that behaves differently from the certified one would void the
whole simulation-ready claim, and the same MJCF on both sides is what makes the
browser authoritative rather than approximate.

Naming is part of the contract. `body_name` is the only place that decides what
things are called, so `app.certify` maps MuJoCo indices back to schema objects by
calling it rather than by parsing strings apart.
"""

from pathlib import Path
from xml.etree import ElementTree as ET

from app.schemas import PartGeometry, SceneGraph, SceneObject, Vec3

__all__ = ["body_name", "build_xml", "free_joint_name", "write_mjcf"]


def body_name(object_id: str, part_id: str) -> str:
    return f"{object_id}/{part_id}"


def free_joint_name(object_id: str) -> str:
    return f"{object_id}/free"


# Contact stiffness. MuJoCo's defaults model a compliant contact, which is right
# for a gripper pad and wrong for a room full of furniture: a body sinks into its
# support until the constraint force balances gravity, and at 224 kg — the mass a
# shell-volume sofa comes out at — that is centimetres of visible penetration the
# stability axis then reports as displacement.
#
# `solref` is (timeconst, dampratio) and the first term has a floor: MuJoCo needs
# timeconst >= 2 * timestep to stay stable, so 0.005 against a 0.002 timestep is
# close to as stiff as this timestep permits. Anything stiffer needs the timestep
# to come down with it, which costs step time on the cost axis.
#
# Measured on the `room` scene, settling for 2 s: a floor lamp went from 335 mm of
# displacement to 1.7 mm. Nothing else moved by more than a millimetre either way,
# so this buys the lamp and costs nothing elsewhere.
#
# **This is a modelling choice, not only a bug fix.** It asserts that reconstructed
# furniture is rigid, and it shifts every number the stability axis reports, so the
# sensitivity of the certification results to it belongs in the evaluation
# alongside the penetration tolerance.
TIMESTEP_S = "0.002"
CONTACT_SOLREF = "0.005 1"
CONTACT_SOLIMP = "0.99 0.999 0.001"


def _add_contact_defaults(root: ET.Element) -> None:
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {"solref": CONTACT_SOLREF, "solimp": CONTACT_SOLIMP})


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def _scaled(vec: Vec3, scale: float) -> tuple[float, float, float]:
    return (vec[0] * scale, vec[1] * scale, vec[2] * scale)


def _half(dims: Vec3, scale: float) -> tuple[float, float, float]:
    # MuJoCo box `size` is a half-extent, not a full dimension. Getting this wrong
    # doubles every object and is invisible until nothing fits inside anything.
    return (dims[0] * scale / 2.0, dims[1] * scale / 2.0, dims[2] * scale / 2.0)


def _sub(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vec3, b: Vec3) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _add_geoms(
    parent: ET.Element,
    obj: SceneObject,
    part: PartGeometry,
    assets: dict[str, tuple[str, float]],
    offset: Vec3 = (0.0, 0.0, 0.0),
) -> None:
    """Collision geometry for one part.

    A part with no meshes falls back to a box from its OBB extent, which is what
    lets the whole pipeline — and every test in this repo — run before any mesh
    reconstruction exists.

    `offset` is non-zero only for the root part. Its body *is* the object frame,
    so unlike every other part it has no body position of its own to carry
    `origin_m`, and the offset has to land on the geometry instead. Dropping it
    silently translates the whole object by the root part's offset.
    """
    if not part.collision_mesh_paths:
        ET.SubElement(
            parent,
            "geom",
            {
                "name": f"{body_name(obj.object_id, part.part_id)}/obb",
                "type": "box",
                "size": _fmt(_half(part.dims_m, obj.scale)),
                "pos": _fmt(offset),
            },
        )
        return

    for index, mesh_path in enumerate(part.collision_mesh_paths):
        asset_name = f"{body_name(obj.object_id, part.part_id)}/{index}"
        # The scale rides on the asset, not the geom: MuJoCo has no per-geom mesh
        # scale, and the file is stored at unit object scale so that changing
        # `SceneObject.scale` never means rewriting geometry. Asset names carry the
        # object id, so no two objects can share an entry and disagree about it.
        assets[asset_name] = (mesh_path, obj.scale)
        ET.SubElement(
            parent,
            "geom",
            {
                "name": asset_name,
                "type": "mesh",
                "mesh": asset_name,
                "pos": _fmt(offset),
                # Convex pieces of one decomposition. MuJoCo needs each piece
                # separately; it does not decompose on your behalf.
            },
        )


def _add_inertial(parent: ET.Element, part: PartGeometry, offset: Vec3 = (0.0, 0.0, 0.0)) -> None:
    """Emit explicit mass properties, or let MuJoCo infer them from geometry.

    Inferred values come from a uniform default density and are almost always
    wrong for real objects, but a body with no mass at all is a hard model error —
    so inference is the right behaviour before the inertia stage has run.
    """
    if part.inertial is None:
        return
    ET.SubElement(
        parent,
        "inertial",
        {
            "pos": _fmt(_add(part.inertial.com_m, offset)),
            "mass": f"{part.inertial.mass_kg:.6g}",
            "diaginertia": _fmt(part.inertial.inertia_diag),
            "quat": _fmt(part.inertial.principal_axes),
        },
    )


def _add_part(
    parent: ET.Element,
    obj: SceneObject,
    part: PartGeometry,
    children: dict[str | None, list[PartGeometry]],
    assets: dict[str, tuple[str, float]],
    origin: Vec3,
) -> None:
    """Recursively emit a part and everything hanging off it."""
    body = ET.SubElement(
        parent,
        "body",
        {
            "name": body_name(obj.object_id, part.part_id),
            "pos": _fmt(_scaled(_sub(part.origin_m, origin), obj.scale)),
        },
    )

    _add_inertial(body, part)
    _add_geoms(body, obj, part, assets)

    for child in children.get(part.part_id, []):
        _add_part(body, obj, child, children, assets, part.origin_m)


def _add_object(
    worldbody: ET.Element, obj: SceneObject, assets: dict[str, tuple[str, float]]
) -> None:
    root = obj.root_part
    body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": body_name(obj.object_id, root.part_id),
            "pos": _fmt(obj.position_m),
            "quat": _fmt(obj.orientation),
        },
    )
    # Every object floats freely so the stability axis can settle it under
    # gravity. Parts within an object are rigidly attached to each other.
    ET.SubElement(body, "freejoint", {"name": free_joint_name(obj.object_id)})

    children: dict[str | None, list[PartGeometry]] = {}
    for part in obj.parts:
        children.setdefault(part.parent_part_id, []).append(part)

    root_offset = _scaled(root.origin_m, obj.scale)
    _add_inertial(body, root, root_offset)
    _add_geoms(body, obj, root, assets, root_offset)
    for child in children.get(root.part_id, []):
        _add_part(body, obj, child, children, assets, root.origin_m)


def build_xml(graph: SceneGraph) -> str:
    """The whole scene as an MJCF string. Pure — no filesystem, so tests are cheap."""
    root = ET.Element("mujoco", {"model": "scenestudio"})

    ET.SubElement(root, "compiler", {"angle": "radian"})

    # `impratio` above 1 raises frictional constraint impedance relative to normal.
    # The default of 1 lets a resting object creep sideways under its own weight,
    # which the stability axis reads as displacement — measured on a reconstructed
    # floor lamp, 324 mm of pure lateral slide with nothing pushing it.
    option = ET.SubElement(root, "option", {"timestep": TIMESTEP_S, "impratio": "10"})
    # MuJoCo excludes contacts between a parent body and its child by default,
    # which would hide overlap between the rigidly-attached parts of one object.
    # Measured: with the default, a 20 cm interpenetration reports ncon=0.
    ET.SubElement(option, "flag", {"filterparent": "disable"})

    _add_contact_defaults(root)

    assets: dict[str, tuple[str, float]] = {}
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": _fmt((0.0, 0.0, graph.floor_height_m)),
            "size": "0 0 0.05",
        },
    )

    for obj in graph.objects:
        _add_object(worldbody, obj, assets)

    if assets:
        asset_el = ET.Element("asset")
        for name, (path, scale) in sorted(assets.items()):
            ET.SubElement(
                asset_el, "mesh", {"name": name, "file": path, "scale": _fmt((scale,) * 3)}
            )
        # Located by lookup rather than a literal index: MuJoCo wants `asset` ahead
        # of `worldbody`, and a hard-coded position silently means the wrong slot the
        # next time a section is added before it.
        root.insert(list(root).index(worldbody), asset_el)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_mjcf(graph: SceneGraph, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scene.xml"
    path.write_text(build_xml(graph))
    return path
