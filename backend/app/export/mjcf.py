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
    assets: dict[str, str],
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
        assets[asset_name] = mesh_path
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
    assets: dict[str, str],
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


def _add_object(worldbody: ET.Element, obj: SceneObject, assets: dict[str, str]) -> None:
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

    option = ET.SubElement(root, "option", {"timestep": "0.002"})
    # MuJoCo excludes contacts between a parent body and its child by default,
    # which would hide overlap between the rigidly-attached parts of one object.
    # Measured: with the default, a 20 cm interpenetration reports ncon=0.
    ET.SubElement(option, "flag", {"filterparent": "disable"})

    assets: dict[str, str] = {}
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
        for name, path in sorted(assets.items()):
            ET.SubElement(asset_el, "mesh", {"name": name, "file": path})
        root.insert(2, asset_el)

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def write_mjcf(graph: SceneGraph, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "scene.xml"
    path.write_text(build_xml(graph))
    return path
