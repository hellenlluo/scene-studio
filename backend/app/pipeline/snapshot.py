"""Render the posed scene from its own camera, for comparison against the photo
it was reconstructed from.

A software rasteriser, not a real renderer. `app.export.gltf` already builds
production visual geometry, and the frontend already renders it properly in
three.js — but that render happens in a browser, and this one has to run inside a
headless background job with no display and no guarantee of a GPU. Neither
`pyrender` nor trimesh's own `Scene.save_image` are usable there (both need an
OpenGL context this process does not have), so this draws filled, shaded
triangles onto a plain array by hand. It only has to be good enough for a VLM to
compare shapes and colours against a photo — not to look at closely.

**Painter's algorithm, not a z-buffer.** Every triangle from every object is
projected once, sorted back-to-front by its camera-space depth, and drawn in that
order so nearer geometry overwrites farther geometry on the pixels they share.
Wrong for two triangles that interpenetrate, which is exactly the case this stage
is trying to surface — but per-pixel correctness is not what a duplicate-object
check needs, and a real z-buffer is a second renderer's worth of engineering to
buy an artifact only this one function reads.

**A second identical rasterisation writes an object-index buffer**, drawn in the
same sorted order so it agrees pixel-for-pixel with the colour image about which
object is nearest where. Reading `object_index == the object we care about` out
of that buffer gives an exact visibility mask, which is what turns into the
numbered outline — the same set-of-mark presentation `app.pipeline.annotate`
uses for masks, applied to a render instead of a photo, so the comparison prompt
sees one consistent visual language across both images.
"""

import io
import logging

import cv2
import numpy as np
import trimesh
from PIL import Image

from app.config import Settings
from app.export.gltf import geometry_for, node_transform
from app.geometry import world_aabb
from app.schemas import CameraPose, SceneGraph

__all__ = ["orbit_camera", "render", "render_views"]

# Degrees either side of the photo's own camera, and how high above the horizon.
# A single view hides exactly the case this stage looks for: two overlapping
# objects that occlude each other from the angle the photo was taken from can
# still read as one blob from the front and reveal themselves as two the moment
# the camera moves. +-50 degrees is enough to see behind most furniture silhouettes
# without swinging so far around that the render stops looking like the same room.
ORBIT_AZIMUTH_DEG = 50.0
ORBIT_ELEVATION_DEG = 32.0
# Margin beyond the tight framing distance, so a view from the side does not clip
# an object that was comfortably in frame from the front.
_FRAMING_MARGIN = 1.35

log = logging.getLogger(__name__)

# Same bright, room-unlikely palette as `app.pipeline.annotate`, duplicated rather
# than imported: it is a handful of RGB literals, not logic, and importing across
# a stage boundary for a shared constant would be a stranger coupling than having
# it twice.
_PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 32),
    (220, 64, 220),
    (32, 220, 220),
    (255, 128, 32),
    (160, 96, 255),
]
_OUTLINE_PX = 3
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BACKGROUND = (235, 235, 235)
# High ambient on purpose: this render exists to be read for colour and shape, not
# to look lit. A physically low ambient put half of every object's own surface
# near black, which is the opposite of what a colour comparison needs.
_AMBIENT = 0.55
# Two lights rather than one, each weighted below the ambient floor rather than
# above it — a single key light left the far side of every object at the ambient
# floor exactly, which reads as flat rather than shaped. `max` rather than a sum:
# this is a legibility hack, not a lighting model, and summing would blow out
# whichever faces both lights happen to catch.
_KEY_DIR = np.array([-0.4, -0.3, 0.85])
_KEY_DIR = _KEY_DIR / np.linalg.norm(_KEY_DIR)
_FILL_DIR = np.array([0.5, 0.4, 0.3])
_FILL_DIR = _FILL_DIR / np.linalg.norm(_FILL_DIR)
_NEAR_M = 0.05


def _camera_basis(camera: CameraPose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right, true-up, forward — orthonormal, world space.

    `camera.up` is a fitted hint, not guaranteed orthogonal to `forward`, so `right`
    is derived first and the true up re-derived from it rather than trusted as given.
    """
    forward = np.asarray(camera.forward, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, dtype=float))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def _face_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """One RGB per face, sampled from the mesh's own texture.

    **Per face, not per object, and this is the whole point of the render.** An
    object-average colour turns a reconstructed sofa into a uniform grey blob —
    which hides exactly what this stage exists to find, because a sofa whose
    upholstery includes two throw pillows is indistinguishable from a bare one
    once every triangle is painted the same colour. Verified the hard way: with a
    single mean colour the verification pass returned "no duplicates" on a scene
    that visibly had two.

    Nearest-texel per vertex, then averaged over the face's three. No bilinear
    filtering and no per-pixel interpolation across the triangle — flat shading is
    all the rasteriser below can draw anyway, and the question being asked of the
    image is "is that a brown pillow" rather than anything a filter would change.
    """
    visual = mesh.visual
    if isinstance(visual, trimesh.visual.TextureVisuals) and visual.material is not None:
        image = getattr(visual.material, "baseColorTexture", None)
        uv = getattr(visual, "uv", None)
        if image is not None and uv is not None and len(uv) == len(mesh.vertices):
            texture = np.asarray(image.convert("RGB"))
            height, width = texture.shape[:2]
            uv = np.asarray(uv, dtype=float)
            # glTF's v runs bottom-up; image rows run top-down.
            px = np.clip((uv[:, 0] * width).astype(int), 0, width - 1)
            py = np.clip(((1.0 - uv[:, 1]) * height).astype(int), 0, height - 1)
            return texture[py, px][mesh.faces].mean(axis=1)

    if isinstance(visual, trimesh.visual.ColorVisuals) and len(mesh.vertices):
        colors = np.asarray(visual.vertex_colors)[:, :3]
        return colors[mesh.faces].mean(axis=1)

    return np.full((len(mesh.faces), 3), 160.0)


def _object_triangles(graph: SceneGraph):
    """World-space triangles, one object's worth at a time: (index, tri, normal, colour).

    Deliberately not decimated. An earlier version decimated each part to a few
    hundred faces to keep the fill loop cheap, which was a mistake twice over:
    `simplify_quadric_decimation` discards UVs outright, and converting the
    texture to vertex colours first does not survive either — measured, a 9130-face
    sofa came back with every vertex the same colour. Since the colour *is* the
    signal here, the full mesh is rasterised. Measured at roughly 100k faces for a
    nine-object scene, which is seconds, once, in a stage that already waits on a
    network call.
    """
    for index, obj in enumerate(graph.objects):
        for part in obj.parts:
            mesh = geometry_for(part).copy()
            if not len(mesh.faces):
                continue
            colors = _face_colors(mesh)
            mesh.apply_transform(node_transform(obj, part))
            for triangle, normal, color in zip(
                mesh.triangles, mesh.face_normals, colors, strict=True
            ):
                yield index, triangle, normal, color


def _scene_bounds(graph: SceneGraph) -> tuple[np.ndarray, np.ndarray]:
    boxes = [world_aabb(obj) for obj in graph.objects]
    low = np.min([b[0] for b in boxes], axis=0)
    high = np.max([b[1] for b in boxes], axis=0)
    return low, high


def orbit_camera(graph: SceneGraph, azimuth_deg: float, elevation_deg: float) -> CameraPose:
    """A camera looking at the scene from a horizontal angle off the photo's own
    viewpoint, at whatever distance frames every object.

    Rotated about world +Z relative to the photo camera's own horizontal facing,
    rather than about a fixed world axis — "50 degrees to the side" should mean
    the same thing regardless of which way the room happens to face in world
    coordinates, and the photo's own forward direction is the only reference this
    scene has for what "the front" means.
    """
    base = graph.camera
    if base is None:
        raise ValueError("cannot orbit a scene with no fitted camera pose")

    forward_flat = np.array([base.forward[0], base.forward[1], 0.0])
    forward_flat = forward_flat / np.linalg.norm(forward_flat)
    theta = np.radians(azimuth_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    horizontal = np.array(
        [
            cos_t * forward_flat[0] - sin_t * forward_flat[1],
            sin_t * forward_flat[0] + cos_t * forward_flat[1],
            0.0,
        ]
    )

    low, high = _scene_bounds(graph)
    centre = (low + high) / 2.0
    radius = float(np.linalg.norm(high - low)) / 2.0
    distance = max(
        radius / np.tan(np.radians(base.vertical_fov_deg) / 2.0) * _FRAMING_MARGIN,
        radius * 1.5,
    )

    elevation = np.radians(elevation_deg)
    offset = np.array(
        [
            -horizontal[0] * np.cos(elevation),
            -horizontal[1] * np.cos(elevation),
            np.sin(elevation),
        ]
    )
    position = centre + offset * distance
    forward = centre - position
    forward = forward / np.linalg.norm(forward)
    return CameraPose(
        position_m=tuple(float(v) for v in position),
        forward=tuple(float(v) for v in forward),
        up=(0.0, 0.0, 1.0),
        vertical_fov_deg=base.vertical_fov_deg,
    )


def render(
    graph: SceneGraph, settings: Settings, aspect: float, camera: CameraPose | None = None
) -> bytes:
    """PNG bytes of the posed scene from `camera` (default: the photo's own).

    Every object's number is `graph.objects` position + 1, fixed regardless of
    whether that object ends up visible in this particular view. A number that
    shifted with visibility would mean "3" naming a different object in the front
    render than in the left one, which breaks the one thing a multi-view
    comparison depends on — a caller referring to a number and meaning the same
    object in every image it appears in.

    `aspect` is the original photo's width/height — matching it for the front view
    is what makes that render and the photo directly comparable rather than
    differently cropped views of the same room. The orbit views reuse it too,
    purely so all three images in a comparison prompt are a consistent shape.
    """
    camera = camera or graph.camera
    if camera is None:
        raise ValueError("cannot render a comparison snapshot without a fitted camera pose")

    height = (
        settings.snapshot_max_dim_px
        if aspect <= 1.0
        else round(settings.snapshot_max_dim_px / aspect)
    )
    width = round(height * aspect)
    right, up, forward = _camera_basis(camera)
    eye = np.asarray(camera.position_m, dtype=float)
    focal = (height / 2.0) / np.tan(np.radians(camera.vertical_fov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0

    def project(point: np.ndarray) -> tuple[float, float, float] | None:
        relative = point - eye
        depth = float(np.dot(relative, forward))
        if depth <= _NEAR_M:
            return None
        return (
            cx + focal * float(np.dot(relative, right)) / depth,
            cy - focal * float(np.dot(relative, up)) / depth,
            depth,
        )

    drawable = []
    for index, triangle, normal, color in _object_triangles(graph):
        # Backface cull: a triangle facing away from the camera contributes
        # nothing but noise to a shape comparison, and skipping it here is half
        # the polygon-fill cost of drawing it.
        centroid = triangle.mean(axis=0)
        if np.dot(normal, eye - centroid) <= 0.0:
            continue
        projected = [project(vertex) for vertex in triangle]
        if any(p is None for p in projected):
            continue
        depth = sum(p[2] for p in projected) / 3.0
        lit = max(
            0.85 * float(np.dot(normal, _KEY_DIR)),
            0.5 * float(np.dot(normal, _FILL_DIR)),
            0.0,
        )
        shade = _AMBIENT + (1.0 - _AMBIENT) * lit
        shaded = tuple(int(min(255, c * shade)) for c in color)
        polygon = np.array([(p[0], p[1]) for p in projected], dtype=np.int32)
        drawable.append((depth, index, polygon, shaded))

    # Painter's algorithm: farthest first, so nearer triangles overwrite farther
    # ones on the pixels they share.
    drawable.sort(key=lambda item: item[0], reverse=True)

    canvas = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)
    # 0 means background; an object's own index is never 0 so the two cannot be
    # confused when a mask is read back out of this.
    id_buffer = np.zeros((height, width), dtype=np.uint8)
    for _, index, polygon, shaded in drawable:
        cv2.fillConvexPoly(canvas, polygon, shaded)
        cv2.fillConvexPoly(id_buffer, polygon, index + 1)

    for index in range(len(graph.objects)):
        mask = id_buffer == (index + 1)
        if not mask.any():
            # Fully occluded or off-frame in this view: nothing to outline. The
            # number is still index + 1 in whichever other view does show it.
            continue

        # By position, not by draw order — matching the id_buffer's own encoding,
        # so object 5 has the same outline colour in every view too.
        colour = _PALETTE[index % len(_PALETTE)]
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, colour, _OUTLINE_PX)

        ys, xs = np.nonzero(mask)
        anchor = (int(xs.mean()), int(ys.mean()))
        if not mask[anchor[1], anchor[0]]:
            nearest = np.argmin((xs - anchor[0]) ** 2 + (ys - anchor[1]) ** 2)
            anchor = (int(xs[nearest]), int(ys[nearest]))

        text = str(index + 1)
        scale = max(0.5, min(height, width) / 900.0)
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 2)
        x, y = anchor
        cv2.rectangle(canvas, (x - 4, y - th - 6), (x + tw + 4, y + 6), colour, -1)
        cv2.putText(canvas, text, (x, y), _FONT, scale, (255, 255, 255), 2, cv2.LINE_AA)

    buffer = Image.fromarray(canvas, mode="RGB")
    out = io.BytesIO()
    buffer.save(out, format="PNG")
    return out.getvalue()


def render_views(graph: SceneGraph, settings: Settings, aspect: float) -> dict[str, bytes]:
    """The front view (the photo's own camera) plus a left and right orbit.

    Front is what compares directly against the photo. The two side views exist
    for what front cannot show: two objects that overlap from the angle the photo
    was taken from can still occlude each other into what reads as a single shape,
    and the point of moving the camera is that they stop lining up the moment it
    does. Numbering is shared across all three — see `render`.
    """
    return {
        "front": render(graph, settings, aspect),
        "left": render(
            graph, settings, aspect, orbit_camera(graph, -ORBIT_AZIMUTH_DEG, ORBIT_ELEVATION_DEG)
        ),
        "right": render(
            graph, settings, aspect, orbit_camera(graph, ORBIT_AZIMUTH_DEG, ORBIT_ELEVATION_DEG)
        ),
    }
