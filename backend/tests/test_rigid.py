"""Reconstruction response handling.

The API call is untestable here, but the two conversions that unpick its response
both fail *silently* when wrong — a mis-ordered quaternion is still a valid
rotation, and a mishandled scale still produces a mesh. Nothing raises; objects
just come out wrong.
"""

import io

import numpy as np
import pytest
import trimesh
from PIL import Image

from app.pipeline.rigid import _load_glb, _quat_wxyz, _split_scale

# --- quaternion order ---------------------------------------------------------


def test_scalar_last_becomes_scalar_first():
    """fal returns XYZW, MuJoCo and this schema use WXYZ. Passing one through as
    the other yields a perfectly valid rotation that is simply the wrong one."""
    assert _quat_wxyz([0.0, 0.0, 0.7071, 0.7071]) == pytest.approx((0.7071, 0.0, 0.0, 0.7071))


def test_identity_survives_the_conversion():
    # XYZW identity is (0,0,0,1); WXYZ identity is (1,0,0,0).
    assert _quat_wxyz([0.0, 0.0, 0.0, 1.0]) == (1.0, 0.0, 0.0, 0.0)


def test_a_missing_or_malformed_rotation_falls_back_to_identity():
    """An object at an unknown orientation is better placed upright than at a
    rotation invented from a partial array."""
    assert _quat_wxyz(None) == (1.0, 0.0, 0.0, 0.0)
    assert _quat_wxyz([0.0, 1.0]) == (1.0, 0.0, 0.0, 0.0)


# --- anisotropic scale --------------------------------------------------------


def test_an_isotropic_scale_leaves_the_shape_alone():
    anisotropy, isotropic = _split_scale([2.0, 2.0, 2.0])
    assert anisotropy == pytest.approx([1.0, 1.0, 1.0])
    assert isotropic == pytest.approx(2.0)


def test_the_shape_part_has_unit_product():
    """That is what makes it pure proportion: it changes the object's shape without
    changing its overall size, so the size stays a single solver variable."""
    anisotropy, _ = _split_scale([4.0, 1.0, 0.5])
    assert float(np.prod(anisotropy)) == pytest.approx(1.0)


def test_the_two_parts_reconstruct_the_original():
    original = [3.0, 0.5, 1.2]
    anisotropy, isotropic = _split_scale(original)
    assert (anisotropy * isotropic) == pytest.approx(original)


def test_a_flat_object_keeps_its_flatness_in_the_mesh():
    """A rug is genuinely much wider than it is thick. That belongs in the geometry,
    not in a scale variable that would break its inertia."""
    anisotropy, _ = _split_scale([10.0, 10.0, 0.1])
    assert anisotropy[2] < anisotropy[0]


def test_a_degenerate_scale_is_ignored_rather_than_propagated():
    """A zero or negative factor would collapse or mirror the mesh. Falling back to
    no scaling leaves the object wrong-sized, which the solver can fix; a mirrored
    mesh it cannot."""
    for bad in ([0.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [float("nan"), 1.0, 1.0], None, [1.0, 2.0]):
        anisotropy, isotropic = _split_scale(bad)
        assert anisotropy == pytest.approx([1.0, 1.0, 1.0])
        assert isotropic == 1.0


# --- the response's actual shape ----------------------------------------------


def test_metadata_values_are_wrapped_one_level_deep():
    """Captured from a real fal response: `"scale": [[0.99, 0.99, 0.99]]`. Read as a
    flat list it is a one-element sequence, every length check rejects it, and the
    object silently falls back to identity rotation and unit scale — which is
    exactly what the first live run produced, without raising anything."""
    from app.pipeline.rigid import _unwrap

    assert _unwrap([[1.0, 2.0, 3.0]]) == [1.0, 2.0, 3.0]


def test_unwrap_leaves_an_already_flat_value_alone():
    """The nesting is an observed quirk, not a guarantee — if it ever stops, this
    must not start mangling correct input."""
    from app.pipeline.rigid import _unwrap

    assert _unwrap([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]
    assert _unwrap(None) is None


def test_parses_a_real_captured_response():
    """Against a response saved verbatim from fal, so a change in their format
    fails here rather than silently degrading every object to identity."""
    import json
    from pathlib import Path

    from app.pipeline.rigid import _unwrap

    raw = json.loads((Path(__file__).parent / "fixtures" / "sam3d_response.json").read_text())
    info = raw["metadata"][0]

    anisotropy, isotropic = _split_scale(info["scale"])
    assert isotropic == pytest.approx(0.9901, abs=1e-3)
    # This particular object came back isotropic, so nothing to bake into the mesh.
    assert anisotropy == pytest.approx([1.0, 1.0, 1.0])

    w, x, y, z = _quat_wxyz(info["rotation"])
    assert (w, x, y, z) == pytest.approx((0.6847, 0.1215, 0.1148, 0.7094), abs=1e-3)
    assert np.linalg.norm([w, x, y, z]) == pytest.approx(1.0, abs=1e-4)

    assert _unwrap(info["translation"])[2] == pytest.approx(2.741, abs=1e-3)


# --- watertightness, measured on a welded copy ---------------------------------


def _textured_glb(mesh: trimesh.Trimesh) -> bytes:
    """A GLB carrying a UV-mapped material, the way SAM 3D returns one.

    The seams are the subject, so they have to be real: a `TextureVisuals` with a
    per-vertex UV array is what makes glTF write one vertex per UV corner, and what
    makes the mesh come back unwelded.
    """
    mesh = mesh.copy()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.random.default_rng(0).random((len(mesh.vertices), 2)),
        image=Image.new("RGB", (4, 4), (128, 128, 128)),
    )
    return trimesh.Scene(mesh).export(file_type="glb")


def test_a_textured_closed_mesh_is_measured_as_closed():
    """The bug this guards: a glTF stores a vertex per UV corner, so a closed
    surface loads split at every seam and reads as not watertight. Every object in
    every real scene came back `None`, and the volume mass depends on was never
    measured."""
    box = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    _, volume, watertight = _load_glb(_textured_glb(box), max_faces=100_000)

    assert watertight
    assert volume == pytest.approx(0.4 * 0.2 * 0.1, rel=1e-6)


def test_the_stored_mesh_keeps_its_seams():
    """Measure on a copy, ship the original. Welding has to pick one UV per merged
    vertex, so doing it in place stretches the texture along every seam — and
    keeping textures is why `max_mesh_faces` is 40k rather than 5k."""
    box = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    data = _textured_glb(box)

    as_loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    mesh, _, _ = _load_glb(data, max_faces=100_000)

    assert len(mesh.vertices) == len(as_loaded.vertices)
    assert isinstance(mesh.visual, trimesh.visual.TextureVisuals)
    assert mesh.visual.material.baseColorTexture is not None


def test_welding_does_not_move_the_volume():
    """Only the verdict changes. trimesh integrates per triangle, so vertex sharing
    never enters the number — which is why this fix is safe to apply to scenes
    already measured the old way."""
    data = _textured_glb(trimesh.creation.icosphere(subdivisions=2, radius=0.1))
    _, volume, _ = _load_glb(data, max_faces=100_000)

    # Against the mesh as loaded, not the one authored above: glTF stores vertices
    # as float32, so a round trip moves the volume by ~2e-8 relative and would
    # swamp the thing being measured.
    as_loaded = trimesh.load(io.BytesIO(data), file_type="glb", force="mesh")
    assert volume == pytest.approx(float(as_loaded.volume), rel=1e-12)


def test_a_genuinely_open_mesh_still_reads_as_open():
    """Welding is not repair. A shell with a hole in it has no trustworthy volume
    and has to say so, or the inertial axis certifies on a number nobody measured."""
    box = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    box.update_faces(np.arange(len(box.faces)) > 1)  # drop a face; leave a hole

    _, volume, watertight = _load_glb(_textured_glb(box), max_faces=100_000)

    assert not watertight
    assert volume is None
