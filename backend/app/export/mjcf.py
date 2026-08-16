from pathlib import Path

from app.schemas import GeometryResult, SceneLayout

__all__ = ["write_mjcf"]


def write_mjcf(geometry: GeometryResult, layout: SceneLayout, out_dir: Path) -> Path:
    """Emit MJCF for the scene.

    Used twice over: as the export format, and as the input the validate stage
    settles. Keep it a pure function of (geometry, layout) so validation and
    export can never drift apart — an exported scene that behaves differently
    from the validated one would make the whole physics-valid claim worthless.

    Reference collision meshes by relative path so the export stays portable.
    """
    raise NotImplementedError("mjcf export")
