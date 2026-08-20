from pathlib import Path

from app.schemas import SceneGraph

__all__ = ["write_urdf"]


def write_urdf(graph: SceneGraph, out_dir: Path) -> Path:
    """Emit URDF for the scene.

    The interchange format, not the authoritative one — URDF cannot express
    everything MJCF can, and where the two disagree MJCF wins because that is
    what certification measured. Note what was lost rather than emitting a URDF
    that quietly means something different.
    """
    raise NotImplementedError("urdf export")
