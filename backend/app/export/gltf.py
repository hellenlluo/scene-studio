from pathlib import Path

from app.schemas import SceneGraph

__all__ = ["write_gltf"]


def write_gltf(graph: SceneGraph, out_dir: Path) -> Path:
    """Emit glTF for the viewer: visual meshes only, posed and scaled.

    Separate from the physics exports on purpose. This carries the visual tier,
    where MJCF carries the collision proxies, and conflating them would mean
    either shipping decomposed hulls to the browser or certifying against
    render meshes.
    """
    raise NotImplementedError("gltf export")
