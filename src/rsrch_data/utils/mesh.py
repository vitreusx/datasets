"""Mesh export utilities."""

from pathlib import Path

import numpy as np
import trimesh


def heightmap_to_glb(
    heightmap: np.ndarray,
    path: str | Path,
    meters_per_px: float = 1.0,
    scale: float = 1.0,
    colormap: np.ndarray | None = None,
) -> None:
    """Export a heightmap to a triangle mesh .glb.

    `heightmap` is (H, W) with row 0 as the north edge, matching
    `NorwayDTM1.get_tile`. Non-finite values (e.g. `-np.inf`) mark missing
    data; any triangle touching one is dropped rather than spanning the gap.
    `scale` uniformly scales the whole model (all three axes alike).

    `colormap`, if given, is an (H, W, 3) float RGB array (values in
    [0, 1]) pixel-aligned with `heightmap`, applied as per-vertex color.

    Vertices are laid out (east, elevation, -north) so that +Y is up, per
    glTF's own convention -- this is what makes Blender's importer (and other
    spec-compliant viewers) show the terrain upright instead of on its side.
    """
    h, w = heightmap.shape
    if colormap is not None and colormap.shape[:2] != (h, w):
        msg = f"colormap shape {colormap.shape[:2]} must match heightmap shape {(h, w)}"
        raise ValueError(msg)

    xs = np.arange(w) * meters_per_px
    ys = (h - 1 - np.arange(h)) * meters_per_px
    xx, yy = np.meshgrid(xs, ys)

    valid = np.isfinite(heightmap)
    zz = np.where(valid, heightmap, 0.0)
    vertices = scale * np.stack([xx, zz, -yy], axis=-1).reshape(-1, 3)

    idx = np.arange(h * w).reshape(h, w)
    tl, tr, bl, br = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    valid_quad = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]

    faces = np.concatenate(
        [
            np.stack([tl, bl, tr], axis=-1)[valid_quad],
            np.stack([tr, bl, br], axis=-1)[valid_quad],
        ]
    )

    vertex_colors = None
    if colormap is not None:
        vertex_colors = np.clip(colormap.reshape(-1, 3), 0.0, 1.0)

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=vertex_colors, process=True
    )
    mesh.remove_unreferenced_vertices()
    mesh.export(path, file_type="glb")
