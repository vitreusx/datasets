"""Mip-NeRF 360 scene dataset."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from PIL import Image

from rsrch_data._colmap.read_write_model import (
    Camera,
    read_cameras_binary,
    read_images_binary,
)
from rsrch_data.registry import register_dataset


def _c2w(entry: object) -> np.ndarray:
    """Invert COLMAP world-to-camera to get a 4×4 camera-to-world matrix."""
    w2c = np.eye(4)
    w2c[:3, :3] = entry.qvec2rotmat()  # type: ignore[attr-defined]
    w2c[:3, 3] = entry.tvec  # type: ignore[attr-defined]
    return np.linalg.inv(w2c)


def _build_k(camera: Camera, scale: float = 1.0) -> np.ndarray:
    """Build a 3×3 intrinsics matrix from a COLMAP camera, optionally rescaled."""
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":  # f, cx, cy
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    elif camera.model == "PINHOLE":  # fx, fy, cx, cy
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    else:  # Generic: treat first param as focal, next two as principal point
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    return np.array(
        [[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]],
        dtype=np.float32,
    )


class Frame(TypedDict):
    """Mip-NeRF 360 frame."""

    image: Image.Image
    """Frame RGB image."""
    c2w: np.ndarray
    """Extrinsics array - shape (4, 4), dtype float32."""
    K: np.ndarray
    """Intrinsics array - shape (3, 3), dtype float32."""


@register_dataset("mip-nerf360")
class MipNerf360(Sequence):
    """Mip-NeRF 360 scene dataset.

    File structure::

        <data_root>/
        ├── images/          # full-resolution JPEGs
        ├── images_2/        # ×2 downsampled
        ├── images_4/        # ×4 downsampled
        ├── images_8/        # ×8 downsampled
        └── sparse/0/
            ├── cameras.bin  # COLMAP intrinsics
            └── images.bin   # COLMAP extrinsics

    Each item is a dict:

    - ``image``: Pillow image.
    - ``c2w``: float32 ndarray (4, 4), camera-to-world.
      COLMAP convention: x right, y down, z into scene.
    - ``K``: float32 ndarray (3, 3), intrinsics scaled for the loaded resolution.
    """

    def __init__(
        self,
        data_root: str | Path,
        downsample: Literal[1, 2, 4, 8] = 1,
    ) -> None:
        root = Path(data_root).expanduser()
        self._img_dir = root / ("images" if downsample == 1 else f"images_{downsample}")
        self._scale = 1.0 / downsample

        sparse_dir = root / "sparse" / "0"
        cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
        images = read_images_binary(str(sparse_dir / "images.bin"))

        self._entries = sorted(images.values(), key=lambda e: e.name)
        self._cameras = cameras

    def __len__(self) -> int:
        """Return number of images in the scene."""
        return len(self._entries)

    def __getitem__(self, idx: int) -> Frame:
        """Return dict with ``image``, ``c2w``, and ``K`` for one view."""
        entry = self._entries[idx]
        img = Image.open(self._img_dir / entry.name).convert("RGB")
        return {
            "image": img,
            "c2w": _c2w(entry).astype(np.float32),
            "K": _build_k(self._cameras[entry.camera_id], self._scale),
        }
