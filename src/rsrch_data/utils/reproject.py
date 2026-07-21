"""Cross-CRS raster reprojection utilities."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


def reproject_to(
    src_path: str | Path,
    dst_crs: str,
    rect: tuple[float, float, float, float],
    px_size: float,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    """Resample a single-band raster onto an axis-aligned grid in another CRS.

    `rect` (left, bottom, right, top) and `px_size` define the destination
    grid, in `dst_crs` units -- e.g. pass the same `rect`/`meters_per_px` used
    for `RgeAlti.get_tile` to get another raster (like WTE) warped onto that
    exact grid, ready to stack as another channel. Use `Resampling.nearest`
    for categorical data (never average/bilinear -- that blends unrelated
    class codes into meaningless ones) and `Resampling.average` (or similar)
    for continuous data like elevation.
    """
    left, bottom, right, top = rect
    out_w = round((right - left) / px_size)
    out_h = round((top - bottom) / px_size)
    dst_transform = from_origin(left, top, px_size, px_size)

    with rasterio.open(src_path) as src:
        dst = np.full((out_h, out_w), src.nodata, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=src.nodata,
            resampling=resampling,
        )
    return dst
