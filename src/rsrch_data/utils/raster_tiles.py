"""Windowed raster access over a flat directory of geospatial tile files."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


class TiledRaster:
    """Windowed heightmap accessor over a flat directory of same-CRS raster tiles.

    Tiles are assumed axis-aligned, in a single shared CRS, each exposing its
    own bounds/nodata via rasterio. Since a download may only cover a random
    subset of tiles, `extent` bounds whatever is on disk -- gaps within it are
    expected, and `get_tile` fills them (and any tile-internal nodata) with
    `-np.inf`.
    """

    def __init__(self, tiles_dir: str | Path, glob: str) -> None:
        """Index the tiles matching `glob` under `tiles_dir`."""
        tiles_dir = Path(tiles_dir)
        paths = sorted(tiles_dir.glob(glob))
        if not paths:
            msg = f"No tiles found in {tiles_dir} (glob: {glob!r})"
            raise ValueError(msg)

        self._tiles: list[tuple[Path, tuple[float, float, float, float]]] = []
        for path in paths:
            with rasterio.open(path) as ds:
                self._tiles.append((path, tuple(ds.bounds)))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Bounding box (left, bottom, right, top) of the tiles on disk."""
        lefts, bottoms, rights, tops = zip(*(b for _, b in self._tiles), strict=True)
        return (min(lefts), min(bottoms), max(rights), max(tops))

    @property
    def tiles(self) -> list[tuple[float, float, float, float]]:
        """Rects (left, bottom, right, top) of the individual tiles on disk."""
        return [bounds for _, bounds in self._tiles]

    @staticmethod
    def _out_size(
        rect: tuple[float, float, float, float], meters_per_px: float
    ) -> tuple[int, int]:
        """Compute (width, height) in pixels of `rect` at `meters_per_px`."""
        if meters_per_px < 1.0:
            msg = (
                f"meters_per_px must be >= 1.0 (native resolution), got {meters_per_px}"
            )
            raise ValueError(msg)

        left, bottom, right, top = rect
        out_w = round((right - left) / meters_per_px)
        out_h = round((top - bottom) / meters_per_px)
        return out_w, out_h

    def get_tile_size(
        self,
        rect: tuple[float, float, float, float],
        meters_per_px: float = 1.0,
    ) -> tuple[int, int]:
        """Get the (width, height) in pixels that `get_tile` would return."""
        return self._out_size(rect, meters_per_px)

    def get_tile(
        self,
        rect: tuple[float, float, float, float],
        meters_per_px: float = 1.0,
    ) -> np.ndarray:
        """Get the heightmap over `rect` (left, bottom, right, top), tile CRS units.

        `meters_per_px` must be >= 1.0 (native resolution); coarser values are
        block-averaged. `-np.inf` marks nodata pixels and areas not covered by
        any downloaded tile.
        """
        left, bottom, right, top = rect
        out_w, out_h = self._out_size(rect, meters_per_px)
        out = np.full((out_h, out_w), -np.inf, dtype=np.float32)

        for path, (t_left, t_bottom, t_right, t_top) in self._tiles:
            ix_left, ix_right = max(left, t_left), min(right, t_right)
            ix_bottom, ix_top = max(bottom, t_bottom), min(top, t_top)
            if ix_left >= ix_right or ix_bottom >= ix_top:
                continue

            sub_w = max(1, round((ix_right - ix_left) / meters_per_px))
            sub_h = max(1, round((ix_top - ix_bottom) / meters_per_px))

            with rasterio.open(path) as ds:
                window = from_bounds(ix_left, ix_bottom, ix_right, ix_top, ds.transform)
                data = ds.read(
                    1,
                    window=window,
                    out_shape=(sub_h, sub_w),
                    resampling=Resampling.average,
                )
                data = np.where(data == ds.nodata, -np.inf, data)

            # Tiles are pasted independently, so adjacent tiles at coarse
            # meters_per_px can round to overlapping/gapped pixel spans here.
            ox = round((ix_left - left) / meters_per_px)
            oy = round((top - ix_top) / meters_per_px)
            out[oy : oy + sub_h, ox : ox + sub_w] = data

        return out
