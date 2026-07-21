"""Windowed access over the World Terrestrial Ecosystems (WTE) 2020 raster."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from dbfread import DBF
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


@dataclass(frozen=True)
class EcosystemClass:
    """One WTE class: a landform x landcover x temperature x moisture combo."""

    value: int
    pixel_count: int
    landform: str
    landcover: str
    temperature: str
    moisture: str
    description: str


class Wte:
    """Windowed accessor over the WTE 2020 raster (see get_wte.py).

    Unlike the tiled DTM datasets, WTE is a single global raster in
    EPSG:4326 (degrees, not meters) at ~250m (0.00224579 degree) resolution.
    Pixels are `uint16` ecosystem class codes, not continuous elevation --
    `get_tile` uses nearest-neighbor resampling (never averaging, which
    would blend unrelated classes into meaningless codes), and missing data
    is the raster's own nodata code rather than `-np.inf`. Use `classes` to
    look up what a class code means.
    """

    def __init__(self, data_root: str | Path):
        """Load the raster header and the class legend under `data_root/raster`."""
        raster_dir = Path(data_root) / "raster"
        self.path = raster_dir / "WorldEcosystem.tif"
        with rasterio.open(self.path) as ds:
            self._bounds = tuple(ds.bounds)
            self._res = ds.res[0]
            self.nodata = int(ds.nodata)

        vat_path = raster_dir / "WorldEcosystem.tif.vat.dbf"
        self.classes: dict[int, EcosystemClass] = {}
        for rec in DBF(vat_path, load=True):
            self.classes[rec["Value"]] = EcosystemClass(
                value=rec["Value"],
                pixel_count=int(rec["Count"]),
                landform=rec["LF_ClassNa"],
                landcover=rec["LC_ClassNa"],
                temperature=rec["Temp_Class"],
                moisture=rec["Moisture_C"],
                description=rec["W_Ecosystm"],
            )

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Bounding box (left, bottom, right, top) in EPSG:4326 degrees."""
        return self._bounds

    def _out_size(
        self, rect: tuple[float, float, float, float], degrees_per_px: float
    ) -> tuple[int, int]:
        """Compute (width, height) in pixels of `rect` at `degrees_per_px`."""
        if degrees_per_px < self._res:
            msg = (
                f"degrees_per_px must be >= native resolution ({self._res}), "
                f"got {degrees_per_px}"
            )
            raise ValueError(msg)

        left, bottom, right, top = rect
        out_w = round((right - left) / degrees_per_px)
        out_h = round((top - bottom) / degrees_per_px)
        return out_w, out_h

    def get_tile_size(
        self,
        rect: tuple[float, float, float, float],
        degrees_per_px: float | None = None,
    ) -> tuple[int, int]:
        """Get the (width, height) in pixels that `get_tile` would return."""
        return self._out_size(rect, degrees_per_px or self._res)

    def get_tile(
        self,
        rect: tuple[float, float, float, float],
        degrees_per_px: float | None = None,
    ) -> np.ndarray:
        """Get ecosystem class codes over `rect` (left, bottom, right, top), degrees.

        `degrees_per_px` defaults to the native resolution and must not be
        finer; coarser values are nearest-neighbor resampled. Pixels outside
        the raster's own extent, or with no data, are `self.nodata`.
        """
        degrees_per_px = degrees_per_px or self._res
        left, bottom, right, top = rect
        out_w, out_h = self._out_size(rect, degrees_per_px)

        r_left, r_bottom, r_right, r_top = self._bounds
        ix_left, ix_right = max(left, r_left), min(right, r_right)
        ix_bottom, ix_top = max(bottom, r_bottom), min(top, r_top)

        out = np.full((out_h, out_w), self.nodata, dtype=np.uint16)
        if ix_left >= ix_right or ix_bottom >= ix_top:
            return out

        sub_w = max(1, round((ix_right - ix_left) / degrees_per_px))
        sub_h = max(1, round((ix_top - ix_bottom) / degrees_per_px))

        with rasterio.open(self.path) as ds:
            window = from_bounds(ix_left, ix_bottom, ix_right, ix_top, ds.transform)
            data = ds.read(
                1,
                window=window,
                out_shape=(sub_h, sub_w),
                resampling=Resampling.nearest,
            )

        ox = round((ix_left - left) / degrees_per_px)
        oy = round((top - ix_top) / degrees_per_px)
        out[oy : oy + sub_h, ox : ox + sub_w] = data
        return out
