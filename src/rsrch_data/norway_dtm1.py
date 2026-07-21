"""Windowed elevation-raster access over locally downloaded Norway DTM1 tiles."""

from pathlib import Path

from rsrch_data.utils.geo_chunks import GeoChunk, load_manifest
from rsrch_data.utils.raster_tiles import TiledRaster


class NorwayDTM1(TiledRaster):
    """Windowed heightmap accessor over Norway DTM1 tiles (see get_norway_dtm1.py).

    Tiles are 15010x15010m squares in ETRS89 / UTM zone 33N (EPSG:25833) at
    native 1m/px resolution.
    """

    def __init__(self, data_root: str | Path):
        """Index the tiles found under `data_root/tiles`."""
        data_root = Path(data_root)
        super().__init__(data_root / "tiles", glob="*.tif")

        manifest_path = data_root / "manifest.json"
        self.manifest: list[GeoChunk] | None = (
            load_manifest(manifest_path) if manifest_path.exists() else None
        )
