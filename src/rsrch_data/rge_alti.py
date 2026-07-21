"""Windowed elevation-raster access over locally downloaded RGE ALTI tiles."""

from pathlib import Path

from rsrch_data.utils.geo_chunks import GeoChunk, load_manifest
from rsrch_data.utils.raster_tiles import TiledRaster


class RgeAlti(TiledRaster):
    """Windowed heightmap accessor over RGE ALTI tiles (see get_rge_alti.py).

    Covers mainland France (incl. Corsica) only, in Lambert-93 (EPSG:2154).
    Tiles are 1000x1000m squares at native 1m/px resolution. IGN ships these
    as .asc (AAIGrid) -- uncompressed ASCII text, ~9x larger and ~5x slower
    to read than binary -- so `get_rge_alti.py`'s downloader converts each
    tile to a DEFLATE-compressed, CRS-tagged GeoTIFF on extraction.
    """

    def __init__(self, data_root: str | Path):
        """Index the tiles found under `data_root/tiles`."""
        data_root = Path(data_root)
        super().__init__(data_root / "tiles", glob="*.tif")

        manifest_path = data_root / "manifest.json"
        self.manifest: list[GeoChunk] | None = (
            load_manifest(manifest_path) if manifest_path.exists() else None
        )
