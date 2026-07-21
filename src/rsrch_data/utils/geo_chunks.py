"""Geography-aware download-chunk metadata: list, query, and cache."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rasterio.warp import transform_bounds

Bbox = tuple[float, float, float, float]  # (left, bottom, right, top)


@dataclass(frozen=True)
class GeoTile:
    """One elevation tile, in its dataset's native CRS.

    `download_ref` is how to locate this tile once its chunk is fetched:
    for Norway, its own direct download URL; for RGE ALTI, its full path
    inside the department archive (a `py7zr` extraction target).
    """

    id: str
    bbox: Bbox
    size: int
    download_ref: str


@dataclass(frozen=True)
class GeoChunk:
    """One independently-downloadable unit: one or more tiles, one CRS.

    `download_urls` is what to fetch to obtain this chunk's tiles: for
    Norway, a single tile's own URL; for RGE ALTI, the department archive's
    volume URL(s) (1 for a plain `.7z`, 2+ for split `.7z.001`, `.7z.002`...).
    """

    id: str
    crs: str
    tiles: list[GeoTile]
    size: int
    download_urls: list[str]


def _bbox_intersects(a: Bbox, b: Bbox) -> bool:
    a_left, a_bottom, a_right, a_top = a
    b_left, b_bottom, b_right, b_top = b
    return (
        a_left < b_right and a_right > b_left and a_bottom < b_top and a_top > b_bottom
    )


def chunks_intersecting(chunks: list[GeoChunk], bbox: Bbox, crs: str) -> list[GeoChunk]:
    """Chunks with >=1 member tile intersecting `bbox` (given in `crs`).

    Assumes `chunks` share a single native CRS (true for both datasets today);
    `bbox` is reprojected once into that CRS before testing.
    """
    if not chunks:
        return []

    dst_crs = chunks[0].crs
    if crs != dst_crs:
        left, bottom, right, top = bbox
        bbox = transform_bounds(crs, dst_crs, left, bottom, right, top)

    return [c for c in chunks if any(_bbox_intersects(t.bbox, bbox) for t in c.tiles)]


def save_manifest(chunks: list[GeoChunk], path: Path) -> None:
    """Save a chunk manifest as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([asdict(c) for c in chunks], f)


def _tile_from_dict(t: dict) -> GeoTile:
    return GeoTile(
        id=t["id"],
        bbox=tuple(t["bbox"]),
        size=t["size"],
        download_ref=t["download_ref"],
    )


def load_manifest(path: Path) -> list[GeoChunk]:
    """Load a chunk manifest saved by `save_manifest`."""
    with path.open() as f:
        raw = json.load(f)
    return [
        GeoChunk(
            id=c["id"],
            crs=c["crs"],
            size=c["size"],
            download_urls=c["download_urls"],
            tiles=[_tile_from_dict(t) for t in c["tiles"]],
        )
        for c in raw
    ]
