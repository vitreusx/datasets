"""Norway DTM1 dataset downloader.

Covers Norway at 1m resolution, in ETRS89 / UTM zone 33N (EPSG:25833).
Tiles are individually downloadable GeoTIFFs already, so there's no
archive-listing step, and a chunk is always exactly one tile.

## Pipeline

Both `build-manifest` and `download` read from the same single flat feed
(`DTM1.atom`, ~2000 `<entry>`s, one per tile -- no pagination, no
sub-feeds):

1. Fetch the whole feed once (`download(feed_url, ...)` in
   `_run_build_manifest`/`_load_or_build_manifest`) -- everything below is
   then pure local parsing, no further network calls.
2. `build_manifest` reads each entry's `<link rel="section">` for the
   tile's download URL + declared byte size, and its `<georss:polygon>`
   for a WGS84 footprint -- reprojected to an exact native-CRS
   (EPSG:25833) bbox via `_polygon_native_bbox`. The polygon is already an
   axis-aligned rectangle in UTM33N; it only *looks* skewed once
   reprojected to WGS84 for the feed.
3. `manifest.json` is written in one shot -- parsing the whole
   (already-downloaded) feed is fast enough that redoing it from scratch
   on a rebuild is never a real cost.
4. `download` selects a subset (region / chunk IDs / size budget) from
   the manifest and fetches each tile's URL directly and concurrently
   (`ThreadPoolExecutor`, `--workers`) -- no extraction step, since tiles
   aren't bundled in an archive.

## Feed format

One flat feed, `DTM1.atom`, whose `<entry>`s carry more metadata than
this script uses -- capture-project provenance, municipality overlap,
etc. Simplified real example below; everything marked unused is genuine
feed content this script simply ignores:

```xml
<entry xmlns:gn="http://geonorge.no/geonorge" xmlns:georss="...">
  <title>Høydedata DTM1 33-173-199</title>              <!-- unused -->
  <content>...</content>                                 <!-- unused -->
  <!-- WGS84 footprint -- reprojected to a native bbox, not used as-is
       (see pipeline step 2). -->
  <georss:polygon>70.80 28.60 70.93 28.69 ...</georss:polygon>
  <category term="..." label="ETRS89 / UTM zone 33N" .../> <!-- unused -->
  <!-- municipalities this tile overlaps -- unused; explains why a tile
       can list more than one <category> of this kind. -->
  <category term="5630" label="Berlevåg" .../>
  <category term="5626" label="Gamvik" .../>
  <gn:coverage>                       <!-- capture-project info, unused -->
    <gn:project number="LACH0009">
      <gn:name>NDH Berlevåg 2pkt 2019</gn:name>
    </gn:project>
  </gn:coverage>
  <!-- the only other field this script needs besides the polygon:
       download URL + declared size. -->
  <link rel="section" type="application/geotiff" length="636651442"
        href="https://nedlasting.geonorge.no/hoydedata/DTM1/33-173-199.tif"/>
</entry>
```
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import tyro
from lxml import etree
from pydantic import BaseModel
from rasterio.warp import transform
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from rsrch_data.utils.cli import print_table
from rsrch_data.utils.download import download
from rsrch_data.utils.geo_chunks import (
    GeoChunk,
    GeoTile,
    chunks_intersecting,
    load_manifest,
    save_manifest,
)
from rsrch_data.utils.misc import parse_size

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "georss": "http://www.georss.org/georss",
}
NATIVE_CRS = "EPSG:25833"
DEFAULT_FEED_URL = (
    "https://nedlasting.geonorge.no/geonorge/ATOM/hoydedata/datasett/DTM1.atom"
)


class _CommonArgs(BaseModel):
    data_root: str
    feed_url: str = DEFAULT_FEED_URL


class BuildManifest(_CommonArgs):
    """Build (or refresh) the tile manifest from the feed."""


class ListChunks(_CommonArgs):
    """List manifest chunks, optionally filtered by region."""

    region: tuple[float, float, float, float] | None = None
    """(left, bottom, right, top) in WGS84 degrees."""


class Download(_CommonArgs):
    """Download a subset of chunks (tiles)."""

    region: tuple[float, float, float, float] | None = None
    """(left, bottom, right, top) in WGS84 degrees."""
    chunk_ids: list[str] | None = None
    subset_size: str | None = None
    """If provided (e.g. "20GiB"), randomly sample tiles up to this total size."""
    seed: int = 0
    """Seed for the tile subset sample."""
    workers: int = 4
    """Number of concurrent download workers."""


Args = (
    Annotated[BuildManifest, tyro.conf.subcommand("build-manifest")]
    | Annotated[ListChunks, tyro.conf.subcommand("list-chunks")]
    | Annotated[Download, tyro.conf.subcommand("download")]
)


def _polygon_native_bbox(polygon_text: str) -> tuple[float, float, float, float]:
    """Reproject a georss:polygon ("lat lon lat lon ...") into NATIVE_CRS bbox."""
    coords = [float(v) for v in polygon_text.split()]
    lats, lons = coords[0::2], coords[1::2]
    xs, ys = transform("EPSG:4326", NATIVE_CRS, lons, lats)
    return (min(xs), min(ys), max(xs), max(ys))


def build_manifest(feed_path: Path) -> list[GeoChunk]:
    """Build one chunk per tile entry in a downloaded DTM1.atom."""
    with feed_path.open("rb") as f:
        root = etree.fromstring(f.read())

    chunks = []
    for entry in root.findall("atom:entry", NS):
        link = entry.find("atom:link[@rel='section']", NS)
        polygon = entry.find("georss:polygon", NS)
        if link is None or polygon is None or polygon.text is None:
            continue
        url = link.get("href")
        if url is None:
            continue

        length = link.get("length")
        size = int(length) if length is not None else 0
        bbox = _polygon_native_bbox(polygon.text)
        tile_id = Path(url).stem

        tile = GeoTile(id=tile_id, bbox=bbox, size=size, download_ref=url)
        chunk = GeoChunk(
            id=tile_id, crs=NATIVE_CRS, tiles=[tile], size=size, download_urls=[url]
        )
        chunks.append(chunk)
    return chunks


def _load_or_build_manifest(data_root: Path, feed_url: str) -> list[GeoChunk]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        return load_manifest(manifest_path)

    feed_path = data_root / "DTM1.atom"
    download(feed_url, feed_path)
    chunks = build_manifest(feed_path)
    save_manifest(chunks, manifest_path)
    return chunks


def _select_by_budget(
    chunks: list[GeoChunk], subset_size: str, seed: int
) -> list[GeoChunk]:
    budget = parse_size(subset_size)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(chunks))
    selected, total = [], 0
    for i in order:
        chunk = chunks[i]
        if total + chunk.size > budget:
            continue
        selected.append(chunk)
        total += chunk.size
    return selected


def _run_build_manifest(args: BuildManifest) -> None:
    data_root = Path(args.data_root)
    feed_path = data_root / "DTM1.atom"
    download(args.feed_url, feed_path)
    chunks = build_manifest(feed_path)
    manifest_path = data_root / "manifest.json"
    save_manifest(chunks, manifest_path)
    print(f"{len(chunks)} chunks -> {manifest_path}")  # noqa: T201


def _run_list_chunks(args: ListChunks) -> None:
    chunks = _load_or_build_manifest(Path(args.data_root), args.feed_url)
    if args.region is not None:
        chunks = chunks_intersecting(chunks, args.region, "EPSG:4326")

    rows = [(c.id, len(c.tiles), f"{c.size / 1e6:.1f} MB") for c in chunks]
    total_gb = sum(c.size for c in chunks) / 1e9
    caption = f"{len(chunks)} chunks, {total_gb:.2f} GB total"
    print_table(["Chunk ID", "Tiles", "Size"], rows, caption=caption)


def _run_download(args: Download) -> None:
    data_root = Path(args.data_root)
    chunks = _load_or_build_manifest(data_root, args.feed_url)

    if args.chunk_ids is not None:
        wanted = set(args.chunk_ids)
        selected = [c for c in chunks if c.id in wanted]
    elif args.region is not None:
        selected = chunks_intersecting(chunks, args.region, "EPSG:4326")
    elif args.subset_size is not None:
        selected = _select_by_budget(chunks, args.subset_size, args.seed)
    else:
        selected = chunks

    tiles = [tile for chunk in selected for tile in chunk.tiles]
    total_size = sum(t.size for t in tiles)
    completed = 0
    lock = threading.Lock()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"0/{len(tiles)} tiles", total=total_size)

        def run_one(tile: GeoTile) -> None:
            nonlocal completed
            download(
                tile.download_ref,
                dest_dir=(data_root / "tiles"),
                display_pbar=False,
                on_chunk=lambda n: progress.update(task, advance=n),
            )
            with lock:
                completed += 1
                progress.update(task, description=f"{completed}/{len(tiles)} tiles")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(run_one, tile) for tile in tiles]
            for f in futures:
                f.result()


def main() -> None:
    """Get Norway DTM1 dataset (build-manifest / list-chunks / download)."""
    args = tyro.cli(Args)
    if isinstance(args, BuildManifest):
        _run_build_manifest(args)
    elif isinstance(args, ListChunks):
        _run_list_chunks(args)
    else:
        _run_download(args)


if __name__ == "__main__":
    main()
