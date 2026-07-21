"""RGE ALTI (French DTM) dataset downloader.

Covers mainland France (incl. Corsica) only, at 1m resolution, in
Lambert-93 / EPSG:2154 (both the IGN69 and Corsica-specific IGN78C vertical
datums share that horizontal CRS). Overseas territories use other CRSs
entirely and are intentionally excluded.

## Why two phases

The dataset is too large (~500GB across ~22k chunks) to just download
wholesale and see what's in it, so the pipeline below is split into two
independent phases around that constraint:

- **`build-manifest`** is a metadata-only pass. It walks the ATOM feeds
  (department list, then each department's archive listing) and, for each
  department, opens its 7z archive *remotely* to read its internal folder
  structure and per-dalle filenames -- all of which is a few small
  requests per department, never the archive payload. The result,
  `manifest.json`, is a full inventory of every dalle's geographic extent
  and which chunk (7z solid block) it lives in.
- **`download`** then works purely from that manifest, never re-touching
  the feeds: given a region, explicit chunk IDs, a department, or a size
  budget, it resolves the minimal set of chunks covering that selection
  and extracts only those, avoiding having to fetch everything.

This is also why the two phases are separately resumable/cacheable: a
build interrupted partway through still leaves a valid (partial)
`manifest.json`, and a `download` never needs to redo the (slow,
rate-limited) metadata walk once it's cached.

## Pipeline: `build-manifest` (`build_manifest`)

Starting from nothing but `feed_url`:

1. GET the top-level product feed's page 1 (`_paginate` ->
   `data_root/feeds/RGEALTI_p1.atom`). Its own root element carries the
   total page count (see the XML below), so page 1 must be fetched before
   we know how many more pages to GET.
2. GET every remaining page the same way (`RGEALTI_p{N}.atom`), and from
   every page's entries, keep only mainland-France ones, reading off each
   one's title, department code/name, and sub-feed URL
   (`_list_departments`, `_department_info`).
3. GET each department's own sub-feed (`_resolve_departments` ->
   `data_root/feeds/{title}.atom`) and read off its 7z archive volume
   URL(s) + declared sizes, ignoring the sibling `.md5` entry
   (`_archive_parts`).
4. For each department not already in a cached `manifest.json` (paced by
   a rate limiter): open its 7z archive *remotely* over HTTP Range
   requests (`rsrch_data.utils.remote_7z`) and read its solid-block
   ("folder") structure -- no archive *payload* bytes are fetched yet,
   only its header. One `Chunk` is recorded per folder, its member
   dalles' bboxes derived from their own filenames, not downloaded
   (`_probe_department_folders`, `_dalle_bbox`; see below for why).
5. `manifest.json` is saved after every department, so an interrupted
   build resumes for free (already-probed departments are skipped).

## Pipeline: `download` (`_run_download`/`_download_chunks`)

Given the manifest from above and a selection (region / chunk IDs /
department / random size-budget subset):

1. Group the selected chunks by their shared archive, then split each
   group into up to `workers` size-balanced sub-batches
   (`_partition_by_size`) so one worker thread's job is never "the whole
   department" even for a single-department download.
2. Per sub-batch: `extract_remote_7z` fetches + decompresses only the
   solid block(s) containing that batch's dalles -- cost is bounded by
   the touched folders' size, not the archive's.
3. Each extracted `.asc` dalle is immediately converted to a compressed
   GeoTIFF and the `.asc` discarded (`_asc_to_geotiff`; see below).

## Feed format

Both feed levels are IGN Geoplateforme-specific (data.geopf.fr) ATOM,
decorated with a `gpf_dl` namespace the plain Atom spec doesn't have.
Simplified real examples below, irrelevant fields cut.

Top-level product feed (`RGEALTI_p{N}.atom`), one `<entry>` per
department+datum+edition -- see `_list_departments`/`_department_info`:

```xml
<feed xmlns:gpf_dl="https://data.geopf.fr/.../gpf_dl.xsd"
      gpf_dl:pagecount="22">              <!-- total page count -->
  <entry>
    <title>RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D006_2024-02-02</title>
    <!-- rel="alternate" here means "this product's own sub-feed",
         NOT a download link -- contrast with the sub-feed below. -->
    <link rel="alternate"
          href=".../RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D006_2024-02-02"/>
    <gpf_dl:zone term="D006" label="D006 Alpes-Maritimes"/>
    <gpf_dl:editionDate>2024-02-02</gpf_dl:editionDate>
  </entry>
  <!-- ...9 more <entry>s on this page; 22 pages total... -->
</feed>
```

Per-department sub-feed (`{title}.atom`), one `<entry>` per downloadable
file -- see `_archive_parts`:

```xml
<feed>
  <!-- the 7z archive itself: ONE such <entry> with rel="alternate" if
       it fits in a single volume (shown here), or SEVERAL with
       rel="section" instead (one per ".7z.001", ".7z.002", ...) if IGN
       split it across volumes. -->
  <entry>
    <link rel="alternate" type="application/x-7z-compressed"
          gpf_dl:length="4145934963"
          href=".../RGEALTI_..._D003_2023-08-10.7z"/>
  </entry>
  <!-- sibling checksum file -- always present, filtered out by
       matching on type="application/x-7z-compressed" rather than by
       filename, since that's the field _archive_parts actually reads. -->
  <entry>
    <link rel="alternate" type="application/octet-stream"
          gpf_dl:length="133"
          href=".../RGEALTI_..._D003_2023-08-10.md5"/>
  </entry>
</feed>
```

## Other notes

Download granularity here is finer than "whole department": each
department's 7z archive is internally split into several solid-compressed
"folders" (~16 to ~2000 dalles each), and `rsrch_data.utils.remote_7z` can
list and selectively extract those without downloading the archive in
full. Chunks in the manifest are one department-folder each.

Per-dalle extents (step 4 above) are **not** sourced from IGN's "tableau
d'assemblage" (TA) tile-index shapefile -- despite the field name
`NOM_DALLE` looking like it names individual 1m tiles, that shapefile
actually indexes a much coarser 5km grid (its tile coordinates are always
multiples of 5; real 1m tile coordinates usually aren't), unrelated to
the 1m dalles this script downloads. Instead, each dalle's exact bbox is
derived straight from its own filename, `RGEALTI_FXX_{X}_{Y}_MNT_...` --
LAMB93 km-grid coordinates, verified byte-exact against real downloaded
tiles' rasterio bounds (`(X*1000-0.5, Y*1000-999.5, X*1000+999.5,
Y*1000+0.5)`, matching the ASC format's pixel-center xllcorner/yllcorner
convention).

IGN ships dalles as `.asc` (Arc/Info ASCII Grid): a 6-line text header
(ncols/nrows/xllcorner/yllcorner/cellsize/NODATA_value) followed by every
cell value written out as ASCII text, with no compression. That measures
~9x larger and ~5x slower to read than binary on a real tile
(8.00MB/61.8ms vs. 0.88MB/12.4ms) -- hence the GeoTIFF conversion in the
download pipeline above.
"""

import re
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import rasterio
import tyro
from lxml import etree
from pydantic import BaseModel
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
from rsrch_data.utils.remote_7z import RateLimiter, extract_remote_7z, open_remote_7z

ATOM_NS = "http://www.w3.org/2005/Atom"
GPF_NS = "https://data.geopf.fr/annexes/ressources/xsd/gpf_dl.xsd"
NS = {"atom": ATOM_NS, "gpf_dl": GPF_NS}

MAINLAND_TITLE_RE = re.compile(r"^RGEALTI_2-0_1M_ASC_LAMB93-IGN")
DALLE_COORD_RE = re.compile(r"RGEALTI_FXX_(\d+)_(\d+)_MNT")
NATIVE_CRS = "EPSG:2154"


class _CommonArgs(BaseModel):
    data_root: str
    feed_url: str = "https://data.geopf.fr/telechargement/resource/RGEALTI"


class BuildManifest(_CommonArgs):
    """Build (or resume/refresh) the chunk manifest from the feed."""


class ListChunks(_CommonArgs):
    """List manifest chunks, optionally filtered by region."""

    region: tuple[float, float, float, float] | None = None
    """(left, bottom, right, top) in WGS84 degrees."""


class ListDepartments(_CommonArgs):
    """List departments present in the manifest, with chunk/tile counts and size."""


class Download(_CommonArgs):
    """Download a subset of chunks (department 7z folders)."""

    region: tuple[float, float, float, float] | None = None
    """(left, bottom, right, top) in WGS84 degrees."""
    chunk_ids: list[str] | None = None
    """Individual chunk IDs to fetch. See `list-chunks` for a list."""
    departments: list[str] | None = None
    """Department titles, or unambiguous substrings (e.g. "D075"), to fetch
    in full."""
    subset_size: str | None = None
    """If provided (e.g. "50GiB"), randomly sample chunks up to this total size."""
    seed: int = 0
    """Seed for the chunk subset sample."""
    workers: int = 4
    """Number of concurrent extraction workers. Each paces itself to the
    server's observed ~1 req/s limit, so this also sets the aggregate
    request rate; empirically, throughput scales well up to ~4 workers with
    no rate-limit rejections, with diminishing returns beyond that."""


Args = (
    Annotated[BuildManifest, tyro.conf.subcommand("build-manifest")]
    | Annotated[ListChunks, tyro.conf.subcommand("list-chunks")]
    | Annotated[ListDepartments, tyro.conf.subcommand("list-departments")]
    | Annotated[Download, tyro.conf.subcommand("download")]
)


def _parse_feed(path: Path) -> etree._Element:
    with path.open("rb") as f:
        return etree.fromstring(f.read())


def _paginate(feed_url: str, feeds_dir: Path) -> list[etree._Element]:
    """Fetch and parse every page of a paginated Geoplateforme atom feed."""
    page1_path = feeds_dir / "RGEALTI_p1.atom"
    download(feed_url, page1_path)
    page1 = _parse_feed(page1_path)
    pagecount = int(page1.get(f"{{{GPF_NS}}}pagecount"))

    pages = [page1]
    sep = "&" if "?" in feed_url else "?"
    for p in range(2, pagecount + 1):
        path = feeds_dir / f"RGEALTI_p{p}.atom"
        download(f"{feed_url}{sep}page={p}", path)
        pages.append(_parse_feed(path))
    return pages


def _archive_parts(sub_feed: etree._Element) -> list[tuple[str, int]]:
    """Get (url, length) for each 7z volume in a department's sub-feed.

    Volumes are `<entry>`s whose link type is 7z, regardless of `rel`
    (single-volume archives use rel="alternate", split ones rel="section");
    this also excludes each sub-feed's sibling `.md5` checksum entry.
    """
    parts = []
    for entry in sub_feed.findall("atom:entry", NS):
        link = entry.find("atom:link", NS)
        if link is None or link.get("type") != "application/x-7z-compressed":
            continue
        url = link.get("href")
        length = link.get(f"{{{GPF_NS}}}length")
        parts.append((url, int(length) if length is not None else 0))
    return parts


def _list_departments(pages: list[etree._Element]) -> list[tuple[str, str]]:
    """Get (title, sub_feed_url) for each mainland department in the feed."""
    departments = []
    for page in pages:
        for entry in page.findall("atom:entry", NS):
            title = entry.find("atom:title", NS).text
            if MAINLAND_TITLE_RE.match(title) is None:
                continue
            link = entry.find("atom:link[@rel='alternate']", NS)
            departments.append((title, link.get("href")))
    return departments


ZONE_LABEL_RE = re.compile(r"^D\d{2,3}[AB]?\s+(.*)$")


def _department_info(pages: list[etree._Element]) -> dict[str, tuple[str, str, str]]:
    """Get title -> (code, French name, edition date), from gpf_dl:zone/editionDate."""
    info = {}
    for page in pages:
        for entry in page.findall("atom:entry", NS):
            title = entry.find("atom:title", NS).text
            if MAINLAND_TITLE_RE.match(title) is None:
                continue
            zone = entry.find("gpf_dl:zone", NS)
            if zone is None or zone.get("term") is None or zone.get("label") is None:
                continue
            code, label = zone.get("term"), zone.get("label")
            m = ZONE_LABEL_RE.match(label)
            name = m.group(1) if m else label
            edition = entry.find("gpf_dl:editionDate", NS)
            edition_date = edition.text if edition is not None else ""
            info[title] = (code, name, edition_date)
    return info


def _resolve_departments(
    departments: list[tuple[str, str]], feeds_dir: Path
) -> list[tuple[str, list[tuple[str, int]]]]:
    """Get (title, archive parts) for each department."""
    resolved = []
    for title, sub_feed_url in departments:
        sub_feed_path = feeds_dir / f"{title}.atom"
        download(sub_feed_url, sub_feed_path)
        parts = _archive_parts(_parse_feed(sub_feed_path))
        resolved.append((title, parts))
    return resolved


def _dalle_bbox(name: str) -> tuple[float, float, float, float] | None:
    """Exact native-CRS bbox of a dalle, derived from its own filename."""
    m = DALLE_COORD_RE.search(name)
    if m is None:
        return None
    x, y = int(m.group(1)), int(m.group(2))
    return (x * 1000 - 0.5, y * 1000 - 999.5, x * 1000 + 999.5, y * 1000 + 0.5)


def _probe_department_folders(
    title: str,
    parts: list[tuple[str, int]],
    rate_limiter: RateLimiter,
) -> list[GeoChunk]:
    """List a department archive's solid blocks and build one Chunk per block."""
    urls = [url for url, _ in parts]
    z, _stream = open_remote_7z(urls, rate_limiter=rate_limiter)
    with z:
        names = z.getnames()
        streams = z.header.main_streams
        n_per_folder = streams.substreamsinfo.num_unpackstreams_folders
        packsizes = streams.packinfo.packsizes

    chunks = []
    file_idx = 0
    folders = zip(n_per_folder, packsizes, strict=True)
    for folder_idx, (n_files, pack_size) in enumerate(folders):
        folder_names = names[file_idx : file_idx + n_files]
        file_idx += n_files

        tiles = []
        for name in folder_names:
            if not name.endswith(".asc"):
                continue
            bbox = _dalle_bbox(name)
            if bbox is None:
                continue
            tiles.append(
                GeoTile(id=Path(name).stem, bbox=bbox, size=0, download_ref=name)
            )

        if not tiles:
            continue

        chunks.append(
            GeoChunk(
                id=f"{title}#{folder_idx}",
                crs=NATIVE_CRS,
                tiles=tiles,
                size=pack_size,
                download_urls=urls,
            )
        )
    return chunks


def build_manifest(data_root: Path, feed_url: str) -> list[GeoChunk]:
    """Build the full department-folder chunk list, resuming from any cache."""
    feeds_dir = data_root / "feeds"
    feeds_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = data_root / "manifest.json"
    existing = load_manifest(manifest_path) if manifest_path.exists() else []
    done_titles = {c.id.split("#", 1)[0] for c in existing}

    pages = _paginate(feed_url, feeds_dir)
    departments = _list_departments(pages)
    resolved = _resolve_departments(departments, feeds_dir)

    rate_limiter = RateLimiter(per_second=1.0)
    chunks = list(existing)
    for title, parts in resolved:
        if title in done_titles:
            continue
        new_chunks = _probe_department_folders(title, parts, rate_limiter)
        chunks.extend(new_chunks)
        save_manifest(chunks, manifest_path)  # persist progress after each department

    return chunks


def _load_or_build_manifest(data_root: Path, feed_url: str) -> list[GeoChunk]:
    manifest_path = data_root / "manifest.json"
    if manifest_path.exists():
        return load_manifest(manifest_path)
    return build_manifest(data_root, feed_url)


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


def _partition_by_size(chunks: list[GeoChunk], n: int) -> list[list[GeoChunk]]:
    """Greedily split chunks into <=n bins, balanced by total size (LPT)."""
    n = max(1, min(n, len(chunks)))
    bins: list[list[GeoChunk]] = [[] for _ in range(n)]
    totals = [0] * n
    for c in sorted(chunks, key=lambda c: c.size, reverse=True):
        i = min(range(n), key=lambda i: totals[i])
        bins[i].append(c)
        totals[i] += c.size
    return bins


def _asc_to_geotiff(asc_path: Path, dest_path: Path) -> None:
    """Convert an .asc (AAIGrid) tile to a compressed, CRS-tagged GeoTIFF.

    .asc stores every cell as ASCII text with no compression -- ~9x larger
    and ~5x slower to read than a DEFLATE+predictor GeoTIFF, empirically
    (measured on a real RGE ALTI tile: 8.00MB/61.8ms vs 0.88MB/12.4ms).
    """
    with rasterio.open(asc_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
    profile.update(
        driver="GTiff",
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        crs=NATIVE_CRS,
    )
    with rasterio.open(dest_path, "w", **profile) as dst:
        dst.write(data, 1)


def _download_chunks(
    selected: list[GeoChunk], data_root: Path, workers: int = 4
) -> None:
    """Download+extract selected chunks, across `workers` concurrent workers.

    Chunks are first grouped by department archive (shared download_urls),
    then each group is further split into size-balanced sub-batches so a
    single department's download can also run concurrently, not just
    downloads spanning multiple departments. Each worker paces its own
    requests independently, so aggregate request rate scales with `workers`.
    """
    tiles_dir = data_root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    by_urls: dict[tuple[str, ...], list[GeoChunk]] = defaultdict(list)
    for chunk in selected:
        by_urls[tuple(chunk.download_urls)].append(chunk)

    batches: list[tuple[tuple[str, ...], list[GeoChunk]]] = [
        (urls, batch)
        for urls, chunks in by_urls.items()
        for batch in _partition_by_size(chunks, workers)
    ]

    total_size = sum(c.size for c in selected)
    completed = 0
    lock = threading.Lock()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"0/{len(batches)} batches", total=total_size)

        def run_batch(urls: tuple[str, ...], chunks: list[GeoChunk]) -> None:
            nonlocal completed
            targets = [t.download_ref for c in chunks for t in c.tiles]
            rate_limiter = RateLimiter(per_second=1.0)
            with tempfile.TemporaryDirectory() as tmp:
                extract_remote_7z(
                    list(urls),
                    targets,
                    Path(tmp),
                    rate_limiter=rate_limiter,
                    on_fetch=lambda n: progress.update(task, advance=n),
                )
                for asc_path in Path(tmp).rglob("*.asc"):
                    tif_path = tiles_dir / f"{asc_path.stem}.tif"
                    _asc_to_geotiff(asc_path, tif_path)
            with lock:
                completed += 1
                progress.update(task, description=f"{completed}/{len(batches)} batches")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_batch, urls, chunks) for urls, chunks in batches]
            for f in futures:
                f.result()


def _match_departments(chunks: list[GeoChunk], wanted: list[str]) -> list[GeoChunk]:
    """Select chunks whose department title matches one of `wanted`.

    Each item in `wanted` may be a full department title or a substring
    (e.g. a bare code like "D075"); a substring must match exactly one
    department title in the manifest.
    """
    all_titles = sorted({c.id.split("#", 1)[0] for c in chunks})
    resolved_titles = set()
    for w in wanted:
        if w in all_titles:
            resolved_titles.add(w)
            continue
        matches = [t for t in all_titles if w in t]
        if len(matches) == 0:
            msg = f"No department matches {w!r}"
            raise ValueError(msg)
        if len(matches) > 1:
            msg = f"{w!r} matches multiple departments: {matches}"
            raise ValueError(msg)
        resolved_titles.add(matches[0])
    return [c for c in chunks if c.id.split("#", 1)[0] in resolved_titles]


def _run_build_manifest(args: BuildManifest) -> None:
    chunks = build_manifest(Path(args.data_root), args.feed_url)
    print(f"{len(chunks)} chunks -> {Path(args.data_root) / 'manifest.json'}")  # noqa: T201


def _run_list_chunks(args: ListChunks) -> None:
    chunks = _load_or_build_manifest(Path(args.data_root), args.feed_url)
    if args.region is not None:
        chunks = chunks_intersecting(chunks, args.region, "EPSG:4326")

    rows = [(c.id, len(c.tiles), f"{c.size / 1e6:.1f} MB") for c in chunks]
    total_gb = sum(c.size for c in chunks) / 1e9
    caption = f"{len(chunks)} chunks, {total_gb:.2f} GB total"
    print_table(["Chunk ID", "Tiles", "Size"], rows, caption=caption)


def _run_list_departments(args: ListDepartments) -> None:
    data_root = Path(args.data_root)
    chunks = _load_or_build_manifest(data_root, args.feed_url)
    pages = _paginate(args.feed_url, data_root / "feeds")
    info = _department_info(pages)

    by_dept: dict[str, list[GeoChunk]] = defaultdict(list)
    for c in chunks:
        by_dept[c.id.split("#", 1)[0]].append(c)

    codes_seen: dict[str, int] = defaultdict(int)
    for title in by_dept:
        codes_seen[info.get(title, ("?", "?", ""))[0]] += 1

    def _sort_key(title: str) -> str:
        return info.get(title, ("?", "?", ""))[0]

    rows = []
    for title in sorted(by_dept, key=_sort_key):
        dept_chunks = by_dept[title]
        n_tiles = sum(len(c.tiles) for c in dept_chunks)
        size_gb = sum(c.size for c in dept_chunks) / 1e9
        code, name, edition_date = info.get(title, ("?", "?", ""))
        if codes_seen[code] > 1:
            name += f" ({edition_date})"
        rows.append((code, name, len(dept_chunks), n_tiles, f"{size_gb:.2f} GB"))

    total_gb = sum(c.size for c in chunks) / 1e9
    caption = f"{len(by_dept)} depts, {len(chunks)} chunks, {total_gb:.2f} GB total"
    print_table(["Code", "Name", "Chunks", "Tiles", "Size"], rows, caption=caption)


def _run_download(args: Download) -> None:
    data_root = Path(args.data_root)
    chunks = _load_or_build_manifest(data_root, args.feed_url)

    if args.chunk_ids is not None:
        wanted = set(args.chunk_ids)
        selected = [c for c in chunks if c.id in wanted]
    elif args.departments is not None:
        selected = _match_departments(chunks, args.departments)
    elif args.region is not None:
        selected = chunks_intersecting(chunks, args.region, "EPSG:4326")
    elif args.subset_size is not None:
        selected = _select_by_budget(chunks, args.subset_size, args.seed)
    else:
        selected = chunks

    _download_chunks(selected, data_root, workers=args.workers)


def main() -> None:
    """Get RGE ALTI dataset (build-manifest / list-chunks / list-departments / dl)."""
    args = tyro.cli(Args)
    if isinstance(args, BuildManifest):
        _run_build_manifest(args)
    elif isinstance(args, ListChunks):
        _run_list_chunks(args)
    elif isinstance(args, ListDepartments):
        _run_list_departments(args)
    else:
        _run_download(args)


if __name__ == "__main__":
    main()
