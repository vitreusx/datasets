"""Norway DTM1 dataset downloader."""

from pathlib import Path

import numpy as np
import tyro
from lxml import etree
from pydantic import BaseModel

from rsrch_data.utils.download import download
from rsrch_data.utils.misc import parse_size


class Args(BaseModel):
    """CLI args for `get_norway_dtm1.py` script."""

    data_root: str
    feed_url: str = (
        "https://nedlasting.geonorge.no/geonorge/ATOM/hoydedata/datasett/DTM1.atom"
    )
    subset_size: str | None = None
    """If provided (e.g. "20GiB"), randomly sample tiles up to this total size."""
    seed: int = 0
    """Seed for the tile subset sample."""


def main(args: Args) -> None:
    """Get Norway DTM1 dataset."""
    data_root = Path(args.data_root)
    download(args.feed_url, data_root / "DTM1.atom")

    with (data_root / "DTM1.atom").open("rb") as f:
        root = etree.fromstring(f.read())

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "georss": "http://www.georss.org/georss",
    }

    tiles = []
    for entry in root.findall("atom:entry", ns):
        link = entry.find("atom:link[@rel='section']", ns)
        if link is None:
            continue
        url = link.get("href")
        if url is None:
            continue
        length = link.get("length")
        tiles.append((url, int(length) if length is not None else 0))

    urls = [url for url, _ in tiles]
    if args.subset_size is not None:
        budget = parse_size(args.subset_size)
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(tiles))
        urls = []
        total = 0
        for i in order:
            url, length = tiles[i]
            if total + length > budget:
                continue
            urls.append(url)
            total += length

    for url in urls:
        download(url, dest_dir=(data_root / "tiles"))


if __name__ == "__main__":
    main(tyro.cli(Args))
