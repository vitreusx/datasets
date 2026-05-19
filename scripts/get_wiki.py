"""Download Wikipedia multistream dumps to a local directory."""

from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download, download_and_extract


class Args(BaseModel):
    """CLI arguments for the Wikipedia dump downloader."""

    data_root: str
    """Output directory in which to save the files."""
    lang: Literal["en", "fr"]
    """Language variant of Wikipedia dump to download."""
    version: str = "20250801"
    """Version of the Wikipedia dump to download (YYYYMMDD format)."""


def main(args: Args) -> None:
    """Download Wikipedia multistream dumps to a local directory."""
    data_root = Path(args.data_root)

    prefix = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.version}"
    lang, ver = args.lang, args.version
    index_url = f"{prefix}/{lang}wiki-{ver}-pages-articles-multistream-index.txt.bz2"
    articles_url = f"{prefix}/{lang}wiki-{ver}-pages-articles-multistream.xml.bz2"

    dest = data_root / PurePosixPath(urlparse(index_url).path).name
    download_and_extract(index_url, dest)

    dest = data_root / PurePosixPath(urlparse(articles_url).path).name
    download(articles_url, dest)


if __name__ == "__main__":
    main(tyro.cli(Args))
