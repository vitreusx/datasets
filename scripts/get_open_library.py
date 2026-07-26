"""Download the Open Library data dump to a local directory."""

from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download

DUMP_URL = "https://openlibrary.org/data/ol_dump_latest.txt.gz"


class Args(BaseModel):
    """CLI args for the Open Library dump downloader."""

    data_root: str
    """Output directory in which to save the dump."""


def main(args: Args) -> None:
    """Download the Open Library data dump to a local directory."""
    data_root = Path(args.data_root)
    dest = data_root / PurePosixPath(urlparse(DUMP_URL).path).name
    download(DUMP_URL, dest)


if __name__ == "__main__":
    main(tyro.cli(Args))
