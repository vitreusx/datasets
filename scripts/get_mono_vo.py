"""Download Mono-VO sequences to a local directory."""

from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Args(BaseModel):
    """CLI arguments for the Mono-VO downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    source: str = "https://vision.im.tum.de/mono"
    """Base URL for the Mono-VO dataset archives."""
    remove_archives: bool = True
    """Whether to remove archive(s) after extraction."""


def main(args: Args) -> None:
    """Download Mono-VO sequences to a local directory."""
    data_root = Path(args.data_root)

    for name in (
        "all_calib_sequences.zip",
        "all_sequences.zip",
    ):
        download_and_extract(
            url=urljoin(args.source, name),
            dest_dir=data_root / name,
            archive_dest_dir=None if args.remove_archives else data_root,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
