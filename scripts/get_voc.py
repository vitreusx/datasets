"""Download Pascal VOC2012 to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Args(BaseModel):
    """CLI arguments for the VOC2012 downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    source: str = (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    )
    """URL for the VOCtrainval_11-May-2012.tar file."""
    remove_archives: bool = True
    """Whether to remove archive(s) after extraction."""


def main(args: Args) -> None:
    """Download and extract VOC2012 to a local directory."""
    data_root = Path(args.data_root)

    download_and_extract(
        url=args.source,
        dest_dir=data_root,
        archive_dest_dir=None if args.remove_archives else data_root,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
