from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download_and_extract


class Args(BaseModel):
    source: str = (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    )
    """An URL for the VOCtrainval_11-May-2012.tar file."""
    remove_archives: bool = True
    """Whether to remove the archives after extraction."""
    data_root: str
    """Output directory in which to place the dataset."""


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    if args.remove_archives:
        archive_dest = None
    else:
        file_name = PurePosixPath(urlparse(args.url).path).name
        archive_dest = data_root / file_name

    download_and_extract(
        url=args.source,
        dest=data_root,
        archive_dest=archive_dest,
    )


if __name__ == "__main__":
    main()
