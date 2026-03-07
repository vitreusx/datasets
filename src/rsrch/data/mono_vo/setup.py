from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download_and_extract


class Args(BaseModel):
    source: str = "https://vision.in.tum.de/mono"
    remove_archives: bool = True
    data_root: str


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    for name in (
        "all_calib_sequences.zip",
        "all_sequences.zip",
    ):
        archive_dest = None if args.remove_archives else data_root / name
        download_and_extract(
            url=urljoin(args.source, name),
            dest=data_root / name,
            archive_dest=archive_dest,
        )
