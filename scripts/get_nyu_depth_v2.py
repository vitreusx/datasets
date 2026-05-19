"""Download NYU Depth V2 to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download


class Args(BaseModel):
    """CLI arguments for the NYU Depth V2 downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    source: str = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    """URL for the nyu_depth_v2_labeled.mat file."""


def main(args: Args) -> None:
    """Download NYU Depth V2 to a local directory."""
    data_root = Path(args.data_root)

    download(
        url=args.source,
        dest_path=data_root / "nyu_depth_v2_labeled.mat",
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
