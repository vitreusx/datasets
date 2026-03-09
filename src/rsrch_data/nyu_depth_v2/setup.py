from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download

URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"


class Args(BaseModel):
    source: str = (
        "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    )
    """URL for the nyu_depth_v2_labeled.mat file."""
    data_root: str
    """Output directory in which to place the dataset."""


def setup(data_root: str | Path, source: str = URL):
    """Setup NYU-Depth-V2.

    :param data_root: Output directory in which to place the dataset.
    :param source: URL for the `nyu_depth_v2_labeled.mat` file."""

    data_root = Path(data_root)

    download(
        url=source,
        dest_path=data_root / "nyu_depth_v2_labeled.mat",
    )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
