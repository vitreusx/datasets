"""Download MNIST to a local directory."""

from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download


class Args(BaseModel):
    """CLI arguments for the MNIST downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    source: str = "http://yann.lecun.com/exdb/mnist/"
    """Base URL for the dataset files."""


def main(args: Args) -> None:
    """Download MNIST files to a local directory."""
    data_root = Path(args.data_root)

    for name in (
        "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ):
        download(
            url=urljoin(args.source, name),
            dest_path=data_root / name,
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
