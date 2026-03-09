from pathlib import Path
from urllib.parse import urljoin

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download


class Args(BaseModel):
    source: str = "http://yann.lecun.com/exdb/mnist/"
    """Source URL for the dataset."""
    data_root: str
    """Output directory in which to place the dataset."""


def setup(
    data_root: str,
    base_url: str = "http://yann.lecun.com/exdb/mnist/",
):
    """Setup MNIST.

    :param data_root: Output directory in which to place the dataset.
    :param source: Source URL for the dataset.
    """

    data_root = Path(data_root)

    for name in (
        "t10k-images-idx3-ubyte.gz",
        "t10k-images-idx1-ubyte.gz",
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
    ):
        source = urljoin(base_url, name)
        download(
            url=source,
            dest_path=data_root / name,
        )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
