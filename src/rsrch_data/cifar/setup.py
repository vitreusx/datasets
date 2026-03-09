from pathlib import Path
from urllib.parse import urljoin

import tyro

from rsrch_data.utils.download import download_and_extract


def setup(
    data_root: str,
    base_url: str = "https://www.cs.toronto.edu/~kriz",
    remove_archives: bool = False,
):
    """Setup CIFAR-10 and CIFAR-100.

    :param data_root: Output directory in which to place the dataset.
    :param base_url: Source URL for the dataset.
    """

    data_root = Path(data_root)

    for name in ("cifar-10-python.tar.gz", "cifar-100-python.tar.gz"):
        source = urljoin(base_url, name)
        download_and_extract(
            url=source,
            dest_dir=data_root,
            archive_dest_dir=None if remove_archives else data_root,
        )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
