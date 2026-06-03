"""Download CIFAR-10 and/or CIFAR-100 to a local directory."""

import tarfile
from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download, get_url_filename

_SOURCES = {
    "cifar-10": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar-100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}

_EXTRACTED_DIRS = {
    "cifar-10": "cifar-10-batches-py",
    "cifar-100": "cifar-100-python",
}


class Args(BaseModel):
    """CLI arguments for the CIFAR downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    dataset: Literal["cifar-10", "cifar-100", "both"] = "both"
    """Which dataset(s) to download."""
    remove_archives: bool = True
    """Whether to remove downloaded archives after extraction."""


def _download_one(name: str, data_root: Path, *, remove_archives: bool) -> None:
    """Download and extract a single CIFAR variant."""
    if (data_root / _EXTRACTED_DIRS[name]).exists():
        return

    url = _SOURCES[name]
    archive = data_root / get_url_filename(url)
    download(url=url, dest_path=archive)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(data_root)  # noqa: S202
    if remove_archives:
        archive.unlink()


def main(args: Args) -> None:
    """Download CIFAR-10 and/or CIFAR-100 to a local directory."""
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    datasets = list(_SOURCES) if args.dataset == "both" else [args.dataset]
    for name in datasets:
        _download_one(name, data_root, remove_archives=args.remove_archives)


if __name__ == "__main__":
    main(tyro.cli(Args))
