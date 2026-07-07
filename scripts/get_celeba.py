"""Download CelebA (Kaggle mirror) to a local directory."""

import subprocess
import sys
from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import extract


class Args(BaseModel):
    """CLI arguments for the CelebA downloader."""

    data_root: str
    """Output directory in which to place the dataset."""
    remove_archive: bool = True
    """Whether to remove the downloaded archive after extraction."""


def main(args: Args) -> None:
    """Download CelebA (Kaggle mirror) via the Kaggle CLI and extract it."""
    data_root = Path(args.data_root)
    if (data_root / "img_align_celeba").exists():
        return
    data_root.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(  # noqa: S603
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "datasets",
            "download",
            "-d",
            "jessicali9530/celeba-dataset",
            "-p",
            str(data_root.absolute()),
        ],
    )

    archive_path = data_root / "celeba-dataset.zip"
    extract(archive_path, data_root)

    if args.remove_archive:
        archive_path.unlink()


if __name__ == "__main__":
    main(tyro.cli(Args))
