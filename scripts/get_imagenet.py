"""Download ImageNet to local directory."""

import subprocess
import sys
from pathlib import Path

import tyro
from pydantic import BaseModel


class Args(BaseModel):
    """Arguments for the `get_imagenet` script."""

    data_root: str
    """Output directory in which to place the dataset."""


def main(args: Args) -> None:
    """Download the ImageNet competition archive via the Kaggle CLI."""
    data_root = Path(args.data_root)
    subprocess.check_call(  # noqa: S603
        [
            sys.executable,
            "-m",
            "kaggle.cli",
            "competitions",
            "download",
            "-c",
            "imagenet-object-localization-challenge",
            "-p",
            str(data_root.absolute()),
        ],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
