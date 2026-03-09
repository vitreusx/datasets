import subprocess
import sys
from pathlib import Path

import tyro
from pydantic import BaseModel


class Args(BaseModel):
    data_root: str


def setup(data_root: str):
    data_root = Path(data_root)
    subprocess.check_call(
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
        ]
    )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
