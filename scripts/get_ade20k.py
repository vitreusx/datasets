"""Download ADE-20k (MIT Scene Parsing) to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Args(BaseModel):
    """CLI args for the script."""

    data_root: Path
    """Directory to write the dataset"""


URLS = (
    "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip",
    "http://sceneparsing.csail.mit.edu/data/ChallengeData2017/release_test.tar",
)


def main(args: Args) -> None:
    """Download ADE-20k (MIT Scene Parsing) to cfg.data_root."""
    args.data_root.mkdir(parents=True, exist_ok=True)
    for url in URLS:
        download_and_extract(url=url, dest_dir=args.data_root)


if __name__ == "__main__":
    main(tyro.cli(Args))
