"""Download DocLayNet to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Config(BaseModel):
    """Configuration for the `get_doclaynet` script."""

    data_root: Path
    """Directory to write the dataset"""

    part1_url: str = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    """URL for Part 1 of mip-NeRF 360"""

    part2_url: str = (
        "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
    )
    """URL for Part 2 of mip-NeRF 360."""


def main(cfg: Config) -> None:
    """Download mip-NeRF 360 to cfg.data_root."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)

    print("Downloading mip-NeRF 360...")
    download_and_extract(cfg.part1_url, cfg.data_root / "part1")
    download_and_extract(cfg.part2_url, cfg.data_root / "part2")


if __name__ == "__main__":
    main(tyro.cli(Config))
