"""Download UVDoc to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Config(BaseModel):
    """Configuration for the `get_uvdoc` script."""

    data_root: Path
    """Directory to write the dataset"""

    uvdoc_final_url: str = "https://igl.ethz.ch/projects/uvdoc/UVDoc_final.zip"
    """URL for DocLayNet_core.zip file."""


def main(cfg: Config) -> None:
    """Download UVDoc to a local directory."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)

    print("Downloading UVDoc...")
    download_and_extract(
        url=cfg.uvdoc_final_url,
        dest_dir=cfg.data_root,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
