"""Download DocLayNet to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download_and_extract


class Config(BaseModel):
    """Configuration for the `get_doclaynet` script."""

    data_root: Path
    """Directory to write the dataset"""

    core_zip_url: str = "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-doclaynet/1.0.0/DocLayNet_core.zip"
    """URL for DocLayNet_core.zip file."""


def main(cfg: Config) -> None:
    """Download DocLayNet to a local directory."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)

    print("Downloading DocLayNet...")
    download_and_extract(
        url=cfg.core_zip_url,
        dest_dir=cfg.data_root / "DocLayNet_core",
        archive_dest_dir=cfg.data_root,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
