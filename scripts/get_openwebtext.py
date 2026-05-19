"""Download OpenWebText to a local directory (raw text, untokenized).

Saves the full train split as a HuggingFace dataset to <data_root>.
The train/val split is handled at load time by openwebtext_loader.
"""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils import hf


class Config(BaseModel):
    """Configuration for the get_openwebtext script."""

    data_root: Path
    """Directory to write the dataset"""


def main(cfg: Config) -> None:
    """Download OpenWebText to cfg.data_root."""
    cfg.data_root.mkdir(parents=True, exist_ok=True)

    print("Downloading OpenWebText...")
    hf.fetch(
        "Skylion007/openwebtext",
        allow_patterns=["plain_text/*", "README.md"],
        data_root=cfg.data_root,
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
