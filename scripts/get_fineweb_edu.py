"""Download fineweb-edu to a local directory."""

from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel

from rsrch_data.utils import hf


class Args(BaseModel):
    """CLI args for the `get_fineweb_edu.py` script."""

    data_root: Path
    """Directory to write the dataset"""

    subset: Literal["sample-10BT"]
    """Subset of the dataset to download"""


def main(args: Args) -> None:
    """Download fineweb-edu to cfg.data_root."""
    args.data_root.mkdir(parents=True, exist_ok=True)

    allow_patterns_map = {
        "sample-10BT": ["sample/10BT/*"],
    }

    print("Downloading mip-NeRF 360...")
    hf.fetch(
        dataset_id="HuggingFaceFW/fineweb-edu",
        data_root=args.data_root,
        allow_patterns=allow_patterns_map[args.subset],
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
