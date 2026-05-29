"""Download pexels-meta to a local directory."""

from pathlib import Path

import tyro
from pydantic import BaseModel

from rsrch_data.utils import hf


class Args(BaseModel):
    """CLI args for the script."""

    data_root: Path
    """Directory to write the dataset"""


def main(args: Args) -> None:
    """Download pexels-meta to cfg.data_root."""
    args.data_root.mkdir(parents=True, exist_ok=True)
    hf.fetch(
        dataset_id="terminusresearch/pexels-metadata-1.71M",
        data_root=args.data_root,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
