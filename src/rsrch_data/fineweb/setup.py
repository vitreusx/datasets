from pathlib import Path
from typing import Literal

import tyro

from rsrch_data.utils import hf


def setup(
    data_root: str | Path,
    subfolder: Literal["sample/10BT"],
    max_dl_size: int | str | None = None,
    seed: int = 0,
):
    hf.fetch(
        dataset_id="HuggingFaceFW/fineweb",
        data_root=data_root,
        allow_patterns=f"{subfolder}/*",
        max_dl_size=max_dl_size,
        seed=seed,
    )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
