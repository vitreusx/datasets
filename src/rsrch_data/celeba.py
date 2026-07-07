"""CelebA dataset loader.

Note: this loads the Kaggle mirror of CelebA
(https://www.kaggle.com/datasets/jessicali9530/celeba-dataset), not the
original Google Drive release -- the official host commonly hits Google's
per-file download quota. The mirror omits identity annotations but keeps
images, attributes, and the train/val/test partition.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from PIL import Image

from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """A CelebA sample."""

    image: Image.Image
    attrs: dict[str, bool]


@register_dataset("celeba")
class CelebA(Sequence):
    """CelebA dataset (Kaggle mirror).

    File structure:
    ```
    <data_root>/
    ├── img_align_celeba/
    │   └── img_align_celeba/
    │       └── {file_id}.jpg
    ├── list_attr_celeba.csv
    └── list_eval_partition.csv
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        data_root = Path(data_root)
        self.img_dir = data_root / "img_align_celeba" / "img_align_celeba"

        partition = pd.read_csv(
            data_root / "list_eval_partition.csv",
            index_col="image_id",
        )
        attrs = pd.read_csv(data_root / "list_attr_celeba.csv", index_col="image_id")

        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self.file_ids: list[str] = list(
            partition.index[partition["partition"] == split_id]
        )

        self.attr_names = list(attrs.columns)
        # Attributes are encoded as {-1, 1} in the CSV; remap to {0, 1}.
        self.attrs = ((attrs.loc[self.file_ids].to_numpy() + 1) // 2).astype(np.int64)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.file_ids)

    def __getitem__(self, index: int) -> Sample:
        """Return sample dict (image, attrs) for the given index."""
        image = Image.open(self.img_dir / self.file_ids[index])
        values = self.attrs[index].astype(bool).tolist()
        attrs = dict(zip(self.attr_names, values, strict=True))
        return {"image": image, "attrs": attrs}
