"""CIFAR-10 and CIFAR-100 dataset loaders."""

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from ruamel.yaml import YAML

from rsrch_data.registry import register_dataset
from rsrch_data.types.image_cls import Metadata, Sample


@register_dataset("cifar-10")
class CIFAR10(Sequence):
    """CIFAR-10 dataset.

    File structure:
    ```
    <data_root>/
    └── cifar-10-batches-py/
        ├── data_batch_{1..5} # Train set
        └── test_batch        # Test set
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
    ):
        data_root = Path(data_root)

        batches = {
            "train": [f"data_batch_{idx}" for idx in range(1, 6)],
            "test": ["test_batch"],
        }[split]

        images, labels = [], []
        for fname in batches:
            with (data_root / "cifar-10-batches-py" / fname).open("rb") as f:
                batch = pickle.load(f, encoding="bytes")
            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])

        images = np.concatenate(images)
        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Sample:
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta() -> Metadata:
        """Return class metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "cifar10.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)


@register_dataset("cifar-100")
class CIFAR100(Sequence):
    """CIFAR-100 dataset.

    File structure:
    ```
    <data_root>/
    └── cifar-100-python/
        ├── train          # Train set
        └── test           # Test set
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test"] = "train",
    ):
        data_root = Path(data_root)

        with (data_root / "cifar-100-python" / split).open("rb") as f:
            data = pickle.load(f, encoding="bytes")
            images, labels = data[b"data"], data[b"fine_labels"]

        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> Sample:
        image = Image.fromarray(self.images[index])
        label = self.labels[index]
        return {"image": image, "label": label}

    @staticmethod
    def meta() -> Metadata:
        """Return class metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "cifar100.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
