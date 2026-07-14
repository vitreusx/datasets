"""CIFAR-10 and CIFAR-100 dataset loaders."""

import hashlib
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from ruamel.yaml import YAML

from rsrch_data.registry import register_dataset
from rsrch_data.types.image_cls import Metadata, Sample

CIFAR10_CHECKSUMS = {
    "data_batch_1": "f962466ef690d46b226450fb9aadc74ba4bc64a76aa526b5827fe4bc5c7125cb",
    "data_batch_2": "766b2cef9fbc745cf056b3152224f7cf77163b330ea9a15f9392beb8b89bc5a8",
    "data_batch_3": "0f00d98ebfb30b3ec0ad19f9756dc2630b89003e10525f5e148445e82aa6a1f9",
    "data_batch_4": "3f7bb240661948b8f4d53e36ec720d8306f5668bd0071dcb4e6c947f78e9682b",
    "data_batch_5": "d91802434d8376bbaeeadf58a737e3a1b12ac839077e931237e0dcd43adcb154",
    "test_batch": "f53d8d457504f7cff4ea9e021afcf0e0ad8e24a91f3fc42091b8adef61157831",
}


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
        """Load CIFAR-10 `split` batches from `data_root`, verifying checksums."""
        self.data_root = Path(data_root)

        batches = {
            "train": [f"data_batch_{idx}" for idx in range(1, 6)],
            "test": ["test_batch"],
        }[split]

        images, labels = [], []
        for fname in batches:
            batch = self._safe_load(fname)
            images.append(batch[b"data"])
            labels.extend(batch[b"labels"])

        images = np.concatenate(images)
        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def _safe_load(self, name: str):
        with (self.data_root / "cifar-10-batches-py" / name).open("rb") as f:
            content = f.read()
            expected = CIFAR10_CHECKSUMS[name]
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected:
                msg = "SHA256 checksums don't match"
                raise ValueError(msg)
            return pickle.loads(content, encoding="bytes")  # noqa: S301

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


CIFAR100_CHECKSUMS = {
    "test": "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864",
    "train": "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b",
}


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
        """Load the CIFAR-100 `split` batch from `data_root`, verifying its checksum."""
        self.data_root = Path(data_root)

        data = self._safe_load(split)
        images, labels = data[b"data"], data[b"fine_labels"]

        images = images.reshape(-1, 3, 32, 32)
        self.images = np.moveaxis(images, 1, -1)
        self.labels = np.array(labels, dtype=np.int32)

    def _safe_load(self, name: str):
        with (self.data_root / "cifar-100-python" / name).open("rb") as f:
            content = f.read()
            expected = CIFAR100_CHECKSUMS[name]
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected:
                msg = "SHA256 checksums don't match"
                raise ValueError(msg)
            return pickle.loads(content, encoding="bytes")  # noqa: S301

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
