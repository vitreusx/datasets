"""ADE-20k (MIT Scene Parsing) dataset."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict

from PIL import Image
from ruamel.yaml import YAML
from typing_extensions import NotRequired

from rsrch_data.registry import register_dataset
from rsrch_data.types.sem_seg import Metadata


class Sample(TypedDict):
    """Sample from ADE20k (MIT Scene Parsing) dataset.

    Extends the base semantic segmentation sample with optional labels for test split.
    """

    image: Image.Image
    labels: NotRequired[Image.Image]


@register_dataset("ade20k")
class ADE20k(Sequence):
    """ADE20k (MIT Scene Parsing) dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split

        if self.split == "train":
            self._img_dir = self.data_root / "ADEChallengeData2016/images/training"
            self._ann_dir = self.data_root / "ADEChallengeData2016/annotations/training"
        elif self.split == "val":
            self._img_dir = self.data_root / "ADEChallengeData2016/images/validation"
            self._ann_dir = (
                self.data_root / "ADEChallengeData2016/annotations/validation"
            )
        elif self.split == "test":
            self._img_dir = self.data_root / "release_test/testing"
            self._ann_dir = None
        else:
            msg = f"Invalid split name {self.split}"
            raise ValueError(msg)

        self._fnames = sorted([p.name for p in self._img_dir.iterdir()])

    def __len__(self):
        return len(self._fnames)

    def __getitem__(self, index: int) -> Sample:
        img_path = self._img_dir / self._fnames[index]
        image = Image.open(img_path)
        if self._ann_dir is not None:
            ann_path = self._ann_dir / f"{img_path.stem}.png"
            labels = Image.open(ann_path)
            return {"image": image, "labels": labels}
        return {"image": image}

    @staticmethod
    def meta() -> Metadata:
        """Return semantic segmentation metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "ade20k.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
