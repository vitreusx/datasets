"""COCO-Stuff semantic segmentation dataset loader."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from PIL import Image
from ruamel.yaml import YAML

from rsrch_data.types.sem_seg import Metadata, Sample

if TYPE_CHECKING:
    from .utils.schema import DetectAnnFile


class COCOStuff(Sequence):
    """COCO-Stuff semantic segmentation dataset."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root).expanduser()
        self.split = split

        ann_path = self.root / f"annotations/stuff_{split}2017.json"
        with ann_path.open() as f:
            self.ann_file: DetectAnnFile = json.load(f)

        self.img_root = self.root / f"{split}2017"
        self.seg_root = self.root / f"annotations/stuff_{split}2017_pixelmaps"

    def __len__(self):
        return len(self.ann_file["images"])

    def __getitem__(self, index: int) -> Sample:
        img_info = self.ann_file["images"][index]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path)

        seg_map_path = self.seg_root / f"{img_path.stem}.png"
        seg_map = Image.open(seg_map_path)

        return {"image": img, "labels": seg_map}

    def _compute_meta(self) -> dict:
        classes = {}
        for cat in self.ann_file["categories"]:
            classes[cat["id"]] = {
                "name": cat["name"],
                "supercategory": cat["supercategory"],
            }

        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta() -> Metadata:
        """Return semantic segmentation metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "coco_stuff.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
