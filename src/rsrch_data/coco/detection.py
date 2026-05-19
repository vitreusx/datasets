"""COCO object detection dataset loader."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

from PIL import Image
from pycocotools.coco import COCO
from ruamel.yaml import YAML

from rsrch_data.types.object_det import Metadata


class Box(NamedTuple):
    """Bounding box in (x, y, width, height) format."""

    x: float
    y: float
    width: float
    height: float


class Detection(TypedDict):
    """A single object detection annotation."""

    category: int
    bbox: Box
    iscrowd: bool | None


class Sample(TypedDict):
    """A COCO detection sample."""

    image: Image
    objects: list[Detection]


class COCODetection(Sequence):
    """COCO detection dataset (bounding boxes only)."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.root = Path(root).expanduser()
        self.split = split

        ann_path = self.root / f"annotations/instances_{split}2017.json"
        self.coco = COCO(ann_path)

        self.img_ids = self.coco.getImgIds()
        self.img_root = self.root / f"{split}2017"

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index: int) -> Sample:
        img_id = self.img_ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_root / img_info["file_name"]
        img = Image.open(img_path)

        ann_ids = self.coco.getAnnIds(img_id)

        detections = [
            {
                "category": coco_ann["category_id"],
                "bbox": Box(coco_ann["bbox"]),
                "iscrowd": coco_ann["iscrowd"] > 0,
            }
            for coco_ann in self.coco.loadAnns(ann_ids)
        ]

        return {"image": img, "objects": detections}

    @staticmethod
    def meta() -> Metadata:
        """Return class metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "coco.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
