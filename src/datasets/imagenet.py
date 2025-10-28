from pathlib import Path
from typing import Literal

import pandas as pd
from PIL import Image

from .meta import ClsMeta
import xml.etree.ElementTree as ET


def _get_label_names(loc_synset_mapping_txt: str | Path):
    names = []
    with open(loc_synset_mapping_txt, "r") as f:
        for line in f:
            line = line.rstrip()
            defs = line[line.index(" ") + 1 :]
            pos = defs.find(",")
            name = defs if pos < 0 else defs[:pos]
            names.append(name)
    return names


class ImageNet:
    """ImageNet dataset.

    The dataset may also be a subset of IN-1k or any compatible one.

    File structure:
    ```
    <data_root>/
    ├── ILSVRC/
    │   ├── Annotations/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.xml
    │   │   └── val/
    │   │       └── {img_id}.xml
    │   ├── Data/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.JPEG
    │   │   ├── val/
    │   │   │   └── {img_id}.JPEG
    │   │   └── test/
    │   │       └── {img_id}.JPEG
    │   └── ImageSets/CLS-LOC/
    │       ├── train_cls.txt
    │       ├── train_loc.txt
    │       ├── val.txt
    │       └── test.txt
    └── LOC_synset_mapping.txt     # A list of labels with WordNet IDs
    ```
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        super().__init__()
        self.root = Path(root).expanduser()
        self.split = split

        self.img_root = self.root / "ILSVRC/Data/CLS-LOC" / split
        self.ann_root = self.root / "ILSVRC/Annotations/CLS-LOC" / split

        cls_lists = {"train": "train_cls.txt", "val": "val.txt", "test": "test.txt"}
        cls_list = self.root / "ILSVRC/ImageSets/CLS-LOC" / cls_lists[split]

        self._paths: list[str] = []
        with open(cls_list, "r") as f:
            for line in f:
                path, index = line.strip().split(" ")
                if int(index) - 1 != len(self._paths):
                    raise RuntimeError("Invalid class order")
                self._paths.append(path)

        self.wnid_to_label = {}
        with open(self.root / "LOC_synset_mapping.txt", "r") as f:
            for label, line in enumerate(f):
                line = line.strip()
                pos = line.index(" ")
                wnid = line[:pos]
                self.wnid_to_label[wnid] = label

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        img_path = self.img_root / (path + ".JPEG")
        img = Image.open(img_path).convert("RGB")

        if self.split == "train":
            label = img_path.parents[1].name
            return {"image": img, "label": label}
        elif self.split == "val":
            with open(self.ann_root / (path + ".xml"), "r") as f:
                ann = ET.parse(f)
                label = self.wnid_to_label[ann.find(".//object/name").text]
            return {"image": img, "label": label}
        elif self.split == "test":
            return img

    def meta(self):
        loc_synset_mapping_txt = self.root / "LOC_synset_mapping.txt"
        label_names = _get_label_names(loc_synset_mapping_txt)
        classes = dict(enumerate(label_names))
        return ClsMeta({"classes": classes, "ignore_index": None})
