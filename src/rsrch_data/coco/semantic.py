"""COCO semantic segmentation dataset (derived from panoptic annotations)."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from rsrch_data.types.sem_seg import Metadata, Sample

from .panoptic import COCOPanoptic


class COCOSemantic(Sequence):
    """A custom dataset for semantic segmentation from COCO-Panoptic."""

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        super().__init__()
        self._panoptic = COCOPanoptic(root=root, split=split)

    def __len__(self):
        return len(self._panoptic)

    def __getitem__(self, index: int) -> Sample:
        item = self._panoptic[index]
        ids = item["ids"]
        uniq, inv = np.unique(ids, return_inverse=True)
        inv = inv.reshape(ids.shape)
        id_to_cat = {info["id"]: info["category_id"] for info in item["segments"]}
        cat_map = np.array(
            [id_to_cat.get(seg_id, 0) for seg_id in uniq], dtype=np.uint8
        )
        labels = cat_map[inv]
        return {"image": item["image"], "labels": Image.fromarray(labels)}

    @staticmethod
    def meta() -> Metadata:
        """Return semantic segmentation metadata derived from the panoptic metadata."""
        panoptic = COCOPanoptic.meta()
        return Metadata(panoptic.classes, panoptic.ignore_index)
