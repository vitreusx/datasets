"""NYU Depth V2 dataset loader."""

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from ruamel.yaml import YAML

from rsrch_data.registry import register_dataset
from rsrch_data.types.sem_seg import Metadata, Sample


@register_dataset("nyu-depth-v2")
class NYUDepthV2(Sequence):
    """NYU Depth V2 dataset."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        mat_file = self.root / "nyu_depth_v2_labeled.mat"
        self._f = h5py.File(mat_file)

    def __len__(self):
        return int(self._f["images"].shape[0])

    def __getitem__(self, index: int) -> Sample:
        image_nd = self._f["images"][index]
        image = Image.fromarray(image_nd, mode="RGB")
        seg_map_nd = self._f["labels"]
        seg_map = Image.fromarray(seg_map_nd, mode="I;16")
        return {"image": image, "labels": seg_map}

    def _compute_meta(self) -> dict:
        names = ["unlabeled"]  # Label zero is ignored
        for ref in self._f["names"][0]:
            value_ds = self._f[ref]
            name = np.asarray(value_ds).astype(np.uint8).tobytes().decode("utf-8")
            names.append(name)

        classes = dict(enumerate(names))
        return {"classes": classes, "ignore_index": 0}

    @staticmethod
    def meta() -> Metadata:
        """Return semantic segmentation metadata loaded from the bundled YAML."""
        yaml = YAML(typ="safe", pure=True)
        with (Path(__file__).parent / "nyu_depth_v2.yml").open() as f:
            data = yaml.load(f)
        return Metadata(**data)
