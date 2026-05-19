"""Panoptic segmentation metadata types."""

from functools import cached_property
from typing import TypedDict

import numpy as np
from PIL import Image

from rsrch_data.utils.colors import hex2rgb

from .utils import is_contiguous


class Sample(TypedDict):
    """A panoptic segmentation sample."""

    image: Image.Image
    labels: Image.Image
    inst_ids: Image.Image


class PanopticLabelInfo(TypedDict):
    """Per-class info for panoptic segmentation."""

    name: str
    isthing: bool
    palette: str | tuple[int, int, int] | None


class Metadata:
    """Metadata for a panoptic segmentation dataset."""

    def __init__(
        self,
        classes: dict[int, str | PanopticLabelInfo],
        ignore_index: int | None = None,
    ):
        self.classes = classes
        self.ignore_index = ignore_index
        if not is_contiguous([*self.classes]):
            msg = "Class labels must be contiguous"
            raise RuntimeError(msg)

    @cached_property
    def label_to_name(self) -> dict[int, str]:
        """Map from integer label to human-readable class name."""
        return {
            label: (info if isinstance(info, str) else info["name"])
            for label, info in self.classes.items()
        }

    def is_stuff(self, label: int) -> bool:
        """Return whether *label* is a stuff (non-instance) category."""
        return self.classes[label]["isthing"]

    @cached_property
    def num_classes(self) -> int:
        """Total number of classes."""
        return len(self.label_to_name)

    @cached_property
    def names(self) -> list[str]:
        """Ordered list of class names."""
        map_ = self.label_to_name
        return [map_[label] for label in map_]

    @cached_property
    def label_to_color(self) -> dict[int, tuple[int, int, int]] | None:
        """Map from label to RGB color, or ``None`` if no palette is defined."""
        map_ = {}
        for label, info in self.classes.items():
            if not isinstance(info, dict) or info.get("palette") is None:
                return None
            color = info["palette"]
            if isinstance(color, str):
                color = hex2rgb(color)
            map_[label] = color

        if not is_contiguous([*map_]):
            msg = "Labels must be contiguous"
            raise ValueError(msg)

        return map_

    @cached_property
    def palette(self) -> np.ndarray | None:
        """Return an ``(N, 3)`` uint8 array of colors, or ``None`` if no palette."""
        map_ = self.label_to_color
        if map_ is None:
            return None

        palette = np.array([map_[label] for label in map_])
        return palette.astype(np.uint8)
