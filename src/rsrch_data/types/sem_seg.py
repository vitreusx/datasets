"""Semantic segmentation metadata types."""

from dataclasses import make_dataclass
from functools import cached_property
from pprint import pformat
from typing import TypedDict

import numpy as np
from PIL import Image

from rsrch_data.utils.colors import Palette, get_palette, hex2rgb

from .utils import is_contiguous


class Sample(TypedDict):
    """A semantic segmentation sample."""

    image: Image.Image
    labels: Image.Image


class SegLabelInfo(TypedDict):
    """Per-class info for semantic segmentation."""

    name: str
    palette: str | tuple[int, int, int] | None


class Metadata:
    """Metadata for semantic segmentation."""

    def __init__(
        self,
        classes: dict[int, str | SegLabelInfo],
        ignore_index: int | None = None,
        ignore_color: tuple[int, int, int] | str | None = None,
    ):
        self.classes = classes
        self.ignore_index = ignore_index
        self.ignore_color = ignore_color
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

    @property
    def num_classes(self) -> int:
        """Total number of classes."""
        return len(self.label_to_name)

    @cached_property
    def names(self) -> list[str]:
        """Ordered list of class names."""
        return [self.label_to_name[label] for label in self.label_to_name]

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
            msg = "Labels' list must be contiguous"
            raise ValueError(msg)
        if self.ignore_color is None:
            msg = "Need to provide ignore_color"
            raise ValueError(msg)
        return map_

    @cached_property
    def palette(self) -> Palette:
        """Return a ``Palette`` for visualizing segmentation maps."""
        color_map = self.label_to_color
        if color_map is not None:
            colors = np.array([color_map[label] for label in color_map])
            colors = colors.astype(np.uint8)
            ignore_color = self.ignore_color
            if isinstance(ignore_color, str):
                ignore_color = hex2rgb(ignore_color)
            return Palette(colors, ignore_color)
        return get_palette(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

    def __repr__(self) -> str:
        """Return pretty-printed representation."""
        cls = make_dataclass(
            self.__class__.__name__,
            ["names", "ignore_index"],
        )
        return pformat(cls(self.names, self.ignore_index))
