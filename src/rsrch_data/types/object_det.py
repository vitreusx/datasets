"""Object detection metadata types."""

from dataclasses import make_dataclass
from functools import cached_property
from pprint import pformat
from typing import NamedTuple, TypedDict

from PIL import Image

from .utils import is_contiguous


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


class Sample(TypedDict):
    """An object detection sample."""

    image: Image.Image
    dets: Detection


class Metadata:
    """Metadata for object detection."""

    def __init__(
        self,
        classes: dict[int, str],
        ignore_index: int | None = None,
    ):
        self.classes = classes
        self.ignore_index = ignore_index
        if not is_contiguous([*self.classes]):
            msg = "Class labels must be contiguous"
            raise RuntimeError(msg)

    @property
    def label_to_name(self) -> dict[int, str]:
        """Map from integer label to human-readable class name."""
        return self.classes

    @property
    def num_classes(self) -> int:
        """Total number of classes."""
        return len(self.label_to_name)

    @cached_property
    def names(self) -> list[str]:
        """Ordered list of class names."""
        return [self.label_to_name[label] for label in self.label_to_name]

    def __repr__(self) -> str:
        """Return pretty-printed representation."""
        cls = make_dataclass(
            self.__class__.__name__,
            ["names", "ignore_index"],
        )
        return pformat(cls(self.names, self.ignore_index))
