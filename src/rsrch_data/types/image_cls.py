"""Image classification metadata types."""

from dataclasses import make_dataclass
from functools import cached_property
from pprint import pformat
from typing import TypedDict

from PIL import Image

from .utils import is_contiguous


class Sample(TypedDict):
    """An image classification sample."""

    image: Image.Image
    label: int


class Metadata:
    """Metadata for image classification."""

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
