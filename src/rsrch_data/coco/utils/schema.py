"""TypedDict schemas for COCO annotation file structures."""

from typing import Any, TypedDict


class Info(TypedDict):
    """Top-level dataset info block."""

    description: str
    url: str
    version: str
    year: int
    contributor: str
    date_created: str


class License(TypedDict):
    """Image license entry."""

    url: str
    id: int
    name: str


class Image(TypedDict):
    """Image metadata entry."""

    license: int
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    id: int


class DetectAnn(TypedDict):
    """A single detection annotation."""

    id: int
    image_id: int
    category_id: int
    segmentation: Any
    area: float
    bbox: tuple[int, int, int, int]
    iscrowd: int


class DetectCategory(TypedDict):
    """A detection category entry."""

    supercategory: str
    id: int
    name: str


class DetectAnnFile(TypedDict):
    """Top-level structure of a COCO detection annotation file."""

    info: Info
    licenses: list[License]
    images: list[Image]
    annotations: list[DetectAnn]
    categories: list[DetectCategory]


class SegmentInfo(TypedDict):
    """Per-segment info within a panoptic annotation."""

    id: int
    category_id: int
    area: int
    bbox: tuple[int, int, int, int]
    iscrowd: int


class PanopticAnn(TypedDict):
    """A single panoptic annotation (one per image)."""

    image_id: int
    file_name: str
    segments_info: list[SegmentInfo]


class PanopticCategory(TypedDict):
    """A panoptic category entry."""

    id: int
    name: str
    supercategory: str
    isthing: int
    color: tuple[int, int, int]


class PanopticAnnFile(TypedDict):
    """Top-level structure of a COCO panoptic annotation file."""

    info: Info
    licenses: list[License]
    images: list[Image]
    annotations: list[PanopticAnn]
    categories: list[PanopticCategory]
