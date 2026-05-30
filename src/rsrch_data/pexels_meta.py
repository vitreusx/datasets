"""Pexels photo metadata loader."""

from pathlib import Path
from typing import TypedDict

from rsrch_data.parquet import ParquetDataset
from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """Sample from Pexels photo metadata records."""

    id: int
    width: float
    height: float
    url: str
    small_url: str
    medium_url: str
    large_url: str
    license: str


@register_dataset("pexels-meta")
class PexelsMeta(ParquetDataset[Sample]):
    """Loads Pexels photo metadata from a local parquet file."""

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
    ):
        data_root = Path(data_root)
        super().__init__([data_root / "photos_sequential.parquet"], batch_size)
        self.data_root = data_root
