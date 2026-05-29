"""Pexels photo metadata loader."""

from pathlib import Path
from typing import TypedDict

from rsrch_data.parquet import ParquetLoader


class Batch(TypedDict):
    """One batch of Pexels photo metadata records."""

    id: list[int]
    width: list[float]
    height: list[float]
    url: list[str]
    small_url: list[str]
    medium_url: list[str]
    large_url: list[str]
    license: list[str]


class PexelsMetaLoader(ParquetLoader[Batch]):
    """Loads Pexels photo metadata from a local parquet file."""

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
    ):
        data_root = Path(data_root)
        super().__init__([data_root / "photos_sequential.parquet"], batch_size)
        self.data_root = data_root
