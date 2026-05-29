"""FineWeb-Edu data loading."""

from pathlib import Path
from typing import Literal, TypedDict

from rsrch_data.parquet import ParquetLoader


class Batch(TypedDict):
    """FineWeb-Edu batch."""

    text: list[str]
    id: list[str]
    dump: list[str]
    url: list[str]
    file_path: list[str]
    language: list[str]
    language_score: list[float]
    token_count: list[int]
    score: list[float]
    int_score: list[int]


class FinewebEduLoader(ParquetLoader[Batch]):
    """Iterable loader over FineWeb-Edu parquet files."""

    def __init__(
        self,
        data_root: str | Path,
        subset: Literal["sample-10BT"],
        batch_size: int,
    ):
        subdir = {"sample-10BT": "sample/10BT"}[subset]
        subset_root = Path(data_root) / subdir
        pq_files = sorted([*subset_root.glob("*.parquet")])
        super().__init__(pq_files, batch_size)
        self.data_root = Path(data_root)
