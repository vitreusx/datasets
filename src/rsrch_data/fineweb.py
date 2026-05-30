"""FineWeb-Edu data loading."""

from pathlib import Path
from typing import Literal, TypedDict

from rsrch_data.parquet import ParquetDataset
from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """FineWeb-Edu sample."""

    text: str
    id: str
    dump: str
    url: str
    file_path: str
    language: str
    language_score: float
    token_count: int
    score: float
    int_score: int


@register_dataset("fineweb-edu")
class FinewebEdu(ParquetDataset[Sample]):
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
