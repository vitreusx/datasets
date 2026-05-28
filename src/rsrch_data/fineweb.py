"""FineWeb-Edu data loading."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

import pyarrow.parquet as pq


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


class FinewebEduLoader(Iterable[Batch]):
    """Iterable loader over FineWeb-Edu parquet files."""

    def __init__(
        self,
        data_root: str | Path,
        subset: Literal["sample-10BT"],
        batch_size: int,
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        subdir = {"sample-10BT": "sample/10BT"}[subset]
        subset_root = data_root / subdir
        self._pq_files = sorted([*subset_root.glob("*.parquet")])

    def __len__(self) -> int:
        total = 0
        for pq_file_path in self._pq_files:
            pf = pq.ParquetFile(pq_file_path)
            total += (pf.metadata.num_rows + self.batch_size - 1) // self.batch_size
        return total

    def __iter__(self):
        for pq_file_path in self._pq_files:
            pf = pq.ParquetFile(pq_file_path)
            for batch in pf.iter_batches(self.batch_size):
                yield batch.to_pydict()
