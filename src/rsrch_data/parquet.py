"""Generic iterable loader for Parquet files."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Generic, TypeVar

import pyarrow.parquet as pq

SampleT = TypeVar("SampleT")


class ParquetDataset(Iterable[SampleT], Generic[SampleT]):
    """Iterates over one or more Parquet files."""

    def __init__(self, pq_files: list[str | Path], batch_size: int):
        self._pq_files = pq_files
        self.batch_size = batch_size

    def __len__(self) -> int:
        total = 0
        for pq_file_path in self._pq_files:
            pf = pq.ParquetFile(pq_file_path)
            total += pf.metadata.num_rows
        return total

    def __iter__(self) -> Iterator[SampleT]:
        for pq_file_path in self._pq_files:
            pf = pq.ParquetFile(pq_file_path)
            for batch in pf.iter_batches(self.batch_size):
                yield from batch.to_pylist()
