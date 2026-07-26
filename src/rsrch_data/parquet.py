"""Generic iterable loader for Parquet files."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Generic, TypeVar

import pyarrow.parquet as pq

SampleT = TypeVar("SampleT")


class ParquetDataset(Iterable[SampleT], Generic[SampleT]):
    """Iterates over one or more Parquet files."""

    def __init__(self, pq_files: list[str | Path], batch_size: int):
        """Wrap a list of Parquet files, iterated in `batch_size` row batches."""
        self._pq_files = pq_files
        self.batch_size = batch_size

    def __len__(self) -> int:
        total = 0
        for pq_file_path in self._pq_files:
            pf = pq.ParquetFile(pq_file_path)
            total += pf.metadata.num_rows
        return total

    def __iter__(self) -> Iterator[SampleT]:
        return self.iter_from(0)

    def _iter_batches_from(
        self, pf: pq.ParquetFile, local_offset: int
    ) -> Iterator[SampleT]:
        for batch in pf.iter_batches(self.batch_size):
            if local_offset >= batch.num_rows:
                local_offset -= batch.num_rows
                continue
            yield from batch.to_pylist()[local_offset:]
            local_offset = 0

    def iter_from(self, start: int = 0) -> Iterator[SampleT]:
        """Iterate samples sequentially, starting at global row `start`.

        File boundaries are skipped for free via each file's row count
        (Parquet footer metadata, no data read); the target file is then
        read in `batch_size` batches as usual, discarding whole batches
        until `start`'s row is reached -- wasted reads are bounded to at
        most one file's worth, not the whole dataset.

        :param start: Global row index to begin at.
        """
        if start < 0:
            start += len(self)
        if not 0 <= start <= len(self):
            msg = f"start={start} out of range for {self!r}"
            raise IndexError(msg)
        if start == len(self):
            return

        remaining = start
        for file_idx, pq_file_path in enumerate(self._pq_files):
            pf = pq.ParquetFile(pq_file_path)
            file_rows = pf.metadata.num_rows
            if remaining >= file_rows:
                remaining -= file_rows
                continue
            yield from self._iter_batches_from(pf, remaining)
            for next_path in self._pq_files[file_idx + 1 :]:
                yield from self._iter_batches_from(pq.ParquetFile(next_path), 0)
            return
