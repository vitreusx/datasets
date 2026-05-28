"""OpenWebText data loading."""

from collections.abc import Iterable
from pathlib import Path
from typing import Literal, TypedDict

import pyarrow.parquet as pq


class Batch(TypedDict):
    text: list[str]


class OpenWebTextLoader(Iterable[Batch]):
    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
        split: Literal["train"] = "train",
    ):
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.split = split

        self._pq_files = sorted(
            [*(self.data_root / "plain_text").glob(f"{self.split}-*.parquet")]
        )

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
