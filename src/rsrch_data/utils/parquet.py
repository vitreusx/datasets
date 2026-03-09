from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


class ParquetFile:
    def __init__(self, source: str | Path):
        self.source = source
        self.pq_file = pq.ParquetFile(source)
        meta = self.pq_file.metadata
        row_counts = [
            meta.row_group(idx).num_rows for idx in range(meta.num_row_groups)
        ]
        self._row_counts = np.array(row_counts)
        self._ends = np.cumsum(self._row_counts)
        self._starts = self._ends - self._row_counts
        self._total = meta.num_rows

    def __len__(self):
        return self._total

    def __getitem__(self, index: int):
        row_group = np.searchsorted(self._ends, index, side="right")
        offset = index - self._starts[row_group]
        row_group = self.pq_file.read_row_group(row_group)
        return row_group.take([offset]).to_pylist()[0]

    def __iter__(self):
        for batch in self.pq_file.iter_batches():
            yield from batch.to_pylist()


class Concat:
    def __init__(self, datasets: list):
        self.datasets = datasets
        self._lengths = np.array([len(ds) for ds in datasets])
        self._ends = np.cumsum(self._lengths)
        self._starts = self._ends - self._lengths
        self._total = int(self._lengths.sum())

    def __len__(self):
        return self._total

    def __getitem__(self, index: int):
        ds_index = np.searchsorted(self._ends, index, side="right")
        offset = index - self._starts[ds_index]
        return self.datasets[int(ds_index)][int(offset)]

    def __iter__(self):
        for ds in self.datasets:
            yield from ds
