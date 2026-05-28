from collections.abc import Iterator
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


def fineweb_loader(
    data_root: str | Path,
    subset: Literal["sample-10BT"],
    batch_size: int,
) -> Iterator[Batch]:
    data_root = Path(data_root)
    subdir = {"sample-10BT": "sample/10BT"}[subset]
    subset_root = data_root / subdir
    pq_files = sorted([*subset_root.glob("*.parquet")])

    for pq_file_path in pq_files:
        pf = pq.ParquetFile(pq_file_path)
        for batch in pf.iter_batches(batch_size):
            yield batch.to_pydict()
