from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pyarrow.parquet as pq


class Batch(TypedDict):
    """FineWeb-Edu batch."""

    text: list[str]
    id: list[str]
    dump: list[str]
    url: list[str]
    file_path: list[str]
    language: list[str]
    language_score: np.ndarray
    token_count: np.ndarray
    score: np.ndarray
    int_score: np.ndarray


def fineweb_loader(
    data_root: str | Path,
    subset: Literal["sample-10BT"],
    batch_size: int,
):
    data_root = Path(data_root)
    subdir = {"sample-10BT": "sample/10BT"}[subset]
    subset_root = data_root / subdir
    pq_files = sorted([*subset_root.glob("*.parquet")])

    for pq_file_path in pq_files:
        pf = pq.ParquetFile(pq_file_path)
        for batch in pf.iter_batches(batch_size):
            batch_py = batch.to_pydict()
            for k in ("language_score", "token_count", "score", "int_score"):
                batch_py[k] = np.array(batch_py[k])
            yield batch_py
