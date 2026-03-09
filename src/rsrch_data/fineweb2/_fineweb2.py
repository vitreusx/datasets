from pathlib import Path
from typing import Literal, Sequence, TypedDict

from rsrch_data.utils.parquet import Concat, ParquetFile


class Item(TypedDict):
    text: str
    id: str
    dump: str
    url: str
    date: str
    file_path: str
    language: str
    language_script: str
    language_score: float
    minhash_cluster_size: int
    top_langs: dict[str, float]


def load_fineweb2(
    data_root: str | Path,
    subfolder: Literal["data/eng_Latn", "data/fra_Latn"],
    split: Literal["train", "val"] = "train",
) -> Sequence[Item]:
    data_folder = Path(data_root) / subfolder / split
    pq_files = sorted([*data_folder.glob("*.parquet")])
    return Concat([ParquetFile(pqf) for pqf in pq_files])
