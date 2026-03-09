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
    minhash_cluster_size: int


def load_fineweb(
    data_root: str | Path,
    subfolder: Literal["sample/10BT"],
) -> Sequence[Item]:
    data_folder = Path(data_root) / subfolder
    pq_files = sorted([*data_folder.glob("*.parquet")])
    return Concat([ParquetFile(pqf) for pqf in pq_files])
