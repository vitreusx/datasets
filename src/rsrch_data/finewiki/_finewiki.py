from pathlib import Path
from typing import Literal, Sequence, TypedDict

from rsrch_data.utils.parquet import Concat, ParquetFile


class Item(TypedDict):
    pass


def load_finewiki(
    data_root: str | Path,
    subfolder: Literal["data/enwiki", "data/frwiki"],
) -> Sequence[Item]:
    data_folder = Path(data_root) / subfolder
    pq_files = sorted([*data_folder.glob("*.parquet")])
    return Concat([ParquetFile(pqf) for pqf in pq_files])
