"""OpenWebText data loading."""

from pathlib import Path
from typing import Literal, TypedDict

from rsrch_data.parquet import ParquetDataset
from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """OpenWebText sample."""

    text: str


@register_dataset("openwebtext")
class OpenWebText(ParquetDataset[Sample]):
    """Iterable loader over OpenWebText parquet files."""

    def __init__(
        self,
        data_root: str | Path,
        batch_size: int,
        split: Literal["train"] = "train",
    ):
        data_root = Path(data_root)
        pq_files = sorted([*(data_root / "plain_text").glob(f"{split}-*.parquet")])
        super().__init__(pq_files, batch_size=batch_size)
        self.data_root = data_root
        self.split = split
