from pathlib import Path
import pyarrow.parquet as pq
from typing import Literal


class Wikitext:
    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "test", "validation"] = "train",
    ):
        self.data_root = Path(data_root)
        self.split = split

        shards = sorted([*(self.data_root / "data").glob(f"{split}-*.parquet")])
        dataset = pq.ParquetDataset(shards)
        self.df = dataset.read().to_pandas()["text"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> str:
        return self.df[index]
