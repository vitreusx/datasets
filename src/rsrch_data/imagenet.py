"""ImageNet data loading."""

import bisect
import io
import threading
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from rsrch_data.registry import register_dataset
from rsrch_data.types.image_cls import Metadata, Sample


def parse_loc_synset_mapping(path: str | Path) -> pd.DataFrame:
    """Parse `LOC_synset_mapping.txt` file."""
    records = []
    path = Path(path)
    with path.open() as f:
        for line_and_newline in f:
            line = line_and_newline.rstrip()
            pos = line.index(" ")
            wnid, defs = line[:pos], line[pos + 1 :]
            pos = defs.find(",")
            name = defs if pos < 0 else defs[:pos]
            records.append((wnid, name, defs))

    return pd.DataFrame.from_records(
        records,
        columns=["wnid", "name", "defs"],
    )


@register_dataset("imagenet")
class ImageNet(Sequence):
    """ImageNet dataset.

    The dataset may also be a subset of IN-1k or any compatible one.

    File structure:
    ```
    <data_root>/
    ├── ILSVRC/
    │   ├── Annotations/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.xml
    │   │   └── val/
    │   │       └── {img_id}.xml
    │   ├── Data/CLS-LOC/
    │   │   ├── train/
    │   │   │   └── {wnid}/
    │   │   │       └── {img_id}.JPEG
    │   │   ├── val/
    │   │   │   └── {img_id}.JPEG
    │   │   └── test/
    │   │       └── {img_id}.JPEG
    │   └── ImageSets/CLS-LOC/
    │       ├── train_cls.txt
    │       ├── train_loc.txt
    │       ├── val.txt
    │       └── test.txt
    └── LOC_synset_mapping.txt     # A list of labels with WordNet IDs
    └── LOC_train_solution.csv
    └── LOC_val_solution.csv
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val", "test"] = "train",
    ):
        """Index ImageNet `split` image paths and WordNet-ID labels from `data_root`."""
        super().__init__()
        self.root = Path(data_root).expanduser()
        self.split = split

        self.img_root = self.root / "ILSVRC/Data/CLS-LOC" / split
        self.ann_root = self.root / "ILSVRC/Annotations/CLS-LOC" / split

        cls_lists = {"train": "train_cls.txt", "val": "val.txt", "test": "test.txt"}
        cls_list = self.root / "ILSVRC/ImageSets/CLS-LOC" / cls_lists[split]

        self.paths: list[str] = []
        with cls_list.open() as f:
            for line in f:
                path, index = line.strip().split(" ")
                if int(index) - 1 != len(self.paths):
                    msg = "Invalid class order"
                    raise RuntimeError(msg)
                self.paths.append(path)

        synset_df = parse_loc_synset_mapping(self.root / "LOC_synset_mapping.txt")
        self.wnid_to_label = {
            wnid: label for label, wnid in enumerate(synset_df["wnid"])
        }

        if split == "train":
            self.wnids = []
            for path in self.paths:
                wnid = path.split("/")[0]
                self.wnids.append(wnid)
        elif split == "val":
            # We get the IDs from the solution file, because it's faster than
            # parsing all the XML files
            sol_df = pd.read_csv(self.root / "LOC_val_solution.csv")
            wnids_map = {}
            for _, row in sol_df.iterrows():
                pred: str = row["PredictionString"]
                wnid = pred[: pred.find(" ")]
                wnids_map[row["ImageId"]] = wnid
            self.wnids = [wnids_map[path] for path in self.paths]

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.paths)

    def __getitem__(self, idx: int) -> Sample | Image.Image:
        """Return sample dict (image, label) for train/val or raw image for test."""
        path = self.paths[idx]
        img_path = self.img_root / (path + ".JPEG")
        img = Image.open(img_path)

        if self.split in ("train", "val"):
            wnid = self.wnids[idx]
            label = self.wnid_to_label[wnid]
            return {"image": img, "label": label}
        return img

    def meta(self) -> Metadata:
        """Build image-classification metadata from the synset mapping file."""
        loc_synset_mapping_txt = self.root / "LOC_synset_mapping.txt"
        synset_df = parse_loc_synset_mapping(loc_synset_mapping_txt)
        label_names = synset_df["name"]
        classes = dict(enumerate(label_names))
        return Metadata(classes)


PARQUET_SCHEMA = pa.schema(
    [
        pa.field("image", pa.binary()),
        pa.field("label", pa.int32()),
        pa.field("wnid", pa.string()),
        pa.field("image_id", pa.string()),
        pa.field("orig_index", pa.int32()),
    ]
)
"""Schema for `ImageNetParquet` shards, shared by every writer of this format
(`pack_in1k_to_parquet.py`, `create_in1k_subset.py`) -- this file is the
single source of truth for it."""

PARQUET_COMPRESSION = {
    "image": "none",  # already JPEG-compressed; re-compressing wastes CPU
    "label": "snappy",
    "wnid": "snappy",
    "image_id": "snappy",
    "orig_index": "snappy",
}


class ParquetSample(Sample):
    """`ImageNetParquet` sample.

    :param image_id: The original `ImageNet.paths` entry (e.g.
        `"n01443537/n01443537_10007"` for train, or a bare stem for val) --
        i.e. which source file this row came from.
    :param orig_index: This row's position in `ds.paths` before the pack
        script's pre-shuffle (0-indexed) -- sorting by this column exactly
        reconstructs the original `train_cls.txt`/`val.txt` order.
    """

    image_id: str
    orig_index: int


_NEEDED_COLUMNS = ("image", "label", "image_id", "orig_index")
"""Columns `ImageNetParquet.__getitem__` reads back out -- `wnid` (see
`PARQUET_SCHEMA`) is written but never read, so row-group reads skip it."""


@register_dataset("imagenet-parquet")
class ImageNetParquet(Sequence):
    """Sequence-conforming loader over pre-shuffled ImageNet Parquet shards.

    Produced by `rsrch_data/scripts/pack_in1k_to_parquet.py`. Intended for
    *sequential* access (e.g. `DataLoader(..., shuffle=False)`): the on-disk
    row order is already a random permutation from pack time, so random
    per-epoch reshuffling on top of this would scatter reads across row
    groups and defeat the cache below, without adding any real randomness
    that isn't already there.

    File structure:
    ```
    <data_root>/
    ├── {split}-00000-of-000NN.parquet
    ├── ...
    └── LOC_synset_mapping.txt
    ```
    """

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val"],
        row_group_cache_size: int = 4,
    ) -> None:
        """Index parquet shard footers (no data read) for `split`.

        :param data_root: Directory of Parquet shards, as written by
            `pack_in1k_to_parquet.py`.
        :param split: Which split's shards to load.
        :param row_group_cache_size: Number of decoded row groups to keep
            cached. Sized for sequential access by a handful of concurrent
            reader threads, not for absorbing a fully random access pattern.
        """
        self.root = Path(data_root).expanduser()
        self.split = split
        self.row_group_cache_size = row_group_cache_size

        self.files = sorted(self.root.glob(f"{split}-*.parquet"))
        if not self.files:
            msg = f"No {split}-*.parquet shards found under {self.root}"
            raise FileNotFoundError(msg)

        # Cumulative row offsets per file, and per-row-group row counts within
        # each file -- built entirely from parquet footers, no data read.
        self._offsets = [0]
        self._row_group_rows: list[list[int]] = []
        for file in self.files:
            pf = pq.ParquetFile(file)
            group_rows = [
                pf.metadata.row_group(i).num_rows
                for i in range(pf.metadata.num_row_groups)
            ]
            self._row_group_rows.append(group_rows)
            self._offsets.append(self._offsets[-1] + sum(group_rows))

        self._cache: OrderedDict[tuple[int, int], pa.Table] = OrderedDict()
        self._cache_lock = threading.Lock()

    def __len__(self) -> int:
        """Return total number of samples across all shards."""
        return self._offsets[-1]

    def _locate(self, idx: int) -> tuple[int, int, int]:
        """Map a global row index to (file_idx, row_group_idx, local_offset)."""
        file_idx = bisect.bisect_right(self._offsets, idx) - 1
        offset_in_file = idx - self._offsets[file_idx]
        for row_group_idx, num_rows in enumerate(self._row_group_rows[file_idx]):
            if offset_in_file < num_rows:
                return file_idx, row_group_idx, offset_in_file
            offset_in_file -= num_rows
        msg = f"Index {idx} out of range for {self!r}"
        raise IndexError(msg)

    def _read_row_group(self, file_idx: int, row_group_idx: int) -> pa.Table:
        """Return a row group as a `pyarrow.Table`, caching on a cache miss.

        Reads only the columns `__getitem__` actually uses (`wnid` is never
        read back out), and keeps the result as a columnar `Table` rather
        than eagerly materializing it into per-row Python dicts -- cheaper
        given `__getitem__` only ever needs one row's fields at a time out of
        it, not all of them.

        On a concurrent miss, two threads may redundantly decode the same row
        group; harmless (idempotent, no data race), just wasted work, and
        rare given this is meant for sequential access.
        """
        key = (file_idx, row_group_idx)
        with self._cache_lock:
            table = self._cache.get(key)
            if table is not None:
                self._cache.move_to_end(key)
                return table

        pf = pq.ParquetFile(self.files[file_idx])
        table = pf.read_row_group(row_group_idx, columns=list(_NEEDED_COLUMNS))

        with self._cache_lock:
            self._cache[key] = table
            self._cache.move_to_end(key)
            while len(self._cache) > self.row_group_cache_size:
                self._cache.popitem(last=False)
        return table

    def __getitem__(self, idx: int) -> ParquetSample:
        """Return sample #idx, reading its row group (cached) as needed."""
        if idx < 0:
            idx += len(self)
        file_idx, row_group_idx, local_offset = self._locate(idx)
        table = self._read_row_group(file_idx, row_group_idx)
        return {
            "image": Image.open(
                io.BytesIO(table.column("image")[local_offset].as_py())
            ),
            "label": table.column("label")[local_offset].as_py(),
            "image_id": table.column("image_id")[local_offset].as_py(),
            "orig_index": table.column("orig_index")[local_offset].as_py(),
        }

    def meta(self) -> Metadata:
        """Build image-classification metadata from the synset mapping file."""
        synset_df = parse_loc_synset_mapping(self.root / "LOC_synset_mapping.txt")
        label_names = synset_df["name"]
        return Metadata(dict(enumerate(label_names)))
