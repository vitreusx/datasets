"""Pack an ImageNet-format dataset into pre-shuffled Parquet shards.

Reads raw JPEG bytes directly (no decode/re-encode) from an ImageNet-format
root (as produced by e.g. create_in1k_subset.py / get_imagenette.py), shuffles
sample order once with a fixed seed, and writes `{split}-NNNNN-of-MMMMM.parquet`
shards -- see `ImageNetParquet` in rsrch_data/imagenet.py for the loader.

The one-time shuffle means row groups are already a random mix of classes, so
`ImageNetParquet` is meant to be read back sequentially (no per-epoch
reshuffling needed).
"""

import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from pydantic import BaseModel
from tqdm.auto import tqdm

from rsrch_data.imagenet import PARQUET_COMPRESSION, PARQUET_SCHEMA, ImageNet
from rsrch_data.utils.misc import parse_size
from rsrch_data.utils.parquet_writer import write_sharded_parquet


class Args(BaseModel):
    """CLI args for the ImageNet-to-Parquet packer."""

    in1k_root: str
    """Source ImageNet-format root (e.g. data/imagenet-100)."""
    output_dir: str
    """Output directory for the packed Parquet shards."""
    row_group_size: int = 2000
    """Rows per Parquet row group."""
    max_shard_size: str = "2GiB"
    """Size cap (of raw image bytes) per shard file before rolling over."""
    seed: int = 0
    """Seed for the one-time pre-shuffle."""


def _rows(ds: ImageNet, order: np.ndarray, split: str) -> Iterable[dict]:
    for i in tqdm(order, desc=f"Packing {split}", unit="img"):
        path = ds.paths[i]
        wnid = ds.wnids[i]
        label = ds.wnid_to_label[wnid]
        img_bytes = (ds.img_root / (path + ".JPEG")).read_bytes()
        yield {
            "image": img_bytes,
            "label": label,
            "wnid": wnid,
            "image_id": path,
            # Position in `ds.paths` (== train_cls.txt/val.txt's row order,
            # 0-indexed) -- lets the original file order be exactly
            # reconstructed later via sort_by("orig_index").
            "orig_index": int(i),
        }


def _pack_split(
    in1k_root: Path,
    output_dir: Path,
    split: Literal["train", "val"],
    row_group_size: int,
    max_shard_bytes: int,
    seed: int,
) -> None:
    """Shuffle and write one split's samples as Parquet shards."""
    ds = ImageNet(in1k_root, split=split)
    if len(ds) == 0:
        return

    order = np.random.default_rng(seed).permutation(len(ds))

    write_sharded_parquet(
        _rows(ds, order, split),
        output_dir,
        split,
        PARQUET_SCHEMA,
        PARQUET_COMPRESSION,
        row_group_size,
        max_shard_bytes,
        size_key="image",
    )


def main(args: Args) -> None:
    """Pack an ImageNet-format dataset into pre-shuffled Parquet shards."""
    in1k_root = Path(args.in1k_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_shard_bytes = int(parse_size(args.max_shard_size))

    for split in ("train", "val"):
        _pack_split(
            in1k_root,
            output_dir,
            split,
            args.row_group_size,
            max_shard_bytes,
            args.seed,
        )

    shutil.copy(
        in1k_root / "LOC_synset_mapping.txt", output_dir / "LOC_synset_mapping.txt"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
