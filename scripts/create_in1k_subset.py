"""Create a class-subset of a Parquet-packed ImageNet-1k dataset.

Takes a Parquet-packed ImageNet-1k (see pack_in1k_to_parquet.py) and filters
it down to a sample of classes, writing another set of Parquet shards.
Row-group boundaries are rebuilt from scratch (filtering leaves the source's
row groups unevenly sized), and labels are recomputed from `wnid` against the
new (sampled) class list rather than copied -- the source's `label` values are
0-indexed against the *full* class list, not the subset's, and would otherwise
land outside `[0, num_classes)`.
"""

from collections.abc import Iterable
from pathlib import Path

import pyarrow.parquet as pq
import tyro
from pydantic import BaseModel
from tqdm.auto import tqdm

from rsrch_data.imagenet import (
    PARQUET_COMPRESSION,
    PARQUET_SCHEMA,
    parse_loc_synset_mapping,
)
from rsrch_data.utils.misc import parse_size
from rsrch_data.utils.parquet_writer import write_sharded_parquet


class Args(BaseModel):
    """CLI args for the ImageNet-1k Parquet subset creator."""

    in1k_root: str
    """Root of a Parquet-packed full ImageNet-1k (see pack_in1k_to_parquet.py)."""
    output_dir: str
    """Output directory for the subset's Parquet shards."""
    synset_mapping_txt: str | None = None
    """Custom LOC_synset_mapping.txt file (defaults to in1k_root's)."""
    num_classes: int | None = None
    """If provided, sample this many classes from the synset mapping."""
    row_group_size: int = 2000
    """Rows per Parquet row group."""
    max_shard_size: str = "1GiB"
    """Size cap (of raw image bytes) per shard file before rolling over."""
    seed: int = 0
    """Seed for the class sample."""


def _filtered_rows(
    files: list[Path], wnid_to_new_label: dict[str, int], split: str
) -> Iterable[dict]:
    """Yield rows whose wnid survived the class sample, relabeled to it."""
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    with tqdm(total=total, desc=f"Filtering {split}", unit="img") as pbar:
        for file in files:
            pf = pq.ParquetFile(file)
            for batch in pf.iter_batches():
                for row in batch.to_pylist():
                    pbar.update(1)
                    new_label = wnid_to_new_label.get(row["wnid"])
                    if new_label is None:
                        continue
                    row["label"] = new_label
                    yield row


def _subset_split(
    in1k_root: Path,
    output_dir: Path,
    split: str,
    wnid_to_new_label: dict[str, int],
    row_group_size: int,
    max_shard_bytes: int,
) -> None:
    """Filter and relabel one split's shards into `output_dir`."""
    if sorted(output_dir.glob(f"{split}-*.parquet")):
        return
    files = sorted(in1k_root.glob(f"{split}-*.parquet"))
    if not files:
        return

    write_sharded_parquet(
        _filtered_rows(files, wnid_to_new_label, split),
        output_dir,
        split,
        PARQUET_SCHEMA,
        PARQUET_COMPRESSION,
        row_group_size,
        max_shard_bytes,
        size_key="image",
    )


def main(args: Args) -> None:
    """Create a class-subset of a Parquet-packed ImageNet-1k dataset."""
    in1k_root = Path(args.in1k_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synset_mapping_txt = (
        Path(args.synset_mapping_txt)
        if args.synset_mapping_txt is not None
        else in1k_root / "LOC_synset_mapping.txt"
    )
    synset_df = parse_loc_synset_mapping(synset_mapping_txt)
    if args.num_classes is not None:
        num_classes = min(args.num_classes, len(synset_df))
        synset_df = synset_df.sample(num_classes, replace=False, random_state=args.seed)

    wnid_to_new_label = {wnid: label for label, wnid in enumerate(synset_df["wnid"])}
    max_shard_bytes = int(parse_size(args.max_shard_size))

    for split in ("train", "val"):
        _subset_split(
            in1k_root,
            output_dir,
            split,
            wnid_to_new_label,
            args.row_group_size,
            max_shard_bytes,
        )

    with (output_dir / "LOC_synset_mapping.txt").open("w") as f:
        for _, row in synset_df.iterrows():
            f.write(f"{row['wnid']} {row['defs']}\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
