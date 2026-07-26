"""Resize a Parquet-packed ImageNet dataset's images to a smaller resolution.

Takes a Parquet-packed ImageNet dataset (see pack_in1k_to_parquet.py) and
re-encodes each image so its smaller dimension is at most `smallest_size`,
preserving aspect ratio. Source images are typically several times larger
than the training crop actually needs (e.g. ImageNet-100's median is ~375px
on the smaller side against a 224px crop), so most of that resolution is
decoded at training time only to be immediately thrown away by the resize --
shrinking it here once cuts both on-disk size and per-epoch decode cost.
Images already at or below `smallest_size` are left unresized (upscaling
would add no information).

Row-group boundaries and row order are preserved as-is from the source
(already pre-shuffled by pack_in1k_to_parquet.py); only the `image` column
changes.
"""

import io
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
import tyro
from PIL import Image
from pydantic import BaseModel
from tqdm.auto import tqdm

from rsrch_data.imagenet import PARQUET_COMPRESSION, PARQUET_SCHEMA
from rsrch_data.utils.misc import parse_size
from rsrch_data.utils.parquet_writer import write_sharded_parquet


class Args(BaseModel):
    """CLI args for the Parquet-packed ImageNet resizer."""

    in1k_root: str
    """Root of a Parquet-packed ImageNet dataset (see pack_in1k_to_parquet.py)."""
    output_dir: str
    """Output directory for the resized Parquet shards."""
    smallest_size: int = 224
    """Target size for each image's smaller dimension."""
    jpeg_quality: int = 90
    row_group_size: int = 2000
    """Rows per Parquet row group."""
    max_shard_size: str = "2GiB"
    """Size cap (actual on-disk bytes) per shard file before rolling over --
    a soft cap: shards can end up slightly larger, by up to one row group's
    worth of data (see write_sharded_parquet)."""
    num_threads: int = 8
    """Decode/resize/encode worker threads -- see rsrch.utils.data.Pipeline's
    docstring for why this shouldn't just be "every physical core": PIL's
    JPEG codec releases the GIL, but past a handful of threads, contention
    outweighs the added parallelism for this kind of work."""


def _resize_image(img_bytes: bytes, smallest_size: int, quality: int) -> bytes:
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    min_dim = min(image.width, image.height)
    if min_dim > smallest_size:
        if image.width > image.height:
            new_w = round(image.width / image.height * smallest_size)
            new_h = smallest_size
        else:
            new_h = round(image.height / image.width * smallest_size)
            new_w = smallest_size
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=quality, subsampling=0)
    return out.getvalue()


def _resized_rows(
    files: list[Path],
    smallest_size: int,
    quality: int,
    num_threads: int,
    split: str,
) -> Iterable[dict]:
    """Yield rows with `image` re-encoded at (at most) `smallest_size`."""
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    with (
        ThreadPoolExecutor(num_threads) as pool,
        tqdm(total=total, desc=f"Resizing {split}", unit="img") as pbar,
    ):
        for file in files:
            pf = pq.ParquetFile(file)
            for batch in pf.iter_batches():
                rows = batch.to_pylist()
                resized = pool.map(
                    lambda r: _resize_image(r["image"], smallest_size, quality),
                    rows,
                )
                for row, new_image in zip(rows, resized, strict=True):
                    row["image"] = new_image
                    pbar.update(1)
                    yield row


def _resize_split(
    in1k_root: Path,
    output_dir: Path,
    split: str,
    smallest_size: int,
    quality: int,
    row_group_size: int,
    max_shard_bytes: int,
    num_threads: int,
) -> None:
    """Resize and re-write one split's shards into `output_dir`."""
    if sorted(output_dir.glob(f"{split}-*.parquet")):
        return
    files = sorted(in1k_root.glob(f"{split}-*.parquet"))
    if not files:
        return

    write_sharded_parquet(
        _resized_rows(files, smallest_size, quality, num_threads, split),
        output_dir,
        split,
        PARQUET_SCHEMA,
        PARQUET_COMPRESSION,
        row_group_size,
        max_shard_bytes,
    )


def main(args: Args) -> None:
    """Resize a Parquet-packed ImageNet dataset's images to `smallest_size`."""
    in1k_root = Path(args.in1k_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_shard_bytes = int(parse_size(args.max_shard_size))

    for split in ("train", "val"):
        _resize_split(
            in1k_root,
            output_dir,
            split,
            args.smallest_size,
            args.jpeg_quality,
            args.row_group_size,
            max_shard_bytes,
            args.num_threads,
        )

    synset_src = in1k_root / "LOC_synset_mapping.txt"
    if synset_src.exists():
        shutil.copy(synset_src, output_dir / "LOC_synset_mapping.txt")


if __name__ == "__main__":
    main(tyro.cli(Args))
