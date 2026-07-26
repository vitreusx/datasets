"""Generic helper for writing rows to size-capped, row-grouped Parquet shards."""

import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_sharded_parquet(  # noqa: C901
    rows: Iterable[dict],
    output_dir: Path,
    prefix: str,
    schema: pa.Schema,
    compression: dict[str, str],
    row_group_size: int,
    max_shard_bytes: int,
) -> None:
    """Write `rows` to `{prefix}-NNNNN-of-MMMMM.parquet` shards under `output_dir`.

    Shards roll over once a shard file's actual on-disk size reaches
    `max_shard_bytes`, checked directly against the output stream after every
    row group -- not estimated from some column's raw (pre-compression)
    byte length, since compression ratio varies wildly by column (an
    already-compressed JPEG column is ~1:1, but zstd-compressed text/JSON can
    shrink several-fold), which would otherwise make the cap inaccurate.
    Each shard is internally chunked into `row_group_size` row groups. Shards
    are written to tempfiles and renamed to their final `N-of-M` name only
    once the total shard count is known, so a failure mid-write leaves no
    partial/misnamed output.

    :param rows: Row dicts matching `schema`'s field names.
    :param output_dir: Directory to write shards into (must already exist).
    :param prefix: Shard filename prefix (e.g. `"train"`/`"val"`).
    :param schema: Target Parquet schema.
    :param compression: Per-column compression, passed to `pq.ParquetWriter`.
    :param row_group_size: Rows per Parquet row group.
    :param max_shard_bytes: Size cap (actual on-disk bytes) per shard file
        before rolling over to a new one. A soft cap, not a hard one: it's
        only checked after a full row group has been written, so a shard can
        end up somewhat larger than this -- by up to one row group's
        on-disk size, which is usually small relative to the cap, but grows
        with `row_group_size` and per-row size.
    """
    shard_tmp_paths: list[Path] = []
    buffer: list[dict] = []
    file_obj = None
    writer = None

    def _open_shard() -> None:
        nonlocal file_obj, writer
        fd, tmp_path_str = tempfile.mkstemp(suffix=".parquet", dir=output_dir)
        os.close(fd)
        tmp_path = Path(tmp_path_str)
        shard_tmp_paths.append(tmp_path)
        file_obj = tmp_path.open("wb")
        writer = pq.ParquetWriter(file_obj, schema, compression=compression)

    def _close_shard() -> None:
        nonlocal file_obj, writer
        writer.close()
        file_obj.close()
        writer = None
        file_obj = None

    def _flush_row_group(*, force: bool = False) -> None:
        nonlocal buffer
        if not buffer or (len(buffer) < row_group_size and not force):
            return
        if writer is None:
            _open_shard()
        table = pa.Table.from_pylist(buffer, schema=schema)
        writer.write_table(table, row_group_size=row_group_size)
        buffer = []
        if file_obj.tell() >= max_shard_bytes:
            _close_shard()

    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= row_group_size:
                _flush_row_group()
        _flush_row_group(force=True)
        if writer is not None:
            _close_shard()
    except Exception:
        if writer is not None:
            _close_shard()
        for p in shard_tmp_paths:
            p.unlink(missing_ok=True)
        raise

    num_shards = len(shard_tmp_paths)
    for i, tmp_path in enumerate(shard_tmp_paths):
        dest = output_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
        shutil.move(str(tmp_path), str(dest))
