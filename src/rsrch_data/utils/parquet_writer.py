"""Generic helper for writing rows to size-capped, row-grouped Parquet shards."""

import os
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_sharded_parquet(
    rows: Iterable[dict],
    output_dir: Path,
    prefix: str,
    schema: pa.Schema,
    compression: dict[str, str],
    row_group_size: int,
    max_shard_bytes: int,
    size_key: str,
) -> None:
    """Write `rows` to `{prefix}-NNNNN-of-MMMMM.parquet` shards under `output_dir`.

    Shards roll over once accumulated `size_key` column bytes reach
    `max_shard_bytes`; each shard is internally chunked into `row_group_size`
    row groups. Shards are written to tempfiles and renamed to their final
    `N-of-M` name only once the total shard count is known, so a failure
    mid-write leaves no partial/misnamed output.

    :param rows: Row dicts matching `schema`'s field names.
    :param output_dir: Directory to write shards into (must already exist).
    :param prefix: Shard filename prefix (e.g. `"train"`/`"val"`).
    :param schema: Target Parquet schema.
    :param compression: Per-column compression, passed to `pq.write_table`.
    :param row_group_size: Rows per Parquet row group.
    :param max_shard_bytes: Size cap (of `size_key` column bytes) per shard
        file before rolling over to a new one.
    :param size_key: Which column's byte length counts toward `max_shard_bytes`.
    """
    shard_tmp_paths: list[Path] = []
    buffer: list[dict] = []
    buffer_bytes = 0

    def _flush_shard() -> None:
        nonlocal buffer, buffer_bytes
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=schema)
        fd, tmp_path_str = tempfile.mkstemp(suffix=".parquet", dir=output_dir)
        os.close(fd)
        tmp_path = Path(tmp_path_str)
        pq.write_table(
            table, tmp_path, row_group_size=row_group_size, compression=compression
        )
        shard_tmp_paths.append(tmp_path)
        buffer = []
        buffer_bytes = 0

    try:
        for row in rows:
            buffer.append(row)
            buffer_bytes += len(row[size_key])
            if buffer_bytes >= max_shard_bytes:
                _flush_shard()
        _flush_shard()
    except Exception:
        for p in shard_tmp_paths:
            p.unlink(missing_ok=True)
        raise

    num_shards = len(shard_tmp_paths)
    for i, tmp_path in enumerate(shard_tmp_paths):
        dest = output_dir / f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet"
        shutil.move(str(tmp_path), str(dest))
