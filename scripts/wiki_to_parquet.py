import argparse
import bz2
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


def batched(xs, batch_size: int, drop_last: bool = False):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if not drop_last and len(batch) > 0:
        yield batch


def parse_size(desc: str):
    value = int(desc[:-1])
    unit = {"k": 1024, "m": 1024**2, "g": 1024**3}[desc[-1].lower()]
    return value * unit


def _task_fn(ms_file: str, offset: int, size: int):
    with open(ms_file, "rb") as ms_f:
        ms_f.seek(offset)
        xml_batch = bz2.decompress(ms_f.read(size))

    xmls = []
    pos = 0
    while True:
        end = xml_batch.find(b"</page>", pos)
        if end < 0:
            break
        xmls.append(xml_batch[pos : end + 7].strip())
        pos = end + 8

    return xmls


def task_fn(arg):
    return _task_fn(*arg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index-file", required=True)
    p.add_argument("--ms-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-size", default="1g")
    args = p.parse_args()

    records = []
    with bz2.open(args.index_file, "r") as f:
        for line in tqdm(f, desc="Parsing index file"):
            line = line.decode()
            pos1 = line.find(":")
            pos2 = line.find(":", pos1 + 1)
            file_offset = int(line[:pos1])
            page_id = int(line[pos1 + 1 : pos2])
            page_title = line[pos2 + 1 : -1]
            records.append((file_offset, page_id, page_title))

    df = pd.DataFrame.from_records(
        records,
        columns=["file_offset", "page_id", "page_title"],
    )

    ms_file_size = Path(args.ms_file).stat().st_size
    file_offsets = df["file_offset"].unique()
    file_ends = np.array([*file_offsets[1:], ms_file_size])
    file_sizes = file_ends - file_offsets

    def get_rows():
        with mp.Pool(os.cpu_count()) as pool:
            task_args = [
                (args.ms_file, offset, size)
                for offset, size in zip(file_offsets, file_sizes, strict=True)
            ]
            for xmls in pool.imap(task_fn, task_args):
                yield from xmls

    schema = pa.schema([("xml", pa.string())])

    def get_batches():
        rows = tqdm(get_rows(), total=len(records), desc="Writing rows")
        for batch in batched(rows, batch_size=1024):
            yield pa.RecordBatch.from_arrays([batch], schema=schema)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    part = 0
    writer = None
    max_split_size = parse_size(args.split_size)

    for batch in get_batches():
        if writer is None:
            dest = output_dir / f"part-{part:05d}.parquet"
            writer = pq.ParquetWriter(dest, schema, compression="zstd")

        writer.write_batch(batch)
        cur_size = dest.stat().st_size
        if cur_size >= max_split_size:
            writer.close()
            writer = None
            part += 1

    num_parts = part + 1 if writer is not None else part
    for part in range(num_parts):
        src = output_dir / f"part-{part:05d}.parquet"
        dest = output_dir / f"part-{part:05d}-of-{num_parts:05d}.parquet"
        src.rename(dest)


if __name__ == "__main__":
    main()
