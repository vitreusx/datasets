"""Utility for tokenizing a text dataset into a flat binary token file."""

import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from tokenizers import Tokenizer
from tqdm.auto import tqdm

from rsrch_data.utils.misc import parse_size

if TYPE_CHECKING:
    from rsrch_data.tokens_bin import Metadata


class Batch(TypedDict):
    """Input batch of text documents."""

    text: list[str]


class Split(TypedDict):
    """Token range for a single shard."""

    start: int
    end: int


def tokenize_text_dataset(  # noqa: C901, PLR0915
    loader: Iterable[Batch],
    tokenizer: Tokenizer,
    dest: str | Path,
    *,
    progress: bool = True,
    max_shard_size: str = "4G",
    tokenizer_id: str | None = None,
) -> None:
    """Tokenize text dataset into a flat sequence of `np.uint16` values.

    The sequence is put into `dest`. If the total size exceeds `max_shared_size`,
    it's split in the following fashion:

    - `{dest.stem}-000xx-of-000yy.{dest.suffix}` for the splits.

    Additionally, following files are created:
    - `{dest}.json` containing the metadata - see `Metadata` for more details.
    - `{dest.stem}.index.bin`, containing the start indices (global offsets) for
      each document
    """
    dest = Path(dest)
    max_bytes = parse_size(max_shard_size)
    max_tokens_per_shard = max_bytes // 2  # uint16 = 2 bytes each

    dest.parent.mkdir(parents=True, exist_ok=True)

    doc_offsets: list[int] = []
    total_tokens = 0
    num_documents = 0
    shard_starts: list[int] = [0]
    shard_tmp_paths: list[Path] = []

    def _open_shard() -> tuple[object, Path]:
        fd, path_str = tempfile.mkstemp(suffix=dest.suffix, dir=dest.parent)
        path = Path(path_str)
        shard_tmp_paths.append(path)
        return open(fd, "wb"), path  # noqa: PTH123

    shard_file, _ = _open_shard()
    current_shard_tokens = 0

    try:
        total = len(loader) if hasattr(loader, "__len__") else None
        pbar = tqdm(loader, unit="batch", total=total, disable=not progress)
        for batch in pbar:
            for enc in tokenizer.encode_batch(batch["text"]):
                ids = np.array(enc.ids, dtype=np.uint16)

                if current_shard_tokens >= max_tokens_per_shard:
                    shard_file.close()
                    shard_starts.append(total_tokens)
                    shard_file, _ = _open_shard()
                    current_shard_tokens = 0

                doc_offsets.append(total_tokens)
                ids.tofile(shard_file)
                total_tokens += len(ids)
                current_shard_tokens += len(ids)
                num_documents += 1
            pbar.set_postfix(docs=num_documents, tok=f"{total_tokens / 1e9:.2f}B")

        shard_file.close()
        shard_file = None

    except Exception:
        if shard_file is not None:
            shard_file.close()
        for p in shard_tmp_paths:
            p.unlink(missing_ok=True)
        raise

    num_shards = len(shard_tmp_paths)
    splits: dict[str, Split] = {}

    if num_shards == 1:
        shutil.move(shard_tmp_paths[0], dest)
    else:
        shard_starts.append(total_tokens)
        for i, tmp_path in enumerate(shard_tmp_paths):
            shard_name = f"{dest.stem}-{i:05d}-of-{num_shards:05d}{dest.suffix}"
            shard_path = dest.parent / shard_name
            shutil.move(tmp_path, shard_path)
            splits[shard_path.name] = {
                "start": shard_starts[i],
                "end": shard_starts[i + 1],
            }

    metadata: Metadata = {
        "num_documents": num_documents,
        "num_tokens": total_tokens,
        "splits": splits,
    }
    if tokenizer_id is not None:
        metadata["tokenizer"] = tokenizer_id
    (dest.parent / f"{dest.name}.json").write_text(json.dumps(metadata, indent=2))

    np.array(doc_offsets, dtype=np.uint64).tofile(
        dest.parent / f"{dest.stem}.index.bin"
    )
