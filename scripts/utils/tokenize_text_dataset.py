"""Utility for tokenizing a text dataset into a flat binary token file."""

import json
import shutil
import tempfile
from collections.abc import Iterable, Iterator
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


class _TokenWriter:
    """Stateful writer for a flat uint16 token file, with shard rotation.

    Shards are written to temp files and only moved into place (and the
    metadata/index sidecars written) on `close()`, so a crash mid-write
    never leaves a corrupt `dest` behind.
    """

    def __init__(
        self,
        dest: str | Path,
        *,
        max_shard_size: str,
        tokenizer_id: str | None,
    ) -> None:
        self._dest = Path(dest)
        self._max_tokens_per_shard = parse_size(max_shard_size) // 2  # uint16 = 2 bytes
        self._tokenizer_id = tokenizer_id
        self._dest.parent.mkdir(parents=True, exist_ok=True)

        self._doc_offsets: list[int] = []
        self.total_tokens = 0
        self.num_documents = 0
        self._shard_starts: list[int] = [0]
        self._shard_tmp_paths: list[Path] = []
        self._current_shard_tokens = 0
        self._shard_file, _ = self._open_shard()

    def _open_shard(self) -> tuple[object, Path]:
        fd, path_str = tempfile.mkstemp(suffix=self._dest.suffix, dir=self._dest.parent)
        path = Path(path_str)
        self._shard_tmp_paths.append(path)
        return open(fd, "wb"), path  # noqa: PTH123

    def write(self, ids: np.ndarray) -> None:
        """Append a single tokenized document."""
        if self._current_shard_tokens >= self._max_tokens_per_shard:
            self._shard_file.close()
            self._shard_starts.append(self.total_tokens)
            self._shard_file, _ = self._open_shard()
            self._current_shard_tokens = 0

        self._doc_offsets.append(self.total_tokens)
        ids.tofile(self._shard_file)
        self.total_tokens += len(ids)
        self._current_shard_tokens += len(ids)
        self.num_documents += 1

    def abort(self) -> None:
        """Discard temp shards after a failed write."""
        self._shard_file.close()
        for p in self._shard_tmp_paths:
            p.unlink(missing_ok=True)

    def close(self) -> None:
        """Finalize: move shards into place, write metadata + index sidecars."""
        self._shard_file.close()

        num_shards = len(self._shard_tmp_paths)
        splits: dict[str, Split] = {}

        if num_shards == 1:
            shutil.move(self._shard_tmp_paths[0], self._dest)
        else:
            self._shard_starts.append(self.total_tokens)
            for i, tmp_path in enumerate(self._shard_tmp_paths):
                shard_name = (
                    f"{self._dest.stem}-{i:05d}-of-{num_shards:05d}{self._dest.suffix}"
                )
                shard_path = self._dest.parent / shard_name
                shutil.move(tmp_path, shard_path)
                splits[shard_path.name] = {
                    "start": self._shard_starts[i],
                    "end": self._shard_starts[i + 1],
                }

        metadata: Metadata = {
            "num_documents": self.num_documents,
            "num_tokens": self.total_tokens,
            "splits": splits,
        }
        if self._tokenizer_id is not None:
            metadata["tokenizer"] = self._tokenizer_id
        (self._dest.parent / f"{self._dest.name}.json").write_text(
            json.dumps(metadata, indent=2)
        )

        np.array(self._doc_offsets, dtype=np.uint64).tofile(
            self._dest.parent / f"{self._dest.stem}.index.bin"
        )


def write_token_docs(
    docs: Iterable[np.ndarray],
    dest: str | Path,
    *,
    total: int | None = None,
    progress: bool = True,
    max_shard_size: str = "4G",
    tokenizer_id: str | None = None,
) -> None:
    """Write pre-tokenized uint16 documents with shard rotation + metadata + index.

    Same on-disk layout as `tokenize_text_dataset` (see `Metadata` in
    `rsrch_data/tokens_bin.py`): `{dest}` (or `{dest.stem}-NNNNN-of-MMMMM{dest.suffix}`
    shards if `max_shard_size` is exceeded), `{dest.name}.json`,
    `{dest.stem}.index.bin`.
    """
    writer = _TokenWriter(
        dest, max_shard_size=max_shard_size, tokenizer_id=tokenizer_id
    )
    try:
        pbar = tqdm(docs, unit="doc", total=total, disable=not progress)
        for ids in pbar:
            writer.write(ids)
            pbar.set_postfix(
                docs=writer.num_documents, tok=f"{writer.total_tokens / 1e9:.2f}B"
            )
    except Exception:
        writer.abort()
        raise
    writer.close()


def _iter_tokenized_docs(
    loader: Iterable[Batch],
    tokenizer: Tokenizer,
    *,
    total: int | None,
    progress: bool,
) -> Iterator[np.ndarray]:
    pbar = tqdm(loader, unit="batch", total=total, disable=not progress)
    num_documents = 0
    total_tokens = 0
    for batch in pbar:
        for enc in tokenizer.encode_batch(batch["text"]):
            ids = np.array(enc.ids, dtype=np.uint16)
            num_documents += 1
            total_tokens += len(ids)
            yield ids
        pbar.set_postfix(docs=num_documents, tok=f"{total_tokens / 1e9:.2f}B")


def _loader_len(loader: Iterable[Batch]) -> int | None:
    # `hasattr(loader, "__len__")` isn't reliable here -- a wrapper like
    # `_BatchedLoader` can define `__len__` unconditionally and have it
    # raise `TypeError` itself when the underlying dataset has none, so
    # catching that directly is the robust check.
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return None


def tokenize_text_dataset(
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
    total = _loader_len(loader)

    # Batch-level progress bar (with docs=/tok= postfix) lives here, so
    # `write_token_docs` sees a flat, un-batched generator and runs with its
    # own progress bar disabled.
    write_token_docs(
        _iter_tokenized_docs(loader, tokenizer, total=total, progress=progress),
        dest,
        progress=False,
        max_shard_size=max_shard_size,
        tokenizer_id=tokenizer_id,
    )
