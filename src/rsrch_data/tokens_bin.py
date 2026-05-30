"""Binary token file dataset."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np

from rsrch_data.registry import register_dataset


class Document(TypedDict):
    """A single tokenized document."""

    tokens: np.ndarray
    """A sequence of tokens for the doc, of shape (L,) and dtype uint16."""


class Split(TypedDict):
    """Token range for a single shard."""

    start: int
    end: int


class Metadata(TypedDict):
    """Metadata loaded from the JSON sidecar produced by `tokenize_text_dataset`."""

    num_documents: int
    """Total number of documents for the dataset."""
    num_tokens: int
    """Total number of tokens for the dataset."""
    tokenizer: str | None = None
    """If provided, path to the tokenizer, e.g. `models/gpt2/tokenizer.json`.
    mostly used to remember which tokenizer was used to generate the file."""
    splits: dict[str, Split]
    """If the file has been split, contains the split info in the form of
    `{[split_path: str]: [begin token index, end token index]}`."""


@register_dataset("tokens-bin-docs")
class TokensBinDocs(Sequence):
    """Random-access dataset over a flat uint16 token binary file.

    The file and optional shards are produced by `tokenize_text_dataset`.
    Each item is a single tokenized document returned as a uint16 token array.

    Expected file layout for ``path = "data/train.bin"``::

        data/
          train.bin             # token data (single-file case)
          train.bin.json        # metadata sidecar
          train.index.bin       # per-document start offsets (uint64)

        # or, when sharded:
          train-00000-of-00003.bin
          train-00001-of-00003.bin
          train-00002-of-00003.bin
          train.bin.json
          train.index.bin
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)

        meta_path = path.parent / f"{path.name}.json"
        with meta_path.open() as f:
            self._meta: Metadata = json.load(f)

        index_path = path.parent / f"{path.stem}.index.bin"
        self._offsets = np.fromfile(index_path, dtype=np.uint64)
        self.num_tokens: int = self._meta["num_tokens"]

        splits = self._meta.get("splits", {})
        if not splits:
            self._shards = [np.memmap(path, dtype=np.uint16, mode="r")]
            self._shard_starts = np.array([0], dtype=np.uint64)
        else:
            ordered = sorted(splits.items(), key=lambda kv: kv[1]["start"])
            self._shards = [
                np.memmap(path.parent / name, dtype=np.uint16, mode="r")
                for name, _ in ordered
            ]
            self._shard_starts = np.array(
                [info["start"] for _, info in ordered], dtype=np.uint64
            )

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> Document:
        if index < 0:
            index += len(self)
        start = int(self._offsets[index])
        end = (
            int(self._offsets[index + 1]) if index + 1 < len(self) else self.num_tokens
        )
        shard_idx = int(np.searchsorted(self._shard_starts, start, side="right")) - 1
        base = int(self._shard_starts[shard_idx])
        tokens = np.array(self._shards[shard_idx][start - base : end - base])
        return {"tokens": tokens}

    def read_tokens(self, start: int, end: int) -> np.ndarray:
        """Read a flat token slice [start, end), spanning shards if necessary."""
        shard_idx = int(np.searchsorted(self._shard_starts, start, side="right")) - 1
        base = int(self._shard_starts[shard_idx])
        shard_cap = (
            int(self._shard_starts[shard_idx + 1])
            if shard_idx + 1 < len(self._shards)
            else self.num_tokens
        )
        if end <= shard_cap:
            return np.array(self._shards[shard_idx][start - base : end - base])
        part1 = np.array(self._shards[shard_idx][start - base :])
        next_base = int(self._shard_starts[shard_idx + 1])
        part2 = np.array(self._shards[shard_idx + 1][: end - next_base])
        return np.concatenate([part1, part2])

    def meta(self) -> Metadata:
        """Return the dataset metadata loaded from the JSON sidecar."""
        return self._meta


class Segment(TypedDict):
    """A fixed-size token window from the flat token stream."""

    tokens: np.ndarray
    """A fixed-size token segment, of shape (L,) and dtype uint16."""


def get_num_of_tokens(path: str | Path) -> int:
    """Return the total token count from the JSON sidecar of a binary token file."""
    path = Path(path)
    meta_path = path.parent / f"{path.name}.json"
    with meta_path.open() as f:
        meta: Metadata = json.load(f)
    return meta["num_tokens"]


@register_dataset("tokens-bin-segments")
class TokensBinSegments:
    """Document-agnostic loader yielding fixed-size token windows.

    Iterates over the flat token stream in strides, ignoring document boundaries.
    Supports `len()` for use with tqdm.

    See `TokensBinDocs` for the expected file layout.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        *,
        start: int | None = None,
        end: int | None = None,
        stride: int | None = None,
    ) -> None:
        self._dataset = TokensBinDocs(path)
        self._seq_len = seq_len
        self._start = start if start is not None else 0
        self._end = end if end is not None else self._dataset.num_tokens
        self._stride = stride if stride is not None else seq_len

    def __len__(self) -> int:
        return len(range(self._start, self._end - self._seq_len, self._stride))

    def __getitem__(self, index: int) -> Segment:
        """Return the token window at position index."""
        if index < 0:
            index += len(self)
        offset = self._start + index * self._stride
        return {"tokens": self._dataset.read_tokens(offset, offset + self._seq_len)}

    def __iter__(self) -> Iterator[Segment]:
        """Yield token windows in order."""
        for offset in range(self._start, self._end - self._seq_len, self._stride):
            yield {"tokens": self._dataset.read_tokens(offset, offset + self._seq_len)}

    def meta(self) -> Metadata:
        """Return the dataset metadata."""
        return self._dataset.meta()
