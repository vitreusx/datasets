"""Binary token file dataset."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict

import numpy as np


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
    num_tokens: int
    tokenizer: str | None
    splits: dict[str, Split]


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
    tokens: np.ndarray
    """A fixed-size token segment, of shape (L,) and dtype uint16."""


class TokensBinLoader:
    """Document-agnostic loader yielding fixed-size token windows.

    Iterates over the flat token stream in strides, ignoring document boundaries.
    Supports `len()` for use with tqdm.

    See `TokensBinDocs` for the expected file layout.
    """

    def __init__(
        self,
        path: str | Path,
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        self._dataset = TokensBinDocs(path)
        self._seq_len = seq_len
        self._stride = stride if stride is not None else seq_len

    def __len__(self) -> int:
        n = self._dataset.num_tokens
        return max(0, (n - self._seq_len) // self._stride + 1)

    def __iter__(self) -> Iterator[Segment]:
        """Yield token windows in order."""
        for i in range(len(self)):
            start = i * self._stride
            yield {"tokens": self._dataset.read_tokens(start, start + self._seq_len)}

    def meta(self) -> Metadata:
        """Return the dataset metadata."""
        return self._dataset.meta()
