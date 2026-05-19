"""OpenWebText data loading."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tiktoken


def openwebtext_loader(
    batch_size: int,
    seq_len: int,
    split: Literal["train", "val"] = "train",
    seed: int = 0,
    data_root: str | Path | None = None,
) -> Iterator[np.ndarray]:
    """Yield infinite batches of shape (batch_size, seq_len + 1) dtype int32.

    The +1 lets the training loop slice (input, target) = (batch[:, :-1], batch[:, 1:]).

    If data_root is given, loads from a local repo snapshot (written by
    scripts/get_openwebtext.py). Otherwise streams from HuggingFace.
    Tokenization happens on the fly in both cases.
    """
    from datasets import load_dataset

    if data_root is not None:
        ds = load_dataset(str(data_root), split="train", trust_remote_code=True)
        n_train = int(0.95 * len(ds))
        indices = range(n_train) if split == "train" else range(n_train, len(ds))
        ds = ds.select(indices)
    else:
        hf_split = "train[:95%]" if split == "train" else "train[95%:]"
        ds = load_dataset(
            "Skylion007/openwebtext",
            split=hf_split,
            trust_remote_code=True,
        )

    ds = ds.shuffle(seed=seed)
    yield from _tokenize_and_batch(ds, batch_size, seq_len)


def _tokenize_and_batch(
    ds: Any,
    batch_size: int,
    seq_len: int,
) -> Iterator[np.ndarray]:
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    chunk = seq_len + 1
    buf: list[int] = []

    for doc in _cycle(ds):
        buf.extend(enc.encode_ordinary(doc["text"]))
        buf.append(eot)

        while len(buf) >= batch_size * chunk:
            block = np.array(buf[: batch_size * chunk], dtype=np.int32)
            buf = buf[batch_size * chunk :]
            yield block.reshape(batch_size, chunk)


def _cycle(ds: Any) -> Iterator[Any]:
    while True:
        yield from ds


def openwebtext_memmap_loader(
    batch_size: int,
    seq_len: int,
    split: Literal["train", "val"] = "train",
    seed: int = 0,
    data_root: str | Path = "data/openwebtext",
) -> Iterator[np.ndarray]:
    """Yield infinite batches of shape (batch_size, seq_len + 1) dtype int32.

    Reads from preprocessed *.bin files produced by scripts/preprocess_openwebtext.py.
    Samples random windows via np.memmap — no tokenization overhead at training time.
    """
    data = np.memmap(Path(data_root) / f"{split}.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng(seed)
    chunk = seq_len + 1
    while True:
        starts = rng.integers(0, len(data) - chunk, size=batch_size)
        yield np.stack([data[s : s + chunk].astype(np.int32) for s in starts])
