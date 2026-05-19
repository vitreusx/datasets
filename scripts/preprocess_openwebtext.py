"""Tokenize OpenWebText and write train.bin / val.bin + train.idx / val.idx.

*.bin  — flat uint16 array of GPT-2 token IDs (little-endian).
*.idx  — flat uint64 array; entry i is the token offset of document i in *.bin.

Usage:
    python scripts/preprocess_openwebtext.py --data-root data/openwebtext
"""

from collections.abc import Iterator
from itertools import batched
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tiktoken
import tyro
from pydantic import BaseModel
from tqdm import tqdm


class Config(BaseModel):
    """Configuration for the preprocess_openwebtext script."""

    data_root: Path = Path("data/openwebtext")
    """Root directory containing the downloaded dataset."""

    val_frac: float = 0.05
    """Fraction of documents to reserve for validation."""

    output_dir: Path | None = None
    """Directory for output .bin files.  Defaults to data_root if not set."""

    seed: int = 0
    """RNG seed for shuffling documents before the train/val split."""


def _iter_docs(data_root: Path, rng: np.random.Generator) -> Iterator[str]:
    files = sorted((data_root / "plain_text").glob("*.parquet"))
    for fi in rng.permutation(len(files)):
        table = pq.read_table(files[fi], columns=["text"])
        texts: list[str] = table["text"].to_pylist()
        for ri in rng.permutation(len(texts)):
            yield texts[ri]


def main(cfg: Config) -> None:
    """Tokenize all documents and write train.bin / val.bin."""
    out_dir = cfg.output_dir or cfg.data_root
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted((cfg.data_root / "plain_text").glob("*.parquet"))
    n_total = sum(pq.read_metadata(str(f)).num_rows for f in files)
    n_train = int((1.0 - cfg.val_frac) * n_total)
    print(f"Dataset: {n_total:,} docs  |  train {n_train:,}  val {n_total - n_train:,}")

    rng = np.random.default_rng(cfg.seed)
    doc_iter = _iter_docs(cfg.data_root, rng)

    enc = tiktoken.get_encoding("gpt2")

    for split_name, n_docs in [("train", n_train), ("val", n_total - n_train)]:
        out_path = out_dir / f"{split_name}.bin"
        idx_path = out_dir / f"{split_name}.idx"
        texts = (next(doc_iter) for _ in range(n_docs))

        total_tokens = 0
        with (
            out_path.open("wb") as f,
            idx_path.open("wb") as idx_f,
            tqdm(total=n_docs, desc=split_name, unit="doc") as pbar,
        ):
            for text_batch in batched(texts, n=256):
                token_batch = enc.encode_batch(text_batch)
                doc_offsets: list[int] = []
                flat_tokens: list[int] = []
                for doc_tokens in token_batch:
                    doc_offsets.append(total_tokens)
                    flat_tokens.extend(doc_tokens)
                    total_tokens += len(doc_tokens)
                np.array(doc_offsets, dtype=np.uint64).tofile(idx_f)
                np.array(flat_tokens, dtype=np.uint16).tofile(f)
                pbar.update(len(token_batch))
                pbar.set_postfix(tokens=f"{total_tokens / 1e9:.2f}B")

        print(f"  {out_path}  {total_tokens:,} tokens  {total_tokens * 2 / 1e9:.2f} GB")


if __name__ == "__main__":
    main(tyro.cli(Config))
