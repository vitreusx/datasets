"""Tokenize OpenWebText and write train.bin / val.bin + metadata / index files.

Usage:
    python scripts/preprocess_openwebtext.py --data-root data/openwebtext
"""

from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel
from tokenizers import Tokenizer
from utils.tokenize_text_dataset import tokenize_text_dataset

from rsrch_data.openwebtext import OpenWebTextLoader


class Args(BaseModel):
    """CLI args for the OpenWebText tokenizer."""

    data_root: Path = Path("data/openwebtext")
    """Root directory containing the downloaded dataset."""

    tokenizer: str = "models/gpt2/tokenizer.json"
    """Path to tokenizer.json file."""

    split: Literal["train"] = "train"

    output_dir: Path | None = None
    """Output directory. Defaults to data_root."""

    batch_size: int = 1024
    """Tokenizer batch size (documents per call)."""

    max_shard_size: str = "4G"
    """Maximum size of a single output shard."""


def main(args: Args) -> None:
    """Tokenize all documents and write train.bin / val.bin."""
    out_dir = args.output_dir or args.data_root
    dest = out_dir / f"{args.split}.bin"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    loader = OpenWebTextLoader(
        args.data_root,
        batch_size=args.batch_size,
        split=args.split,
    )

    tokenize_text_dataset(
        loader,
        tokenizer,
        dest,
        max_shard_size=args.max_shard_size,
        tokenizer_id=args.tokenizer,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
