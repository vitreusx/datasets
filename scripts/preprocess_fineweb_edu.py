"""Tokenize FineWeb-Edu and write token binary files + metadata / index files.

Usage:
    python scripts/preprocess_fineweb_edu.py --data-root data/fineweb-edu
"""

from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel
from tokenizers import Tokenizer
from utils.tokenize_text_dataset import tokenize_text_dataset

from rsrch_data.fineweb import FinewebEduLoader


class Args(BaseModel):
    """CLI args for the FineWeb-Edu tokenizer."""

    data_root: Path = Path("data/fineweb-edu")
    """Root directory containing the downloaded dataset."""

    tokenizer: str = "models/gpt2/tokenizer.json"
    """Path to tokenizer.json file."""

    subset: Literal["sample-10BT"] = "sample-10BT"
    """Dataset subset to tokenize."""

    output_dir: Path | None = None
    """Output directory. Defaults to data_root."""

    batch_size: int = 1024
    """Tokenizer batch size (documents per call)."""

    max_shard_size: str = "4G"
    """Maximum size of a single output shard."""


def main(args: Args) -> None:
    """Tokenize all documents and write binary token files."""
    out_dir = args.output_dir or args.data_root
    dest = out_dir / f"{args.subset}.bin"

    tokenizer = Tokenizer.from_file(args.tokenizer)
    loader = FinewebEduLoader(
        args.data_root,
        subset=args.subset,
        batch_size=args.batch_size,
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
