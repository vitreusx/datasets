"""Split an already-tokenized dataset into train/val by whole document."""

from pathlib import Path

import numpy as np
import tyro
from pydantic import BaseModel
from utils.tokenize_text_dataset import write_token_docs

from rsrch_data.tokens_bin import TokensBinDocs


class Args(BaseModel):
    """CLI args for splitting a tokenized dataset into train/val by document."""

    data_root: str
    source_split: str = "all"
    train_split: str = "train"
    val_split: str = "val"
    val_frac: float = 0.005
    seed: int = 0
    max_shard_size: str = "4G"


def main(args: Args) -> None:
    """Randomly assign whole documents from `source_split` to train/val splits."""
    if (
        args.source_split in (args.train_split, args.val_split)
        or args.train_split == args.val_split
    ):
        msg = (
            "train_split, val_split and source_split must all be distinct -- "
            "writing a split under the same name as the source (or the other "
            "destination split) would truncate/overwrite a .bin file that "
            "TokensBinDocs is still reading from mid-iteration."
        )
        raise ValueError(msg)

    source = TokensBinDocs(args.data_root, split=args.source_split)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(source))

    val_target = args.val_frac * source.num_tokens
    val_indices: list[int] = []
    train_indices: list[int] = []
    val_tokens = 0
    for i in perm:
        if val_tokens < val_target:
            val_indices.append(int(i))
            val_tokens += len(source[int(i)]["tokens"])
        else:
            train_indices.append(int(i))

    # Neither list is sorted back to original document order: each document
    # is read exactly once regardless (train indices are never touched
    # above), so sorting only changes access order, not pass count -- and
    # `train_loader`'s own runtime shuffle already reads the training split
    # in this same random-access pattern (tolerated there since profiling
    # shows this pipeline is compute- not data-bound), so there's nothing to
    # gain by special-casing it here. Leaving both in permutation order also
    # means train.bin itself is now pre-shuffled at rest -- see
    # `Config.train_shuffle` in `gpt/train.py`.
    tokenizer_id = source.meta().get("tokenizer")
    data_root = Path(args.data_root)

    for split_name, indices in (
        (args.train_split, train_indices),
        (args.val_split, val_indices),
    ):
        docs = (source[i]["tokens"] for i in indices)
        write_token_docs(
            docs,
            data_root / f"{split_name}.bin",
            total=len(indices),
            max_shard_size=args.max_shard_size,
            tokenizer_id=tokenizer_id,
        )

    train_tokens = source.num_tokens - val_tokens
    print(f"{args.train_split}: {len(train_indices)} docs, {train_tokens} tokens")
    print(f"{args.val_split}: {len(val_indices)} docs, {val_tokens} tokens")


if __name__ == "__main__":
    main(tyro.cli(Args))
