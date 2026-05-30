"""Tokenize any registered text dataset into flat binary token files.

Usage:
    python scripts/tokenize.py --help
    python scripts/tokenize.py wikipedia
        --data-root data/wiki --lang en --version 20231101
"""

import inspect
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import tyro
from tokenizers import Tokenizer
from utils.tokenize_text_dataset import tokenize_text_dataset

from rsrch_data.registry import get_registry


class _BatchedLoader:
    """Wrap a sample-level dataset into batches of text strings."""

    def __init__(self, dataset: object, batch_size: int) -> None:
        self._dataset = dataset
        self._batch_size = batch_size

        if hasattr(self._dataset, "__len__"):

            def __len__(self) -> int:  # noqa: ANN001, N807
                """Return number of batches."""
                return math.ceil(len(self._dataset) / self._batch_size)  # type: ignore[arg-type]

            self.__len__ = __len__

    def __iter__(self) -> Iterator[dict[str, list[str]]]:
        """Yield batches of text strings."""
        batch: list[str] = []
        for sample in self._dataset:
            if "text" not in sample:
                msg = f"Sample missing 'text' key; got: {list(sample)}"
                raise ValueError(msg)
            batch.append(sample["text"])
            if len(batch) >= self._batch_size:
                yield {"text": batch}
                batch = []
        if batch:
            yield {"text": batch}


def main() -> None:
    """Tokenize a registry dataset into flat binary token files."""
    registry = get_registry()

    annotated = [
        Annotated[
            cls,
            tyro.conf.subcommand(
                name,
                description=(inspect.getdoc(cls) or "").partition("\n")[0],
            ),
        ]
        for name, cls in registry.items()
    ]
    union_type = annotated[0]
    for t in annotated[1:]:
        union_type = union_type | t

    @dataclass
    class Args:
        dataset: union_type  # pyright: ignore[reportInvalidTypeForm]
        output_path: str
        tokenizer: str = "models/gpt2/tokenizer.json"
        batch_size: int = 1024
        max_shard_size: str = "4G"

    tyro_conf = (tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes)
    args = tyro.cli(Args, config=tyro_conf)

    dest = Path(args.output_path)

    tokenizer = Tokenizer.from_file(args.tokenizer)
    loader = _BatchedLoader(args.dataset, args.batch_size)

    tokenize_text_dataset(
        loader,
        tokenizer,
        dest,
        max_shard_size=args.max_shard_size,
        tokenizer_id=args.tokenizer,
    )


if __name__ == "__main__":
    main()
