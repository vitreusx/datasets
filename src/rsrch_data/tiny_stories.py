"""TinyStories dataset."""

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, TypedDict

from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """Sample from TinyStories dataset."""

    text: str


@register_dataset("tiny-stories")
class TinyStories(Iterable):
    """TinyStories dataset."""

    def __init__(
        self,
        data_root: str | Path,
        split: Literal["train", "val"] = "train",
    ):
        self.data_root = Path(data_root)
        self.split = split

    def __iter__(self) -> Iterator[Sample]:
        split_name = {"train": "train", "val": "valid"}[self.split]
        file_name = self.data_root / f"TinyStoriesV2-GPT4-{split_name}.txt"
        # TinyStories files are rather freely set documents separated by
        # <|endoftext|> markers.
        left = ""
        with file_name.open() as f:
            while chunk := f.read(1024):
                docs = chunk.split("<|endoftext|>")
                if len(docs) == 1:
                    left = left + docs[0]
                else:
                    yield {"text": (left + docs[0]).strip()}
                    for doc in docs[1:-1]:
                        yield {"text": doc.strip()}
                    left = docs[-1]
        if len(left) > 0:
            yield {"text": left}
