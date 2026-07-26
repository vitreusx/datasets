"""Open Library data loading."""

import gzip
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypedDict

from rsrch_data.registry import register_dataset


class Sample(TypedDict):
    """Open Library record (see `get_open_library.py` for the dump format)."""

    key: str
    revision: int
    last_modified: str
    record: dict
    """The record body, decoded from JSON."""


@register_dataset("open-library")
class OpenLibrary(Iterable[Sample]):
    """Sequential loader over the raw Open Library dump.

    Streams `ol_dump_latest.txt.gz` (as downloaded by `get_open_library.py`)
    line by line.
    """

    def __init__(self, data_root: str | Path):
        """Init OpenLibrary loader."""
        self.data_root = Path(data_root)
        self.dump_path = self.data_root / "ol_dump_latest.txt.gz"

    def __iter__(self) -> Iterator[Sample]:
        with gzip.open(self.dump_path, "rt", encoding="utf-8") as dump:
            for line in dump:
                record_type, key, revision, last_modified, json_str = line.rstrip(
                    "\n"
                ).split("\t", maxsplit=4)
                yield {
                    "record_type": record_type,
                    "key": key,
                    "revision": int(revision),
                    "last_modified": last_modified,
                    "record": json.loads(json_str),
                }
