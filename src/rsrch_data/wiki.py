"""Wikipedia dump dataset loader."""

import bz2
import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from itertools import pairwise
from pathlib import Path
from typing import TypedDict

from rsrch_data.registry import register_dataset


@register_dataset("wiki-xml")
class WikiXml(Sequence):
    """Wikipedia multistream dump dataset (Raw XML)."""

    def __init__(self, data_root: str | Path, lang: str, version: str):
        self.data_root = Path(data_root)
        self.lang = lang
        self.version = version

        index_path = f"{lang}wiki-{version}-pages-articles-multistream-index.txt"
        self.index_path = self.data_root / index_path

        self._index = None

        xml_path = f"{lang}wiki-{version}-pages-articles-multistream.xml.bz2"
        self.xml_path = self.data_root / xml_path

    def __len__(self):
        return len(self.index)

    @property
    def index(self) -> list[str]:
        """Lazily loaded list of index lines."""
        if self._index is None:
            with self.index_path.open() as f:
                self._index = [*f]
        return self._index

    def __getitem__(self, idx: int) -> ET.Element | None:
        line = self.index[idx]
        offset, id_, title = line.split(":", maxsplit=2)
        offset = int(offset)
        title = title.rstrip()

        with self.xml_path.open("rb") as f:
            decomp = bz2.BZ2Decompressor()
            f.seek(offset)
            data = io.BytesIO()
            data.write(b"<root>")
            while True:
                block = f.read(4096)
                try:
                    data.write(decomp.decompress(block))
                except EOFError:
                    break
            data.write(b"</root>")
            data.seek(0)
            xml = ET.parse(data)  # noqa: S314

        return xml.find(f".//page[id = '{id_}']")

    def __iter__(self) -> Iterator[ET.Element]:
        index = []
        for line in self.index:
            offset, _, title = line.split(":", maxsplit=2)
            offset = int(offset)
            title = title.rstrip()
            index.append((offset, title))

        offsets = []
        for offset, _ in index:
            if len(offsets) == 0 or offset != offsets[-1]:
                offsets.append(offset)

        with self.xml_path.open("rb") as f:
            for begin, end in pairwise(offsets):
                f.seek(begin)
                data = io.BytesIO()
                data.write(b"<root>")
                block = f.read(end - begin)
                data.write(bz2.decompress(block))
                data.write(b"</root>")
                data.seek(0)
                xml = ET.parse(data)  # noqa: S314
                yield from xml.getroot()


class TextSample(TypedDict):
    """Text sample from Wikipedia dump."""

    text: str


@register_dataset("wiki-text")
class WikiText(Sequence[TextSample]):
    """Wikipedia multistream dump dataset (Text with templates)."""

    def __init__(
        self,
        data_root: str | Path,
        lang: str,
        version: str,
        remove_links: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.lang = lang
        self.version = version
        self.remove_links = remove_links
        self._xml = WikiXml(data_root, lang=lang, version=version)

    def __len__(self):
        return len(self._xml)

    def __getitem__(self, index: int) -> TextSample:
        xml = self._xml[index]
        text = xml.find("revision/text").text
        if self.remove_links:
            text = self._remove_links(text)
        return {"text": text}

    def __iter__(self) -> Iterator[TextSample]:
        for xml in self._xml:
            text = xml.find("revision/text").text
            if self.remove_links:
                text = self._remove_links(text)
            yield {"text": text}

    def _remove_links(self, text: str) -> str:
        # For piped links [[Target|Display]], keep only the display text
        text = re.sub(r"\[\[[^\[\]]*\|([^\[\]]*)\]\]", r"\1", text)
        # Remove remaining links without pipes
        text = re.sub(r"\[\[[^\[\]]*\]\]", "", text)
        return text
