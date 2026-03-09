import bz2
import io
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path


class Wiki:
    def __init__(self, data_root: str | Path, lang: str, version: str):
        self.data_root = Path(data_root)
        self.lang = lang
        self.version = version

        index_path = f"{lang}wiki-{version}-pages-articles-multistream-index.txt"
        self.index_path = self.data_root / index_path

        self._index = None

        xml_path = f"{lang}wiki-{version}-pages-articles-multistream.xml.bz2"
        self.xml_path = self.data_root / xml_path

        self._decomp = bz2.BZ2Decompressor()

    def __len__(self):
        return len(self.index)

    @property
    def index(self):
        if self._index is None:
            with open(self.index_path, "r") as f:
                self._index = [*f]
        return self._index

    def __getitem__(self, idx: int):
        line = self.index[idx]
        offset, id_, title = line.split(":", maxsplit=2)
        offset = int(offset)
        title = title.rstrip()

        with open(self.xml_path, "rb") as f:
            f.seek(offset)
            data = io.BytesIO()
            data.write(b"<root>")
            while True:
                block = f.read(262144)
                try:
                    data.write(self._decomp.decompress(block))
                except EOFError:
                    break
            data.write(b"</root>")
            data.seek(0)
            xml = ET.parse(data)  # noqa: S314

        return xml.find(f".//page[id = '{id_}']")

    def __iter__(self):
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

        with open(self.xml_path, "rb") as f:
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
