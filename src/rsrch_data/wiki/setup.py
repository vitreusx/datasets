from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import tyro

from rsrch_data.utils.download import download, download_and_extract


def setup(
    data_root: str | Path,
    lang: Literal["en", "fr"],
    version: str = "20250801",
):
    """Setup Wikipedia.

    :param data_root: Output directory in which to save the files.
    :param lang: Language variant of Wikipedia dump to download.
    :param version: Version of the Wikipedia dump to download. Follows
        `YYYYMMDD` format."""

    data_root = Path(data_root)

    prefix = f"https://dumps.wikimedia.org/{lang}wiki/{version}"
    index_url = (
        f"{prefix}/{lang}wiki-{version}-pages-articles-multistream-index.txt.bz2"
    )
    articles_url = f"{prefix}/{lang}wiki-{version}-pages-articles-multistream.xml.bz2"

    dest = data_root / PurePosixPath(urlparse(index_url).path).name
    download_and_extract(index_url, dest)

    dest = data_root / PurePosixPath(urlparse(articles_url).path).name
    download(articles_url, dest)


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
