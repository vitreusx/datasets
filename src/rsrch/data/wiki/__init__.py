from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import tyro
from pydantic import BaseModel

from rsrch.data.utils import download


class Args(BaseModel):
    lang: Literal["en", "fr"]
    version: str = "20250801"
    data_root: str


def main():
    args = tyro.cli(Args)
    data_root = Path(args.data_root)

    index_url = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.version}/{args.lang}wiki-{args.version}-pages-articles-multistream-index.txt.bz2"
    articles_url = f"https://dumps.wikimedia.org/{args.lang}wiki/{args.version}/{args.lang}wiki-{args.version}-pages-articles-multistream.xml.bz2"

    for url in (index_url, articles_url):
        dest = data_root / PurePosixPath(urlparse(url).path).name
        download(url, dest)


if __name__ == "__main__":
    main()
