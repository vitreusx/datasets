from pathlib import Path
from urllib.parse import urljoin

import tyro

from rsrch_data.utils.download import download_and_extract


def setup(
    data_root: str,
    base_url: str = "https://vision.im.tum.de/mono",
    remove_archives: bool = True,
):
    data_root = Path(data_root)

    for name in (
        "all_calib_sequences.zip",
        "all_sequences.zip",
    ):
        download_and_extract(
            url=urljoin(base_url, name),
            dest_dir=data_root / name,
            archive_dest_dir=None if remove_archives else data_root,
        )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
