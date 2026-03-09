from pathlib import Path

import tyro

from rsrch_data.utils.download import download_and_extract


def setup(
    data_root: str,
    source: str = "https://hdvila.blob.core.windows.net/dataset/hdvila100m.zip",
    remove_archives: bool = True,
):
    data_root = Path(data_root)

    download_and_extract(
        url=source,
        dest_dir=data_root,
        archive_dest_dir=None if remove_archives else data_root,
    )


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
