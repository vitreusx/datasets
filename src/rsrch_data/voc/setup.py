from pathlib import Path

import tyro

from rsrch_data.utils.download import download_and_extract

URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"


def setup(
    data_root: str | Path,
    source: str = URL,
    remove_archives: bool = True,
):
    """Setup VOC2012.

    :param data_root: Output directory in which to place the dataset.
    :param source: An URL for the `VOCtrainval_11-May-2012.tar` file.
    :param remove_archives: Whether to remove archive(s) after extraction."""

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
