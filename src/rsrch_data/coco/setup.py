import shutil
import zipfile
from pathlib import Path
from typing import Literal
from urllib.parse import urljoin

import tyro

from rsrch_data.utils.download import download_and_extract

COCOModule = Literal["detection", "panoptic", "stuff"]


def setup(
    data_root: str,
    base_url: str = "http://images.cocodataset.org",
    modules: tuple[COCOModule, ...] = ("detection", "panoptic", "stuff"),
    remove_archives: bool = False,
):
    """Setup COCO dataset(s).

    :param data_root: Output directory in which to place the dataset.
    :param base_url: Source URL from which to fetch the archives.
    :param modules: COCO sub-datasets to fetch.
    :param remove_archives: Whether to remove the downloaded archives after
        extraction."""

    data_root = Path(data_root)
    ann_dir = data_root / "annotations"

    for name in ("train2017", "val2017"):
        source = urljoin(base_url, f"zips/{name}.zip")
        archive_dest = None if remove_archives else data_root
        download_and_extract(
            url=source,
            dest_dir=data_root,
            archive_dest_dir=archive_dest,
        )

    if "detections" in modules:
        suffix = "annotations/annotations_trainval2017.zip"
        source = urljoin(base_url, suffix)
        archive_dest = None if remove_archives else data_root
        download_and_extract(
            url=source,
            dest_dir=data_root,
            archive_dest_dir=archive_dest,
        )

    if "panoptic" in modules:
        suffix = "annotations/panoptic_annotations_trainval2017.zip"
        source = urljoin(base_url, suffix)
        archive_dest = None if remove_archives else data_root
        download_and_extract(
            url=source,
            dest_dir=data_root,
            archive_dest_dir=archive_dest,
        )

        for split in ("train", "val"):
            split_zip = ann_dir / f"panoptic_{split}2017.zip"
            with zipfile.ZipFile(split_zip) as sf:
                sf.extractall(ann_dir)
            split_zip.unlink()

        shutil.rmtree(data_root / "__MACOSX")
        shutil.rmtree(ann_dir / "__MACOSX")

    if "stuff" in modules:
        suffix = "annotations/stuff_annotations_trainval2017.zip"
        source = urljoin(base_url, suffix)
        archive_dest = None if remove_archives else data_root
        download_and_extract(
            url=source,
            dest_dir=data_root,
            archive_dest_dir=archive_dest,
        )

        for split in ("train", "val"):
            split_zip = ann_dir / f"stuff_{split}2017_pixelmaps.zip"
            with zipfile.ZipFile(split_zip) as sf:
                sf.extractall(ann_dir)
            split_zip.unlink()

        shutil.rmtree(ann_dir / "deprecated-challenge2017")


def main():
    tyro.cli(setup)


if __name__ == "__main__":
    main()
