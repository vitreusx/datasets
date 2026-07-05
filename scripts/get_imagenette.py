"""Download Imagenette and repackage it into the ImageNet/ILSVRC layout.

Imagenette (https://github.com/fastai/imagenette) is a real 10-class subset
of ImageNet, small and fast enough to train on for pipeline sanity checks.
Rather than adding a separate dataset class, this script re-lays the
downloaded archive out as a miniature ILSVRC root (`ILSVRC/Data/CLS-LOC/...`,
`LOC_synset_mapping.txt`, `ImageSets/CLS-LOC/{train_cls,val}.txt`,
`LOC_val_solution.csv`), so it can be loaded with the existing
`rsrch_data.imagenet.ImageNet` class unmodified.
"""

import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Literal

import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download, get_url_filename

_SOURCES = {
    "full": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    320: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    160: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
}

_EXTRACTED_DIRS = {
    "full": "imagenette2",
    320: "imagenette2-320",
    160: "imagenette2-160",
}

# wnid -> class name, ordered to match the label indices assigned below.
_WNID_TO_NAME = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


class Args(BaseModel):
    """CLI arguments for the Imagenette downloader."""

    data_root: str
    """Output directory; will hold an ImageNet-layout dataset root."""
    size: Literal["full", 320, 160] = 160
    """Which resized variant to download; 160 is smallest/fastest to train on."""


def _write_synset_mapping(data_root: Path) -> None:
    lines = [f"{wnid} {name}\n" for wnid, name in _WNID_TO_NAME.items()]
    (data_root / "LOC_synset_mapping.txt").write_text("".join(lines))


def _relayout_train(extracted_dir: Path, data_root: Path) -> list[str]:
    """Move train's per-class folders under ILSVRC/Data/CLS-LOC/train.

    :return: `wnid/stem` paths, as real ImageNet's `train_cls.txt` expects.
    """
    src_dir = extracted_dir / "train"
    dest_dir = data_root / "ILSVRC/Data/CLS-LOC/train"
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for wnid in _WNID_TO_NAME:
        dest_class_dir = dest_dir / wnid
        shutil.move(str(src_dir / wnid), str(dest_class_dir))
        paths.extend(
            f"{wnid}/{img_path.stem}"
            for img_path in sorted(dest_class_dir.glob("*.JPEG"))
        )
    return paths


def _relayout_val(extracted_dir: Path, data_root: Path) -> list[tuple[str, str]]:
    """Flatten val's per-class folders under ILSVRC/Data/CLS-LOC/val.

    Real ImageNet's val split has no per-class subfolders -- the wnid for
    each image is only recoverable via `LOC_val_solution.csv` -- unlike
    Imagenette's val, which (like its train) is organized by class folder.

    :return: `(stem, wnid)` pairs, in the same order images are flattened.
    """
    src_dir = extracted_dir / "val"
    dest_dir = data_root / "ILSVRC/Data/CLS-LOC/val"
    dest_dir.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []
    for wnid in _WNID_TO_NAME:
        src_class_dir = src_dir / wnid
        for img_path in sorted(src_class_dir.glob("*.JPEG")):
            shutil.move(str(img_path), str(dest_dir / img_path.name))
            entries.append((img_path.stem, wnid))
    return entries


def _write_image_sets(data_root: Path, paths: list[str], file_names: list[str]) -> None:
    image_sets_dir = data_root / "ILSVRC/ImageSets/CLS-LOC"
    image_sets_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"{path} {idx}\n" for idx, path in enumerate(paths, start=1)]
    for file_name in file_names:
        (image_sets_dir / file_name).write_text("".join(lines))


def _write_val_solution(data_root: Path, entries: list[tuple[str, str]]) -> None:
    lines = ["ImageId,PredictionString\n"]
    lines.extend(f"{stem},{wnid} 0 0 1 1\n" for stem, wnid in entries)
    (data_root / "LOC_val_solution.csv").write_text("".join(lines))


def main(args: Args) -> None:
    """Download Imagenette and lay it out as an ImageNet/ILSVRC-style root."""
    data_root = Path(args.data_root)
    if (data_root / "LOC_synset_mapping.txt").exists():
        return
    data_root.mkdir(parents=True, exist_ok=True)

    url = _SOURCES[args.size]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        archive = tmp_dir_path / get_url_filename(url)
        download(url=url, dest_path=archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_dir_path)  # noqa: S202
        extracted_dir = tmp_dir_path / _EXTRACTED_DIRS[args.size]

        train_paths = _relayout_train(extracted_dir, data_root)
        val_entries = _relayout_val(extracted_dir, data_root)

    _write_synset_mapping(data_root)
    _write_image_sets(data_root, train_paths, ["train_cls.txt", "train_loc.txt"])
    _write_image_sets(data_root, [stem for stem, _ in val_entries], ["val.txt"])
    _write_val_solution(data_root, val_entries)


if __name__ == "__main__":
    main(tyro.cli(Args))
