"""Resize IN-1k to a given size, and export it."""

import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tyro
from PIL import Image
from pydantic import BaseModel
from tqdm.auto import tqdm


def transform_in1k(
    input_dir: str | Path,
    output_dir: str | Path,
    image_transform: Callable[[Image.Image], Image.Image],
    export_opts: dict | None = None,
) -> None:
    """Transform images of ImageNet-1k."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "ILSVRC/Annotations",
        "ILSVRC/ImageSets",
        "LOC_synset_mapping.txt",
        "LOC_train_solution.csv",
        "LOC_val_solution.csv",
    ):
        input_path = input_dir / name
        output_path = output_dir / name
        if not input_path.exists() or output_path.exists():
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if input_path.is_file():
            shutil.copyfile(input_path, output_path)
        else:
            shutil.copytree(input_path, output_path)

    image_paths = []
    train_img_root = input_dir / "ILSVRC/Data/CLS-LOC/train"
    with (input_dir / "ILSVRC/ImageSets/CLS-LOC/train_cls.txt").open() as f:
        for line in f:
            stem = line.split(maxsplit=1)[0].strip()
            image_paths.append(train_img_root / f"{stem}.JPEG")

    val_img_root = input_dir / "ILSVRC/Data/CLS-LOC/val"
    with (input_dir / "ILSVRC/ImageSets/CLS-LOC/val.txt").open() as f:
        for line in f:
            stem = line.split(maxsplit=1)[0].strip()
            image_paths.append(val_img_root / f"{stem}.JPEG")

    if export_opts is None:
        export_opts = {"quality": 90, "subsampling": 0}

    def process(input_path: Path) -> None:
        output_path = output_dir / input_path.relative_to(input_dir)
        image = Image.open(input_path)
        out_image = image_transform(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_image.save(output_path, **export_opts)

    with ThreadPoolExecutor() as pool:
        tasks = [pool.submit(process, path) for path in image_paths]
        for _ in tqdm(as_completed(tasks), total=len(tasks)):
            pass


class Args(BaseModel):
    """Args for `resize_in1k` script."""

    input_dir: str
    output_dir: str
    center_crop: bool = True
    smallest_size: int | None = None


def resize_in1k(args: Args) -> None:
    """Resize IN-1k dataset and export it to a new location."""

    def image_transform(image: Image.Image) -> Image.Image:
        if args.center_crop:
            min_size = min(image.width, image.height)
            left = (image.width - min_size) // 2
            top = (image.height - min_size) // 2
            image = image.crop((left, top, left + min_size, top + min_size))
        if args.smallest_size is not None:
            if image.width > image.height:
                new_w = int(image.width / image.height * args.smallest_size)
                new_h = args.smallest_size
            else:
                new_h = int(image.height / image.width * args.smallest_size)
                new_w = args.smallest_size
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    transform_in1k(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_transform=image_transform,
    )


if __name__ == "__main__":
    resize_in1k(tyro.cli(Args))
