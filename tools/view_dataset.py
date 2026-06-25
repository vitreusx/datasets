"""View dataset samples."""

import inspect
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import tyro
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.pretty import pretty_repr

from rsrch_data.registry import get_registry


def _view_images(dataset: object, db_uri: str | None) -> None:
    """View image dataset samples in FiftyOne."""
    if db_uri is not None:
        os.environ["FIFTYONE_DATABASE_URI"] = db_uri

    import fiftyone as fo  # type: ignore[import-untyped]

    def _fo_samples(tmp_path: Path) -> Iterator[Any]:
        for i, sample in enumerate(dataset):
            image = sample if isinstance(sample, Image.Image) else sample["image"]
            filepath = getattr(image, "filename", None) or ""
            if not filepath:
                filepath = str(tmp_path / f"{i:07d}.png")
                image.save(filepath)
            yield fo.Sample(filepath=filepath)

    fo_dataset = fo.Dataset()
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = fo.launch_app(fo_dataset)
        fo_dataset.add_samples(_fo_samples(Path(tmp_dir)))
        session.wait()


def _view_spatial(dataset: object) -> None:
    """View spatial dataset (images + camera poses) in Rerun."""
    import numpy as np
    import rerun as rr

    rr.init("view_dataset", spawn=True)

    for i, sample in enumerate(dataset):
        image: Image.Image = sample["image"]
        c2w: np.ndarray = sample["c2w"]
        k: np.ndarray = sample["K"]

        rr.set_time_sequence("frame", i)
        rr.log(
            "world/camera",
            rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
        )
        rr.log(
            "world/camera",
            rr.Pinhole(
                image_from_camera=k,
                width=image.width,
                height=image.height,
                camera_xyz=rr.ViewCoordinates.RDF,
            ),
        )
        rr.log("world/camera/image", rr.Image(np.array(image)))


def _view_text(dataset: object) -> None:
    """View text dataset samples via rich + less."""
    less = shutil.which("less") or "less"
    proc = subprocess.Popen(  # noqa: S603
        [less, "-R"], stdin=subprocess.PIPE, text=True
    )
    console = Console(file=proc.stdin, force_terminal=True)
    try:
        for i, sample in enumerate(dataset):
            console.print(Panel(sample["text"], title=f"Sample {i}"))
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
        proc.wait()


def _view_json(dataset: Any) -> None:
    """View JSON dataset samples via rich + less."""
    less = shutil.which("less") or "less"
    proc = subprocess.Popen(  # noqa: S603
        [less, "-R"], stdin=subprocess.PIPE, text=True
    )
    console = Console(file=proc.stdin, force_terminal=True)
    try:
        for i, sample in enumerate(dataset):
            console.print(Panel(pretty_repr(sample), title=f"Sample {i}"))
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
        proc.wait()


def _is_json_serializable(value: Any) -> bool:
    try:
        json.dumps(value)
        return True  # noqa: TRY300
    except Exception:
        return False


def main() -> None:
    """View dataset samples."""
    annotated = [
        Annotated[
            cls,
            tyro.conf.subcommand(
                name,
                description=(inspect.getdoc(cls) or "").partition("\n")[0],
            ),
        ]
        for name, cls in get_registry().items()
    ]
    union_type = annotated[0]
    for t in annotated[1:]:
        union_type = union_type | t

    @dataclass
    class Args:
        dataset: union_type  # pyright: ignore[reportInvalidTypeForm]
        db_uri: str | None = None

    tyro_conf = (tyro.conf.OmitArgPrefixes, tyro.conf.OmitSubcommandPrefixes)
    args = tyro.cli(Args, config=tyro_conf)

    it = iter(args.dataset)
    sample = next(it)
    dataset = itertools.chain([sample], it)

    if isinstance(sample, Mapping) and "c2w" in sample:
        _view_spatial(dataset)
    elif isinstance(sample, Image.Image) or (
        isinstance(sample, Mapping) and isinstance(sample.get("image"), Image.Image)
    ):
        _view_images(dataset, args.db_uri)
    elif isinstance(sample, Mapping) and isinstance(sample.get("text"), str):
        _view_text(dataset)
    elif _is_json_serializable(sample):
        _view_json(dataset)
    else:
        msg = f"Don't know how to view samples of type {type(sample)!r}"
        raise TypeError(msg)


if __name__ == "__main__":
    main()
