import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm


def get_url_filename(url: str):
    return PurePosixPath(urlparse(url).path).name


def download(
    url: str,
    dest_path: str | Path | None = None,
    dest_dir: str | Path | None = None,
    display_pbar: bool = True,
):
    if dest_path is not None:
        dest_path = Path(dest_path)
    elif dest_dir is not None:
        dest_path = Path(dest_dir) / get_url_filename(url)
    else:
        msg = "Either `dest_path` or `dest_dir` must be provided"
        raise ValueError(msg)

    if dest_path.exists():
        return

    scheme = urlparse(url).scheme
    if scheme in ("http", "https"):
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            open(dest_path, "wb") as file,
            tqdm(
                desc=f"{url} -> {dest_path}",
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                disable=not display_pbar,
            ) as bar,
        ):
            for data in resp.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    else:
        msg = f"Unsupported scheme {scheme}"
        raise ValueError(msg)


def download_and_extract(
    url: str,
    dest_dir: str | Path,
    archive_dest_path: str | Path | None = None,
    archive_dest_dir: str | Path | None = None,
    display_pbar: bool = True,
):
    with ExitStack() as stack:
        if archive_dest_path is not None:
            archive_dest_path = Path(archive_dest_path)
        elif archive_dest_dir is not None:
            archive_dest_path = Path(archive_dest_dir) / get_url_filename(url)
        else:
            tmp_d = stack.enter_context(tempfile.TemporaryDirectory())
            archive_dest_path = Path(tmp_d) / get_url_filename(url)

        download(
            url=url,
            dest_path=archive_dest_path,
            display_pbar=display_pbar,
        )

        archive_fmt = PurePosixPath(urlparse(url).path).suffix
        if archive_fmt == ".zip":
            archive = zipfile.ZipFile(archive_dest_path)
        elif archive_fmt == ".tar":
            archive = tarfile.open(archive_dest_path, "r")  # noqa: SIM115
        elif archive_fmt == ".tar.gz":
            archive = tarfile.open(archive_dest_path, "r:gz")  # noqa: SIM115
        else:
            msg = f"Unknown archive format {archive_fmt}"
            raise ValueError(msg)

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        with archive as xf:
            xf.extractall(dest_dir)  # noqa: S202
