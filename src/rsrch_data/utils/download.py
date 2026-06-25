"""File download and extraction utilities."""

import asyncio
import bz2
import os
import tarfile
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx
import requests
from tqdm.auto import tqdm


async def async_download(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    sem: asyncio.Semaphore,
) -> None:
    """Download a single file asynchronously with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers = {}
    if dest.exists():
        existing = dest.stat().st_size
        headers["Range"] = f"bytes={existing}-"
    else:
        existing = 0

    async with sem, client.stream("GET", url, headers=headers) as r:
        if r.status_code in (404, 416):
            return
        r.raise_for_status()
        mode = "ab" if existing else "wb"
        with dest.open(mode) as f:
            async for chunk in r.aiter_bytes(65536):
                f.write(chunk)


def _tar_stream_mode(url: str) -> str:
    suffix = PurePosixPath(urlparse(url).path).suffix
    return "r|gz" if suffix == ".gz" else "r|"


def _tar_file_mode(url: str) -> str:
    suffix = PurePosixPath(urlparse(url).path).suffix
    return "r:gz" if suffix == ".gz" else "r:"


async def async_download_and_extract(
    client: httpx.AsyncClient,
    url: str,
    dest_dir: Path,
    sem: asyncio.Semaphore,
    archive_dest_path: Path | None = None,
    archive_dest_dir: Path | None = None,
) -> None:
    """Download and extract a tar archive asynchronously.

    If neither archive_dest_path nor archive_dest_dir is given, streams
    directly through a pipe into tarfile without saving to disk.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    if archive_dest_path is not None or archive_dest_dir is not None:
        if archive_dest_path is None:
            archive_dest_path = Path(archive_dest_dir) / get_url_filename(url)
        await async_download(client, url, archive_dest_path, sem)
        with tarfile.open(archive_dest_path, _tar_file_mode(url)) as tf:
            tf.extractall(dest_dir)  # noqa: S202
    else:
        rp_fd, wp_fd = os.pipe()
        rp = os.fdopen(rp_fd, "rb", 0)
        wp = os.fdopen(wp_fd, "wb", 0)
        loop = asyncio.get_running_loop()

        def _extract() -> None:
            with tarfile.open(fileobj=rp, mode=_tar_stream_mode(url)) as tf:
                tf.extractall(dest_dir)  # noqa: S202

        extract = loop.run_in_executor(None, _extract)
        try:
            async with sem, client.stream("GET", url) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(65536):
                    await loop.run_in_executor(None, wp.write, chunk)
        finally:
            wp.close()

        await extract
        rp.close()


def get_url_filename(url: str) -> str:
    """Extract the filename portion from a URL."""
    return PurePosixPath(urlparse(url).path).name


def download(
    url: str,
    dest_path: str | Path | None = None,
    dest_dir: str | Path | None = None,
    *,
    display_pbar: bool = True,
) -> None:
    """Download url to dest_path (or dest_dir/<filename>).

    Resumes the download if the file already exists.
    """
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
        resp = requests.get(url, stream=True, timeout=(10, None))
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with (
            dest_path.open("wb") as file,
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
    dest_path: str | Path,
    archive_dest_path: str | Path | None = None,
    archive_dest_dir: str | Path | None = None,
    *,
    display_pbar: bool = True,
) -> None:
    """Download an archive from url and extract it into dest path.

    Destination can be either a directory or a file, depending on the archive.
    """
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

        dest_path = Path(dest_path)

        archive_fmt = PurePosixPath(urlparse(url).path).suffix
        if archive_fmt == ".zip":
            dest_path.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_dest_path) as xf:
                xf.extractall(dest_path)  # noqa: S202
        elif archive_fmt == ".tar":
            dest_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_dest_path, "r") as xf:
                xf.extractall(dest_path)  # noqa: S202
        elif archive_fmt == ".tar.gz":
            dest_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_dest_path, "r:gz") as xf:
                xf.extractall(dest_path)  # noqa: S202
        elif archive_fmt.endswith(".bz2"):
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with bz2.open(archive_dest_path) as xf, dest_path.open("rb") as f:
                f.write(xf.read())
        else:
            msg = f"Unknown archive format {archive_fmt}"
            raise ValueError(msg)
