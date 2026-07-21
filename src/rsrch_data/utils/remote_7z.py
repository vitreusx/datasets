"""Read/extract 7z archives over HTTP without downloading them in full.

7z's own metadata ("header": file list, and which files share a compressed
"folder"/solid-block) sits near the end of the archive and is tiny relative
to the payload. `py7zr` already knows how to parse it and to extract just the
folder(s) containing a chosen subset of files -- it only needs a seekable
byte stream, which doesn't have to be a local file. `RemoteSevenZip` supplies
that stream via HTTP Range requests, so both listing and targeted extraction
only fetch the bytes actually needed (empirically: a few KB to list, and at
most one solid-block's worth of bytes to extract a subset -- far less than
the whole archive either way).
"""

import io
import time
from collections.abc import Callable
from pathlib import Path

import py7zr
import requests


class RateLimiter:
    """Simple min-interval pacer, so we don't trip a server's rate limit."""

    def __init__(self, per_second: float = 1.0) -> None:
        """Allow at most `per_second` calls to `wait` per second."""
        self.min_interval = 1.0 / per_second
        self._last = 0.0

    def wait(self) -> None:
        """Block until it's been at least `1/per_second` since the last call."""
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last = time.monotonic()


class RemoteSevenZip(io.RawIOBase):
    """A seekable stream over one or more remote 7z volumes, via HTTP Range."""

    def __init__(
        self,
        urls: list[str],
        rate_limiter: RateLimiter | None = None,
        on_fetch: Callable[[int], None] | None = None,
    ) -> None:
        """Index volume sizes for `urls` (1 per 7z volume, in order).

        `on_fetch`, if given, is called with the number of bytes fetched
        after every `read()` -- useful for progress reporting.
        """
        super().__init__()
        self.urls = urls
        self.rate_limiter = rate_limiter or RateLimiter()
        self.on_fetch = on_fetch
        self.pos = 0
        self.total_fetched = 0
        self.num_requests = 0

        self.sizes = [self._fetch_size(url) for url in urls]
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.size = self.offsets[-1]

    def _fetch_size(self, url: str) -> int:
        self.rate_limiter.wait()
        resp = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=30)
        self.num_requests += 1
        return int(resp.headers["content-range"].split("/")[-1])

    def seekable(self) -> bool:
        """Report that this stream supports `seek`."""
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the virtual read position, spanning all volumes as one stream."""
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        """Get the current virtual read position."""
        return self.pos

    def readable(self) -> bool:
        """Report that this stream supports `read`."""
        return True

    def _volume_at(self, byte_pos: int) -> int:
        return next(
            i
            for i in range(len(self.sizes))
            if self.offsets[i] <= byte_pos < self.offsets[i + 1]
        )

    def _fetch_range(self, vol_idx: int, start: int, end: int) -> bytes:
        self.rate_limiter.wait()
        resp = requests.get(
            self.urls[vol_idx], headers={"Range": f"bytes={start}-{end}"}, timeout=30
        )
        self.num_requests += 1
        return resp.content

    def readinto(self, b: bytearray) -> int:
        """Read into a pre-allocated buffer, as required by `io.RawIOBase`."""
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, n: int = -1) -> bytes:
        """Read up to `n` bytes (or to EOF) from the current position."""
        end_pos = self.size if (n is None or n < 0) else min(self.pos + n, self.size)
        if self.pos >= end_pos:
            return b""

        out = bytearray()
        cur = self.pos
        while cur < end_pos:
            vol_idx = self._volume_at(cur)
            vol_start = self.offsets[vol_idx]
            local_start = cur - vol_start
            local_end = min(end_pos, self.offsets[vol_idx + 1]) - vol_start - 1
            data = self._fetch_range(vol_idx, local_start, local_end)
            out += data
            cur += len(data)

        self.pos = cur
        self.total_fetched += len(out)
        if self.on_fetch is not None:
            self.on_fetch(len(out))
        return bytes(out)


def open_remote_7z(
    urls: list[str],
    rate_limiter: RateLimiter | None = None,
    on_fetch: Callable[[int], None] | None = None,
) -> tuple[py7zr.SevenZipFile, RemoteSevenZip]:
    """Open a remote 7z archive (1+ volumes) for listing/extraction.

    Returns the open `SevenZipFile` (caller must close it, or use as a
    context manager) alongside the underlying `RemoteSevenZip` stream, which
    exposes `total_fetched`/`num_requests` for cost tracking.
    """
    stream = RemoteSevenZip(urls, rate_limiter=rate_limiter, on_fetch=on_fetch)
    return py7zr.SevenZipFile(stream, mode="r"), stream


def extract_remote_7z(
    urls: list[str],
    targets: list[str],
    dest_dir: Path,
    rate_limiter: RateLimiter | None = None,
    on_fetch: Callable[[int], None] | None = None,
) -> None:
    """Extract just `targets` from a remote 7z archive into `dest_dir`.

    Only downloads+decompresses the solid block(s) containing `targets`, not
    the whole archive. `on_fetch`, if given, is called with the number of
    bytes fetched after every underlying read -- useful for progress bars.
    """
    with open_remote_7z(urls, rate_limiter=rate_limiter, on_fetch=on_fetch)[0] as z:
        z.extract(path=dest_dir, targets=targets)
