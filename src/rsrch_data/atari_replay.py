"""Reader for the Atari transition shards written by `rsrch.rl.dataset.Recorder`.

Two accessors, mirroring `tokens_bin.py`'s `TokensBinDocs`/`TokensBinSegments`
split: `AtariEpisodes` gives whole, variable-length episodes (see its
docstring for the on-disk file structure); `AtariWindows` wraps it to give
fixed-length windows that never cross an episode boundary (for offline
world-model training).
"""

import json
import zlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from rsrch_data.registry import register_dataset


class Episode(TypedDict):
    """One full episode (or on-disk segment, see module docstring)."""

    obs: np.ndarray
    """Observation frames, shape `(L, H, W)`, dtype `uint8`."""

    act: np.ndarray
    """Actions, shape `(L,)`, dtype `int32`. `act[-1]` is invalid -- no
    action was taken from the episode's last row."""

    reward: np.ndarray
    """Rewards, shape `(L,)`, dtype `float32`."""

    term: np.ndarray
    """Terminal flags, shape `(L,)`, dtype `bool`."""

    trunc: np.ndarray
    """Truncation flags, shape `(L,)`, dtype `bool`."""

    is_first: np.ndarray
    """Shape `(L,)`, dtype `bool`. `is_first[0]` is `True` iff this is a
    genuine episode start (not a safety-valve continuation segment); the
    rest are always `False`."""


class Window(TypedDict):
    """A fixed-length window into one episode; never crosses its boundary."""

    obs: np.ndarray
    """Observation frames, shape `(L, H, W)`, dtype `uint8`."""

    act: np.ndarray
    """Actions, shape `(L,)`, dtype `int32`. `act[-1]` is invalid iff this
    window ends exactly at its episode's last row."""

    reward: np.ndarray
    """Rewards, shape `(L,)`, dtype `float32`."""

    term: np.ndarray
    """Terminal flags, shape `(L,)`, dtype `bool`."""

    trunc: np.ndarray
    """Truncation flags, shape `(L,)`, dtype `bool`."""

    is_first: np.ndarray
    """Shape `(L,)`, dtype `bool`. `is_first[0]` is `True` only when the
    window starts at row 0 of a genuine episode start; the rest are always
    `False` (windows never cross an episode boundary)."""


@dataclass
class _Shard:
    """In-memory handle to one shard's index/step arrays and frame file."""

    frames_path: Path
    idx: np.ndarray
    act: np.ndarray
    reward: np.ndarray
    term: np.ndarray
    trunc: np.ndarray
    episodes: np.ndarray


class _ChunkCache:
    """Small manual LRU cache of decompressed frame chunks."""

    def __init__(self, maxsize: int = 32) -> None:
        """Cache up to `maxsize` decompressed chunks."""
        self._maxsize = maxsize
        self._data: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get(self, key: tuple[int, int]) -> np.ndarray | None:
        """Return the cached chunk for `key`, or `None` on a miss."""
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: tuple[int, int], value: np.ndarray) -> None:
        """Cache `value` under `key`, evicting the oldest entry if full."""
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


_CHECKED_META_FIELDS = ("chunk_size", "obs_shape", "obs_dtype", "num_actions", "env")


@register_dataset("atari-replay-episodes")
class AtariEpisodes(Sequence):
    """Random-access dataset over whole episodes from `Recorder`-written shards.

    File structure:
    ```
    <data_root>/
    ├── shard-00000/
    │   ├── meta.json         # codec/chunk_size/obs_shape/.../env config
    │   ├── frames.bin        # zlib-compressed chunks of `chunk_size` frames
    │   ├── frames.idx.npy    # (num_chunks + 1,) uint64 byte offsets into frames.bin
    │   └── steps.npz         # act/reward/term/trunc (K,) + episodes (E, 3) int64
    └── shard-00001/
        └── ...
    ```

    Row order within a shard is episode-contiguous: even though transitions
    arrive interleaved across parallel envs during collection, each
    episode's rows are written contiguously and in time order.
    `episodes[e] = (start, end, first)` gives row range `[start, end)` for
    episode `e`; `first` is 1 if the episode segment starts at a genuine env
    reset, 0 if it's a continuation of an episode force-closed by
    `Recorder`'s safety valve in a previous shard. Chunks are fixed-size
    over this row order and may straddle episode boundaries. Row `i` holds
    `(obs_i, act_i, reward_i)` with `act_i`/`reward_i` being the
    action/reward *from* `obs_i` (not leading to it); `term_i`/`trunc_i`
    describe `obs_i` itself. An episode segment's last row has no valid
    `act`/`reward`. See `rsrch.rl.dataset`'s module docstring (the writer)
    for the full rationale.

    Owns shard loading/validation and chunk decompression (cached); each
    `__getitem__` decompresses the whole episode. For a sub-window instead
    (the common case for training), use `read_range`/`episode_length`
    directly, or `AtariWindows`, rather than slicing `self[index]` -- that
    would decompress the full episode just to throw most of it away.
    """

    def __init__(self, data_root: str | Path) -> None:
        """Index all shards under `data_root`.

        :param data_root: Dataset root containing `shard-*/` subdirectories.
        """
        data_root = Path(data_root)
        shard_dirs = sorted(data_root.glob("shard-*"))
        if not shard_dirs:
            msg = f"No shards found under {data_root}"
            raise ValueError(msg)

        self._metas: list[dict[str, Any]] = []
        self._shards: list[_Shard] = []
        self._episode_meta: list[tuple[int, int, int, bool]] = []

        for shard_idx, shard_dir in enumerate(shard_dirs):
            with (shard_dir / "meta.json").open() as f:
                meta = json.load(f)

            if shard_idx > 0:
                self._check_meta(meta, shard_dir, shard_dirs[0])
            self._metas.append(meta)

            idx = np.load(shard_dir / "frames.idx.npy")
            steps = np.load(shard_dir / "steps.npz")
            shard = _Shard(
                frames_path=shard_dir / "frames.bin",
                idx=idx,
                act=steps["act"],
                reward=steps["reward"],
                term=steps["term"],
                trunc=steps["trunc"],
                episodes=steps["episodes"],
            )
            self._shards.append(shard)

            for start, end, first in shard.episodes:
                self._episode_meta.append(
                    (shard_idx, int(start), int(end), bool(first))
                )

        self.chunk_size: int = self._metas[0]["chunk_size"]
        self.obs_shape: tuple[int, ...] = tuple(self._metas[0]["obs_shape"])
        self.obs_dtype: npt.DTypeLike = np.dtype(self._metas[0]["obs_dtype"])
        self.num_actions: int = self._metas[0]["num_actions"]

        self._cache = _ChunkCache(maxsize=32)

    def _check_meta(
        self, meta: dict[str, Any], shard_dir: Path, first_dir: Path
    ) -> None:
        first_meta = self._metas[0]
        for field in _CHECKED_META_FIELDS:
            if meta[field] != first_meta[field]:
                msg = (
                    f"Shard metadata mismatch in field {field!r}: "
                    f"{first_dir} has {first_meta[field]!r}, "
                    f"{shard_dir} has {meta[field]!r}"
                )
                raise ValueError(msg)

    def __len__(self) -> int:
        return len(self._episode_meta)

    def __getitem__(self, index: int) -> Episode:
        if index < 0:
            index += len(self)
        shard_idx, start, end, first = self._episode_meta[index]
        return self._read(shard_idx, start, end, is_first_at_0=first)

    def episode_length(self, index: int) -> int:
        """Row count of episode `index`, without reading any frame data.

        :param index: Episode index into this dataset.
        """
        if index < 0:
            index += len(self)
        _, start, end, _ = self._episode_meta[index]
        return end - start

    def read_range(self, index: int, start: int, end: int) -> Window:
        """Read local rows `[start, end)` of episode `index`.

        Only decompresses the chunks the requested range actually touches --
        what `AtariWindows` uses instead of `self[index]` to avoid paying for
        the whole episode on every window.

        :param index: Episode index into this dataset.
        :param start: Local start row, inclusive.
        :param end: Local end row, exclusive.
        """
        if index < 0:
            index += len(self)
        shard_idx, ep_start, ep_end, first = self._episode_meta[index]
        row_start, row_end = ep_start + start, ep_start + end
        if row_start < ep_start or row_end > ep_end:
            msg = (
                f"range [{start}, {end}) exceeds episode {index}'s length "
                f"{ep_end - ep_start}"
            )
            raise IndexError(msg)
        return self._read(
            shard_idx, row_start, row_end, is_first_at_0=first and start == 0
        )

    def _read(
        self, shard_idx: int, row_start: int, row_end: int, *, is_first_at_0: bool
    ) -> dict[str, np.ndarray]:
        shard = self._shards[shard_idx]

        c0 = row_start // self.chunk_size
        c1 = (row_end - 1) // self.chunk_size
        chunks = [self._get_chunk(shard_idx, c) for c in range(c0, c1 + 1)]
        frames = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        local_start = row_start - c0 * self.chunk_size
        obs = frames[local_start : local_start + (row_end - row_start)]

        is_first = np.zeros(row_end - row_start, dtype=np.bool_)
        is_first[0] = is_first_at_0

        return {
            "obs": obs,
            "act": shard.act[row_start:row_end],
            "reward": shard.reward[row_start:row_end],
            "term": shard.term[row_start:row_end],
            "trunc": shard.trunc[row_start:row_end],
            "is_first": is_first,
        }

    def _get_chunk(self, shard_idx: int, chunk_idx: int) -> np.ndarray:
        key = (shard_idx, chunk_idx)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        shard = self._shards[shard_idx]
        off0 = int(shard.idx[chunk_idx])
        off1 = int(shard.idx[chunk_idx + 1])
        with shard.frames_path.open("rb") as f:
            f.seek(off0)
            compressed = f.read(off1 - off0)
        raw = zlib.decompress(compressed)
        chunk = np.frombuffer(raw, dtype=self.obs_dtype).reshape(-1, *self.obs_shape)

        self._cache.put(key, chunk)
        return chunk

    def meta(self) -> dict[str, Any]:
        """Return the dataset's (shard-identical) config, plus `num_shards`."""
        base = dict(self._metas[0])
        for field in ("num_frames", "num_episodes", "num_chunks", "created_at"):
            base.pop(field, None)
        base["num_shards"] = len(self._shards)
        return base


@register_dataset("atari-replay-windows")
class AtariWindows(Sequence):
    """Fixed-length windows into `AtariEpisodes`, never crossing an episode boundary.

    Episodes shorter than `seq_len` are skipped, not padded. Wraps an
    `AtariEpisodes` and uses its `read_range` (not `__getitem__`), so a
    window only decompresses the chunks it actually touches, regardless of
    how long its episode is.
    """

    def __init__(
        self,
        data_root: str | Path,
        seq_len: int = 64,
        stride: int = 1,
    ) -> None:
        """Index all windows across every episode under `data_root`.

        :param data_root: Dataset root containing `shard-*/` subdirectories.
        :param seq_len: Length of each returned window, in rows.
        :param stride: Step between consecutive window start offsets within
            an episode.
        """
        self._episodes = AtariEpisodes(data_root)
        self.seq_len = seq_len
        self.stride = stride

        self._index: list[tuple[int, int]] = []
        for ep_idx in range(len(self._episodes)):
            n = self._episodes.episode_length(ep_idx)
            if n < seq_len:
                continue
            for offset in range(0, n - seq_len + 1, stride):
                self._index.append((ep_idx, offset))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Window:
        if index < 0:
            index += len(self)
        ep_idx, offset = self._index[index]
        return self._episodes.read_range(ep_idx, offset, offset + self.seq_len)

    def meta(self) -> dict[str, Any]:
        """Return the dataset's (shard-identical) config, plus `num_shards`."""
        return self._episodes.meta()
