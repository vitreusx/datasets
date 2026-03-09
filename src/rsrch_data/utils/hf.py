import re
from pathlib import Path

import numpy as np
from huggingface_hub import DryRunFileInfo, snapshot_download

SIZE_PAT = r"^(?P<count>[1-9][0-9]*(.[0-9]+)?)\s*(?P<unit>[a-z]*)$"

UNITS = {
    "b": 1,
    "k": 1024,
    "kb": 1024,
    "m": 1024**2,
    "mb": 1024**2,
    "g": 1024**3,
    "gb": 1024**3,
}

DATASET_ID = "HuggingFaceFW/fineweb-2"


def parse_size(size: int | str):
    if isinstance(size, str):
        size = size.lower()
        m = re.match(SIZE_PAT, size)
        if m is None:
            msg = f"{m} is not a size string (regex: {SIZE_PAT})"
            raise ValueError(msg)
        count = float(m.group("count"))
        unit = m.group("unit")
        size = count * UNITS[unit]
    return size


def fetch(
    dataset_id: str,
    data_root: str | Path,
    allow_patterns: str | list[str] | None = None,
    max_dl_size: int | str | None = None,
    seed: int = 0,
):
    if max_dl_size is not None:
        files: list[DryRunFileInfo] = snapshot_download(
            dataset_id,
            repo_type="dataset",
            allow_patterns=allow_patterns,
            dry_run=True,
        )
        files = sorted(files, key=lambda info: info.filename)
        gen = np.random.default_rng(seed=seed)
        files = [files[idx] for idx in gen.permutation(len(files))]
        total, idx = 0, 0
        max_dl_size = parse_size(max_dl_size)
        file_list = []
        while total < max_dl_size and idx < len(files):
            info = files[idx]
            file_list.append(info.filename)
            total += info.file_size
            idx += 1
        allow_patterns = file_list

    snapshot_download(
        dataset_id,
        repo_type="dataset",
        local_dir=data_root,
        allow_patterns=allow_patterns,
    )
