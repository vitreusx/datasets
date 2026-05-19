"""Download OpenAI VPT contractor dataset."""

import asyncio
from pathlib import Path
from typing import Literal

import httpx
import tyro
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from rsrch_data.utils.download import async_download

DatasetName = Literal[
    "6xx",
    "7xx",
    "8xx",
    "9xx",
    "10xx",
    "find-cave",
    "waterfall",
    "pen-animals",
    "build-house",
]

INDEXES: dict[DatasetName, str] = {
    "6xx": "all_6xx_Jun_29.json",
    "7xx": "all_7xx_Apr_6.json",
    "8xx": "all_8xx_Jun_29.json",
    "9xx": "all_9xx_Jun_29.json",
    "10xx": "all_10xx_Jun_29.json",
    "find-cave": "find-cave-Jul-28.json",
    "waterfall": "waterfall-Jul-28.json",
    "pen-animals": "pen-animals-Jul-28.json",
    "build-house": "build-house-Jul-28.json",
}


class Config(BaseModel):
    """Configuration for the get_vpt script."""

    out: Path = Path("data/vpt")
    """Directory to write the dataset"""

    datasets: list[DatasetName] = list(INDEXES.keys())  # type: ignore[assignment]
    """Datasets to download"""

    concurrency: int = 16
    """Number of concurrent downloads"""

    base_url: str = "https://openaipublic.blob.core.windows.net/minecraft-rl/snapshots/"
    """Base URL for index files"""


async def main(cfg: Config) -> None:
    """Fetch index files and download all requested datasets."""
    urls: list[tuple[str, Path]] = []
    async with httpx.AsyncClient(timeout=30) as client:
        for name in cfg.datasets:
            r = await client.get(cfg.base_url + INDEXES[name])
            r.raise_for_status()
            index = r.json()
            basedir = index["basedir"]
            for relpath in index["relpaths"]:
                stem = relpath.removesuffix(".mp4")
                for ext in (".mp4", ".jsonl", "-options.json"):
                    urls.append(  # noqa: PERF401
                        (basedir + stem + ext, cfg.out / (stem + ext)),
                    )

    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        with tqdm(total=len(urls), unit="file") as pbar:
            await asyncio.gather(
                *(async_download(client, url, dest, sem, pbar) for url, dest in urls)
            )


if __name__ == "__main__":
    asyncio.run(main(tyro.cli(Config)))
