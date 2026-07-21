"""World Terrestrial Ecosystems (WTE) 2020 dataset downloader.

A single global 250m-resolution categorical raster (431 ecosystem types,
EPSG:4326) from USGS/Esri/TNC, distributed via ScienceBase as an Esri map
package -- itself just a 7z archive around a plain GeoTIFF plus a value
attribute table (VAT) giving each class code's ecosystem legend.

https://www.sciencebase.gov/catalog/item/6296791ed34ec53d276bb293
"""

import shutil
from pathlib import Path

import py7zr
import tyro
from pydantic import BaseModel

from rsrch_data.utils.download import download

ARCHIVE_URL = (
    "https://www.sciencebase.gov/catalog/file/get/6296791ed34ec53d276bb293"
    "?f=__disk__8f%2F6d%2F7a%2F8f6d7a1a72784bc0a890073c4c4d5bbea80c7479"
)
ARCHIVE_MEMBER_DIR = "commondata/raster_data"
ARCHIVE_MEMBERS = [
    f"{ARCHIVE_MEMBER_DIR}/{name}"
    for name in (
        "WorldEcosystem.tif",
        "WorldEcosystem.tfw",
        "WorldEcosystem.tif.aux.xml",
        "WorldEcosystem.tif.vat.dbf",
        "WorldEcosystem.tif.vat.cpg",
        "WorldEcosystem.tif.xml",
    )
]


class Args(BaseModel):
    """CLI args for `get_wte.py` script."""

    data_root: str
    archive_url: str = ARCHIVE_URL


def main(args: Args) -> None:
    """Get the WTE 2020 dataset (single global raster, EPSG:4326)."""
    data_root = Path(args.data_root)
    raster_dir = data_root / "raster"
    if (raster_dir / "WorldEcosystem.tif").exists():
        return

    archive_path = data_root / "USGSEsriTNCWorldTerrestrialEcosystems2020.mpkx"
    download(args.archive_url, archive_path)

    raster_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, "r") as z:
        z.extract(path=raster_dir, targets=ARCHIVE_MEMBERS)

    for member in ARCHIVE_MEMBERS:
        (raster_dir / member).rename(raster_dir / Path(member).name)
    shutil.rmtree(raster_dir / ARCHIVE_MEMBER_DIR.split("/", maxsplit=1)[0])


if __name__ == "__main__":
    main(tyro.cli(Args))
