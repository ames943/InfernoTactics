"""Fetch pre-fire LANDFIRE 2024 FBFM40 data for the study area.

The official LANDFIRE Web Coverage Service returns pixel values suitable for
analysis, unlike a rendered WMS image. The request is spatially clipped before
download, avoiding the multi-gigabyte CONUS archive.
"""

import os

import rasterio
import requests

from infernotactics.data.config import (
    BBOX_LEFT_BOTTOM_RIGHT_TOP,
    DATA_DIR,
    FUEL_MODEL_PATH,
    STUDY_AREA_NAME,
)


WCS_URL = "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_2024/wcs"
COVERAGE_ID = "landfire_wcs__LF2024_FBFM40_CONUS"
WCS_VERSION = "2.0.1"


def fetch_fuels():
    """Download a clipped categorical FBFM40 GeoTIFF and validate it."""
    west, south, east, north = BBOX_LEFT_BOTTOM_RIGHT_TOP
    params = [
        ("service", "WCS"),
        ("version", WCS_VERSION),
        ("request", "GetCoverage"),
        ("coverageId", COVERAGE_ID),
        ("format", "image/tiff"),
        ("subset", f"Lat({south},{north})"),
        ("subset", f"Long({west},{east})"),
        ("subsettingcrs", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("outputcrs", "http://www.opengis.net/def/crs/EPSG/0/5070"),
    ]

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Fetching LANDFIRE 2024 FBFM40 for {STUDY_AREA_NAME}")
    response = requests.get(WCS_URL, params=params, timeout=180)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "tiff" not in content_type.lower():
        raise RuntimeError(
            f"LANDFIRE returned {content_type!r}, not a GeoTIFF: {response.text[:500]}"
        )

    with rasterio.io.MemoryFile(response.content) as memory_file:
        with memory_file.open() as source:
            values = source.read(1)
            profile = source.profile
            unique_values = sorted(int(value) for value in set(values.ravel()))

    profile.update(driver="GTiff", compress="lzw")
    temporary_path = f"{FUEL_MODEL_PATH}.tmp.tif"
    with rasterio.open(temporary_path, "w", **profile) as destination:
        destination.write(values, 1)
    os.replace(temporary_path, FUEL_MODEL_PATH)

    print(f"Saved {len(response.content):,} bytes to {FUEL_MODEL_PATH}")
    print(f"Shape={values.shape} CRS={profile['crs']} fuel-model values={unique_values}")
    return values


if __name__ == "__main__":
    fetch_fuels()
