"""Build an integrity and provenance manifest for local source datasets."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from infernotactics.data.config import (
    BUILDINGS_PATH,
    ELEVATION_PATH,
    FUEL_MODEL_PATH,
    POPULATION_PATH,
    REAL_DEPOTS_PATH,
    ROADS_GRAPHML_PATH,
    SOURCE_MANIFEST_PATH,
    WEATHER_CSV_PATH,
)
from infernotactics.world import file_sha256


SOURCES = {
    "elevation": {
        "path": ELEVATION_PATH,
        "provider": "USGS 3DEP",
        "product": "3DEP elevation, 10 m",
        "source": "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WMSServer",
    },
    "fuel_model": {
        "path": FUEL_MODEL_PATH,
        "provider": "LANDFIRE / USGS",
        "product": "LF2024 FBFM40 CONUS",
        "source": "https://edcintl.cr.usgs.gov/geoserver/landfire_wcs/conus_2024/wcs",
    },
    "buildings": {
        "path": BUILDINGS_PATH,
        "provider": "LA GeoHub",
        "product": "LARIAC4 Building Footprints",
        "source": "https://services5.arcgis.com/7nsPwEMP38bSkCjy/arcgis/rest/services/Building_Footprints/FeatureServer/0",
    },
    "roads": {
        "path": ROADS_GRAPHML_PATH,
        "provider": "OpenStreetMap",
        "product": "OSMnx drive network",
        "source": "https://www.openstreetmap.org/",
    },
    "population": {
        "path": POPULATION_PATH,
        "provider": "WorldPop",
        "product": "USA 2020 population, 1 km",
        "source": "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/USA/",
    },
    "weather": {
        "path": WEATHER_CSV_PATH,
        "provider": "NOAA ASOS via Iowa Environmental Mesonet",
        "product": "KSMO observations, 2025-01-07 through 2025-01-09 UTC",
        "source": "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
    },
    "stations": {
        "path": REAL_DEPOTS_PATH,
        "provider": "InfernoTactics curated from LAFD sources",
        "product": "Modeled station roster",
        "source": "version-controlled project file",
    },
}


def build_manifest():
    assets = {}
    missing = []
    for name, source in SOURCES.items():
        path = source["path"]
        if not os.path.exists(path):
            missing.append(path)
            continue
        assets[name] = {
            **{key: value for key, value in source.items() if key != "path"},
            "file": Path(os.path.relpath(path, os.path.dirname(SOURCE_MANIFEST_PATH))).as_posix(),
            "bytes": os.path.getsize(path),
            "sha256": file_sha256(path),
        }
    if missing:
        raise FileNotFoundError("Cannot build complete source manifest; missing: " + ", ".join(missing))
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "assets": assets,
    }
    temporary_path = f"{SOURCE_MANIFEST_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    os.replace(temporary_path, SOURCE_MANIFEST_PATH)
    print(f"Wrote {SOURCE_MANIFEST_PATH} with {len(assets)} verified assets")
    return manifest


if __name__ == "__main__":
    build_manifest()
