"""
Builds the unified static spatial grid that the CNN branch consumes.

Rasterizes elevation, slope, buildings, roads, and a placeholder fuel layer
into a single stack of aligned numpy arrays (same shape, same affine
transform), all in EPSG:5070 (USA Contiguous Albers Equal Area, meters) so
that grid cells are approximately uniform physical size across the study
area. Elevation.tif is already in this CRS; buildings/roads (EPSG:4326) are
reprojected into it before rasterizing, which guarantees every layer lines
up on the same pixel grid with no separate resampling-alignment step needed.

Run directly to (re)build data/grid_static.npy and data/grid_meta.json:
    python -m src.env.grid_builder
"""

import json
import os
import sys

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import (  # noqa: E402
    BUILDINGS_PATH,
    DATA_DIR,
    ELEVATION_PATH,
    ROADS_GRAPHML_PATH,
)

CELL_SIZE_M = 30.0

# Building coverage is estimated by rasterizing at a finer sub-resolution and
# average-pooling up to the target cell size (a cheap Monte-Carlo-style area
# estimate), rather than doing an exact polygon/cell intersection-area calc.
BUILDING_SUBSAMPLE_FACTOR = 6  # 30m / 6 = 5m sub-pixels

# Hard cap on per-building HEIGHT before rasterizing. The LARIAC4 source data
# has a handful of LiDAR outliers (e.g. one building at ~220m, implausible for
# this area) -- almost certainly noise (crane, antenna, bad return) rather
# than a real structure. 150m comfortably covers the tallest legitimate
# high-rises near Westwood/Wilshire while dropping those spikes.
BUILDING_HEIGHT_CAP_M = 150.0

GRID_STATIC_PATH = os.path.join(DATA_DIR, "grid_static.npy")
GRID_META_PATH = os.path.join(DATA_DIR, "grid_meta.json")

LAYER_NAMES = [
    "elevation",
    "slope",
    "building_density",
    "building_height",
    "road_mask",
    "fuel_density",
    "water_mask",
]

# HEURISTIC coastline threshold, not real hydrology/coastline vector data. The
# DEM reports the Pacific as ~0m elevation, and within this bbox the *only*
# place elevation dips this low is the actual coastline strip along Santa
# Monica Canyon / PCH at the study area's SW corner (verified against the
# built grid: cells below this threshold form a single contiguous region in
# the bottom-left corner, rows>=285/cols<=142 out of 316x595 -- no inland
# false positives from canyon bottoms or flats elsewhere in this bbox). If the
# study area is ever expanded to include a real inland low point (e.g. a
# reservoir), this heuristic should be replaced with an actual water polygon
# layer (e.g. OSM natural=water / NHD).
WATER_ELEVATION_THRESHOLD_M = 2.0


def _target_grid_geometry(src_bounds, cell_size):
    left, bottom, right, top = src_bounds
    width = int(round((right - left) / cell_size))
    height = int(round((top - bottom) / cell_size))
    transform = Affine(cell_size, 0.0, left, 0.0, -cell_size, top)
    return width, height, transform


def _resample_elevation(cell_size):
    with rasterio.open(ELEVATION_PATH) as src:
        src_arr = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        bounds = src.bounds

    width, height, dst_transform = _target_grid_geometry(bounds, cell_size)
    dst_arr = np.zeros((height, width), dtype=np.float32)

    # Downsampling 10m -> 30m: average avoids aliasing that nearest/bilinear
    # point-sampling would introduce.
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=src_crs,
        resampling=Resampling.average,
    )
    return dst_arr, dst_transform, width, height, src_crs


def _compute_slope_degrees(elevation, cell_size):
    dz_dy, dz_dx = np.gradient(elevation, cell_size)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    return np.degrees(slope_rad).astype(np.float32)


def _rasterize_building_density(buildings, transform, width, height, subsample):
    fine_transform = transform * Affine.scale(1.0 / subsample)
    fine_shape = (height * subsample, width * subsample)
    geoms = [
        (geom, 1)
        for geom in buildings.geometry
        if geom is not None and not geom.is_empty
    ]
    fine_mask = rasterize(
        geoms,
        out_shape=fine_shape,
        transform=fine_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )
    density = fine_mask.reshape(height, subsample, width, subsample).mean(axis=(1, 3))
    return density.astype(np.float32)


def _rasterize_building_height(buildings, transform, width, height):
    # rasterize() burns shapes in list order, with later shapes overwriting
    # earlier ones per-pixel (MergeAlg.replace, the default) -- so sorting
    # ascending by height means the tallest building touching a pixel is
    # whatever gets burned last, giving an approximate per-cell max height.
    ordered = buildings.sort_values("HEIGHT", ascending=True)
    capped_heights = ordered["HEIGHT"].clip(upper=BUILDING_HEIGHT_CAP_M)
    shapes = [
        (geom, float(h))
        for geom, h in zip(ordered.geometry, capped_heights)
        if geom is not None and not geom.is_empty
    ]
    height_grid = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype=np.float32,
        all_touched=False,
    )
    return height_grid


def _rasterize_roads(edges, transform, width, height):
    shapes = [
        (geom, 1) for geom in edges.geometry if geom is not None and not geom.is_empty
    ]
    road_mask = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,  # a road only 1px wide must still mark any cell it crosses
    )
    return road_mask


def _heuristic_water_mask(elevation, threshold_m=WATER_ELEVATION_THRESHOLD_M):
    """Coastline mask via a low-elevation heuristic -- see the module-level
    comment on WATER_ELEVATION_THRESHOLD_M for why this is safe in this bbox
    and when it would need to be replaced with real coastline data."""
    return elevation <= threshold_m


def _placeholder_fuel_density(building_density, road_mask, elevation, water_mask):
    # PLACEHOLDER FUEL MODEL. We don't have real LANDFIRE fuel data
    # (FBFM40 fuel model / fuel load rasters) yet. Until that's integrated,
    # approximate fuel density heuristically: vegetation fills whatever isn't
    # built on or paved, tapered down slightly at higher elevation (sparser,
    # rockier chaparral on the upper slopes vs. denser growth in canyons).
    # Roads act as fuel breaks. Replace this function once LANDFIRE data is
    # available -- do not treat these values as real fuel-load estimates.
    elev_range = max(elevation.max() - elevation.min(), 1e-6)
    elev_norm = (elevation - elevation.min()) / elev_range
    fuel = 1.0 - building_density
    fuel *= (1.0 - 0.3 * elev_norm)
    fuel = np.where(road_mask == 1, 0.05, fuel)
    fuel = np.where(water_mask, 0.0, fuel)  # water never carries fuel
    return np.clip(fuel, 0.0, 1.0).astype(np.float32)


def build_grid(cell_size_m=CELL_SIZE_M):
    print(f"Building simulation grid at {cell_size_m}m resolution...")

    elevation, transform, width, height, crs = _resample_elevation(cell_size_m)
    print(f"Grid dimensions: {width} x {height} cells ({width * height:,} cells total)")
    print(f"Working CRS: {crs}")

    slope = _compute_slope_degrees(elevation, cell_size_m)

    print("Loading + reprojecting buildings...")
    buildings = gpd.read_file(BUILDINGS_PATH).to_crs(crs)
    building_density = _rasterize_building_density(
        buildings, transform, width, height, BUILDING_SUBSAMPLE_FACTOR
    )
    building_height = _rasterize_building_height(buildings, transform, width, height)

    print("Loading + reprojecting roads...")
    graph = ox.load_graphml(ROADS_GRAPHML_PATH)
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True).to_crs(crs)
    road_mask = _rasterize_roads(edges, transform, width, height)

    water_mask = _heuristic_water_mask(elevation)
    print(f"Water/ocean cells (heuristic, elevation <= {WATER_ELEVATION_THRESHOLD_M}m): "
          f"{int(water_mask.sum()):,} of {water_mask.size:,}")

    # Zero out building layers over water too -- guards against stray
    # rasterized artifacts (e.g. a pier polygon) making a water cell look
    # ignitable via the building_density term in FireSim.ignitability.
    building_density = np.where(water_mask, 0.0, building_density).astype(np.float32)
    building_height = np.where(water_mask, 0.0, building_height).astype(np.float32)

    fuel_density = _placeholder_fuel_density(building_density, road_mask, elevation, water_mask)

    stacked = np.stack(
        [
            elevation,
            slope,
            building_density,
            building_height,
            road_mask.astype(np.float32),
            fuel_density,
            water_mask.astype(np.float32),
        ],
        axis=0,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(GRID_STATIC_PATH, stacked)

    meta = {
        "layer_names": LAYER_NAMES,
        "cell_size_m": cell_size_m,
        "width": width,
        "height": height,
        "crs": str(crs),
        # affine transform coefficients (a, b, c, d, e, f) mapping
        # (col, row) -> (x, y) in `crs`; use with pyproj to recover lon/lat.
        "transform": list(transform)[:6],
        "bounds": list(array_bounds(height, width, transform)),  # left, bottom, right, top
    }
    with open(GRID_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved static grid stack -> {GRID_STATIC_PATH}  shape={stacked.shape}")
    print(f"Saved grid metadata -> {GRID_META_PATH}")

    print("\n--- Layer stats ---")
    for name, layer in zip(LAYER_NAMES, stacked):
        print(f"{name:18s} min={layer.min():8.3f}  max={layer.max():8.3f}  mean={layer.mean():8.3f}")

    return stacked, meta


if __name__ == "__main__":
    build_grid()
