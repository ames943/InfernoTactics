"""
Builds the unified static spatial grid that the CNN branch consumes.

Rasterizes elevation, slope, buildings, roads, LANDFIRE FBFM40 fuel classes,
and real gridded population density into one aligned NumPy stack
(same shape, same affine transform), all in EPSG:5070 (USA Contiguous Albers
Equal Area, meters) so
that grid cells are approximately uniform physical size across the study
area. Elevation.tif is already in this CRS; buildings/roads (EPSG:4326) are
reprojected into it before rasterizing, which guarantees every layer lines
up on the same pixel grid with no separate resampling-alignment step needed.

Run directly to (re)build data/grid_static.npy and data/grid_meta.json:
    python -m infernotactics.world.builder
"""

import json
import os

import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject

from infernotactics.data.config import (
    BUILDINGS_PATH,
    DATA_DIR,
    ELEVATION_PATH,
    FUEL_MODEL_PATH,
    POPULATION_PATH,
    ROADS_GRAPHML_PATH,
)
from infernotactics.world.layers import STATIC_LAYER_NAMES  # noqa: E402

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

LAYER_NAMES = list(STATIC_LAYER_NAMES)

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


def _rasterize_population_density(transform, width, height, dst_crs):
    """Reproject/resample the WorldPop population-count raster (see
    fetch_population.py -- 2020 gridded population, ~1km cells, EPSG:4326)
    onto this grid's (30m, EPSG:5070) pixel grid, then compress to [0, 1].

    Resampling direction here is UPSAMPLING (1km source -> 30m destination,
    the opposite of elevation's 10m -> 30m downsampling), so bilinear
    (smooth interpolation between neighboring source cells) is the
    appropriate choice, vs. elevation's Resampling.average (appropriate for
    downsampling/aggregating finer source pixels into a coarser destination).

    Normalization: raw population counts are extremely right-skewed (near-
    zero across most of the Topanga hillside, spiking in dense pockets near
    Westwood/UCLA), so a straight min-max scale would crush every
    mid-density neighborhood down near 0 just because one small area is
    ~100x denser. log1p(count) compresses that skew before min-max scaling
    to [0, 1] over this grid's own actual range -- makes the layer usable
    directly as a CNN input channel and as the reward multiplier's
    population_weight (see inferno_env.py).
    """
    with rasterio.open(POPULATION_PATH) as src:
        src_arr = src.read(1).astype(np.float32)
        src_transform = src.transform
        src_crs = src.crs
        nodata = src.nodata

    if nodata is not None:
        src_arr = np.where(src_arr == nodata, 0.0, src_arr)
    src_arr = np.clip(src_arr, 0.0, None)  # floor any stray negative/nodata artifacts to 0

    dst_arr = np.zeros((height, width), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    dst_arr = np.clip(dst_arr, 0.0, None)  # bilinear can ring slightly negative near steep edges

    log_density = np.log1p(dst_arr)
    log_range = max(log_density.max() - log_density.min(), 1e-6)
    return ((log_density - log_density.min()) / log_range).astype(np.float32)


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


# Relative ignitability used by the current cellular automaton. The source
# categories are real FBFM40 classes; these scalar values are an explicit
# model adapter, not a claim that categorical model codes are fuel loads.
_FBFM40_IGNITABILITY = {
    91: 0.00, 92: 0.00, 93: 0.30, 98: 0.00, 99: 0.05,
    101: 0.35, 102: 0.55, 103: 0.65, 104: 0.75, 105: 0.55,
    106: 0.70, 107: 0.85, 108: 0.80, 109: 1.00,
    121: 0.55, 122: 0.70, 123: 0.82, 124: 0.95,
    141: 0.35, 142: 0.45, 143: 0.50, 144: 0.65, 145: 0.55,
    146: 0.68, 147: 0.80, 148: 0.88, 149: 1.00,
    161: 0.40, 162: 0.50, 163: 0.65, 164: 0.75, 165: 0.85,
    181: 0.25, 182: 0.35, 183: 0.45, 184: 0.50, 185: 0.55,
    186: 0.62, 187: 0.70, 188: 0.78, 189: 0.85,
    201: 0.50, 202: 0.65, 203: 0.78, 204: 0.90,
}


def _rasterize_fuel_density(transform, width, height, dst_crs):
    """Align categorical LANDFIRE FBFM40 and map it to CA ignitability."""
    if not os.path.exists(FUEL_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {FUEL_MODEL_PATH}. Run `python -m infernotactics.data.fetch_fuels` "
            "before rebuilding the world. Set INFERNO_ALLOW_HEURISTIC_FUEL=1 "
            "only when intentionally reproducing the legacy synthetic layer."
        )
    with rasterio.open(FUEL_MODEL_PATH) as source:
        source_values = source.read(1)
        destination_values = np.zeros((height, width), dtype=np.int16)
        reproject(
            source=source_values,
            destination=destination_values,
            src_transform=source.transform,
            src_crs=source.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    ignitability = np.zeros((height, width), dtype=np.float32)
    unknown = set(int(value) for value in np.unique(destination_values))
    unknown.difference_update(_FBFM40_IGNITABILITY)
    unknown.discard(0)
    if unknown:
        raise ValueError(f"Unmapped LANDFIRE FBFM40 values: {sorted(unknown)}")
    for fuel_model, value in _FBFM40_IGNITABILITY.items():
        ignitability[destination_values == fuel_model] = value
    return ignitability


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

    if os.environ.get("INFERNO_ALLOW_HEURISTIC_FUEL") == "1":
        print("WARNING: using legacy heuristic fuel layer by explicit request")
        fuel_density = _placeholder_fuel_density(
            building_density, road_mask, elevation, water_mask
        )
        fuel_source = "legacy_heuristic"
    else:
        print("Loading + aligning LANDFIRE 2024 FBFM40 fuel models...")
        fuel_density = _rasterize_fuel_density(transform, width, height, crs)
        fuel_density = np.where(road_mask == 1, fuel_density * 0.20, fuel_density)
        fuel_density = np.where(water_mask, 0.0, fuel_density).astype(np.float32)
        fuel_source = "LANDFIRE LF2024 FBFM40 via official WCS"

    print("Loading + reprojecting population...")
    population_density = _rasterize_population_density(transform, width, height, crs)

    stacked = np.stack(
        [
            elevation,
            slope,
            building_density,
            building_height,
            road_mask.astype(np.float32),
            fuel_density,
            water_mask.astype(np.float32),
            population_density,
        ],
        axis=0,
    )

    os.makedirs(DATA_DIR, exist_ok=True)
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
        "fuel_source": fuel_source,
    }
    grid_temp = f"{GRID_STATIC_PATH}.tmp.npy"
    meta_temp = f"{GRID_META_PATH}.tmp"
    np.save(grid_temp, stacked)
    with open(meta_temp, "w", encoding="utf-8") as output:
        json.dump(meta, output, indent=2)
        output.write("\n")
    os.replace(grid_temp, GRID_STATIC_PATH)
    os.replace(meta_temp, GRID_META_PATH)

    print(f"Saved static grid stack -> {GRID_STATIC_PATH}  shape={stacked.shape}")
    print(f"Saved grid metadata -> {GRID_META_PATH}")

    print("\n--- Layer stats ---")
    for name, layer in zip(LAYER_NAMES, stacked):
        print(f"{name:18s} min={layer.min():8.3f}  max={layer.max():8.3f}  mean={layer.mean():8.3f}")

    return stacked, meta


if __name__ == "__main__":
    build_grid()
