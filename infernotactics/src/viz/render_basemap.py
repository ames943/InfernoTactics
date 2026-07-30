"""Render the static grid layers to a basemap image for the live player.

The player draws fire, resources and effects on top of this image, so it must
be a faithful picture of the terrain the simulation actually runs on: the real
USGS elevation, the real OSM road mask, the real LARIAC building footprints and
the coastline water mask, all straight out of ``grid_static.npy``.  Nothing here
is decorative -- every pixel comes from a layer the fire model reads.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import LAYER_INDEX  # noqa: E402


UPSCALE = 3  # basemap pixels per grid cell; fire is drawn at cell resolution on top
DISPLAY_GAMMA_LIFT = 0.88  # midtone lift so the texture survives linear-space lighting


def _normalize(values, low=None, high=None):
    low = float(np.min(values)) if low is None else low
    high = float(np.max(values)) if high is None else high
    if high - low < 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _hillshade(elevation, cell_size_m, azimuth_deg=315.0, altitude_deg=42.0, z_factor=2.4):
    """Standard Horn-gradient hillshade, light from the north-west."""
    dz_dy, dz_dx = np.gradient(elevation.astype(np.float32) * z_factor, cell_size_m)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    azimuth = np.radians(360.0 - azimuth_deg + 90.0)
    altitude = np.radians(altitude_deg)
    shaded = (np.sin(altitude) * np.cos(slope)
              + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect))
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)


def _lerp(color_a, color_b, weight):
    weight = weight[..., None]
    return color_a * (1.0 - weight) + color_b * weight


def render(grid_static, meta, upscale=UPSCALE):
    """Return a PIL RGB image of the static terrain at ``upscale`` px per cell."""
    height, width = meta["height"], meta["width"]
    cell_size_m = float(meta["cell_size_m"])

    elevation = grid_static[LAYER_INDEX["elevation"]]
    water = grid_static[LAYER_INDEX["water_mask"]] > 0.5
    road = grid_static[LAYER_INDEX["road_mask"]] > 0.05
    fuel = grid_static[LAYER_INDEX["fuel_density"]]
    buildings = grid_static[LAYER_INDEX["building_density"]]
    population = grid_static[LAYER_INDEX["population_density"]]

    # --- Terrain base: fuel-tinted ground, elevation-banded, hillshaded -------
    dry_low = np.array([0.196, 0.169, 0.114], dtype=np.float32)   # dry canyon floor
    dry_high = np.array([0.353, 0.294, 0.196], dtype=np.float32)  # exposed ridge
    chaparral = np.array([0.133, 0.196, 0.098], dtype=np.float32)  # vegetated fuel
    rock = np.array([0.400, 0.353, 0.298], dtype=np.float32)       # bare steep face

    elevation_norm = _normalize(np.where(water, np.nan, elevation))
    elevation_norm = np.nan_to_num(elevation_norm, nan=0.0)
    ground = _lerp(dry_low, dry_high, elevation_norm ** 0.75)
    ground = _lerp(ground, chaparral, np.clip(fuel, 0.0, 1.0) * 0.80)

    # Steep faces shed soil and hold less brush, so let rock show through on
    # them. Without this the whole basin renders as one flat olive tone, because
    # fuel_density is a near-uniform placeholder layer.
    slope = grid_static[LAYER_INDEX["slope"]]
    steepness = _normalize(slope, low=float(np.percentile(slope, 55)),
                           high=float(np.percentile(slope, 99)))
    ground = _lerp(ground, rock, steepness * 0.20)

    # Deliberately gentle: the 3D view lights this texture again with a real
    # directional light, so baking full-contrast hillshade in would double-shade
    # every slope. Enough is kept to carry drainage detail below mesh resolution.
    shade = _hillshade(elevation, cell_size_m)
    ground *= (0.62 + 0.52 * shade)[..., None]

    # --- Developed area: population glow, then building footprints -----------
    # Buildings are deliberately warm and roads cool, so the two read apart in
    # dense Westwood/Brentwood where footprints and streets sit on top of each
    # other -- with one gray for both, the built-up south-east turns to mush.
    urban = np.array([0.243, 0.239, 0.271], dtype=np.float32)
    ground = _lerp(ground, urban, np.clip(population, 0.0, 1.0) * 0.34)

    building_strength = np.clip(buildings / 0.45, 0.0, 1.0)
    built = np.array([0.600, 0.522, 0.408], dtype=np.float32)
    ground = _lerp(ground, built, building_strength * 0.86)

    # --- Water: flat, cool, unshaded -----------------------------------------
    ocean = np.array([0.043, 0.114, 0.196], dtype=np.float32)
    ground = np.where(water[..., None], ocean, ground)

    # The 3D view samples this as an sRGB texture, so the GPU decodes it to
    # linear before lighting it. Values that look right in a flat 2D map
    # (median byte ~43, i.e. 0.024 linear) come out almost black once lit, so
    # lift the midtones here. Shadow detail survives; highlights are preserved.
    image = np.clip(ground, 0.0, 1.0) ** DISPLAY_GAMMA_LIFT
    rgb = (image * 255.0).astype(np.uint8)

    # Smooth the terrain on upscale so hillshade reads as landform, not pixels.
    big = Image.fromarray(rgb, mode="RGB").resize(
        (width * upscale, height * upscale), Image.BICUBIC
    )

    # --- Roads drawn after the smooth resize so they stay thin and crisp -----
    road_big = np.array(
        Image.fromarray((road * 255).astype(np.uint8), mode="L").resize(
            (width * upscale, height * upscale), Image.NEAREST
        )
    ) > 127
    canvas = np.array(big, dtype=np.float32) / 255.0
    road_color = np.array([0.702, 0.729, 0.769], dtype=np.float32)
    canvas[road_big] = canvas[road_big] * 0.38 + road_color * 0.62

    out = (np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def main():
    import argparse
    import json

    from env.inferno_env import GRID_META_PATH, GRID_STATIC_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--upscale", type=int, default=UPSCALE)
    args = parser.parse_args()

    grid_static = np.load(GRID_STATIC_PATH).astype(np.float32)
    with open(GRID_META_PATH) as handle:
        meta = json.load(handle)
    image = render(grid_static, meta, upscale=args.upscale)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "basemap.png")
    image.save(out, optimize=True)
    print(f"wrote {out} ({image.width}x{image.height}, {os.path.getsize(out) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
