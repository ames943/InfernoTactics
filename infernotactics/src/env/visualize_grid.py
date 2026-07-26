"""
Quick visual sanity check for data/grid_static.npy: plots each static layer
as a subplot so misalignment/flipping between layers is easy to spot by eye
(e.g. roads should trace through gaps in dense building clusters, both
should sit at lower elevation than the surrounding hillsides).

    python -m src.env.visualize_grid
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR  # noqa: E402
from env.grid_builder import GRID_META_PATH, GRID_STATIC_PATH, LAYER_NAMES  # noqa: E402

OUTPUT_PATH = os.path.join(DATA_DIR, "grid_visualization.png")

# (cmap, label) per layer, in the same order as LAYER_NAMES
LAYER_STYLE = {
    "elevation": ("terrain", "Elevation (m)"),
    "slope": ("YlOrRd", "Slope (deg)"),
    "building_density": ("Greys", "Building density (0-1)"),
    "building_height": ("plasma", "Building height (m)"),
    "road_mask": ("Greys", "Road mask"),
    "fuel_density": ("YlGn", "Fuel density (placeholder, 0-1)"),
    "water_mask": ("Blues", "Water mask (heuristic, 0-1)"),
}


def visualize():
    stacked = np.load(GRID_STATIC_PATH)
    with open(GRID_META_PATH) as f:
        meta = json.load(f)

    assert list(stacked.shape[1:]) == [meta["height"], meta["width"]]
    assert stacked.shape[0] == len(LAYER_NAMES)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.flatten()

    for ax, name, layer in zip(axes, LAYER_NAMES, stacked):
        cmap, label = LAYER_STYLE[name]
        # origin="upper": row 0 is the northernmost row, matching the affine
        # transform (which has a negative y-step) -- so this must match the
        # raster's real north-up orientation, not be flipped.
        im = ax.imshow(layer, cmap=cmap, origin="upper")
        ax.set_title(label, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(LAYER_NAMES):]:
        ax.axis("off")

    fig.suptitle(
        f"InfernoTactics static grid — {meta['width']}x{meta['height']} cells "
        f"@ {meta['cell_size_m']}m ({meta['crs']})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved visualization -> {OUTPUT_PATH}")

    # Composite overlay: buildings + roads on top of a terrain hillshade-ish
    # backdrop, as an extra alignment check (do roads/buildings sit in the
    # low-lying developed areas, not scattered over open hillside?).
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    elevation = stacked[LAYER_NAMES.index("elevation")]
    building_density = stacked[LAYER_NAMES.index("building_density")]
    road_mask = stacked[LAYER_NAMES.index("road_mask")]

    ax2.imshow(elevation, cmap="terrain", origin="upper", alpha=0.9)
    building_overlay = np.ma.masked_where(building_density <= 0, building_density)
    ax2.imshow(building_overlay, cmap="Reds", origin="upper", alpha=0.8, vmin=0, vmax=1)
    road_overlay = np.ma.masked_where(road_mask <= 0, road_mask)
    ax2.imshow(road_overlay, cmap="Greys_r", origin="upper", alpha=0.6, vmin=0, vmax=1)
    ax2.set_title("Overlay check: buildings (red) + roads (white) on elevation")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig2.tight_layout()
    overlay_path = os.path.join(DATA_DIR, "grid_overlay_check.png")
    fig2.savefig(overlay_path, dpi=150)
    print(f"Saved overlay check -> {overlay_path}")


if __name__ == "__main__":
    visualize()
