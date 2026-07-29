"""Assemble the self-contained 3D simulation viewer.

Produces ONE html file with no external requests: the terrain mesh, its texture,
the vendored three.js build and the model's exported trajectories are all inlined.
That matters because the viewer is opened straight off disk (``file://``), where
``fetch()`` of sibling files is blocked -- and because a demo should not depend on
a CDN, an API token or a network connection.

Assets built here
-----------------
basemap  JPEG of the real static layers (see render_basemap.py).
elevation  The real USGS 3DEP elevation grid, encoded losslessly as an RGB PNG:
    decimetres packed as R = high byte, G = low byte.  A 16-bit PNG would be
    truncated to 8 bits when read back through a canvas, and raw float32 base64
    is ~4x larger, so byte-splitting is both exact and compact.
trajectories  The deterministic rollouts from export_trajectory.py.
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import GRID_META_PATH, GRID_STATIC_PATH, LAYER_INDEX  # noqa: E402
from viz import render_basemap  # noqa: E402


VIZ_DIR = os.path.dirname(os.path.abspath(__file__))
ELEVATION_SCALE = 10.0  # decimetre precision, well inside uint16 for a 686 m range


def encode_elevation_png(elevation):
    """Pack elevation into an RGB PNG as exact 16-bit decimetres."""
    scaled = np.clip(np.round(elevation * ELEVATION_SCALE), 0, 65535).astype(np.uint16)
    rgb = np.zeros(scaled.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = (scaled >> 8).astype(np.uint8)
    rgb[..., 1] = (scaled & 0xFF).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def encode_basemap_jpeg(grid_static, meta, upscale, quality):
    image = render_basemap.render(grid_static, meta, upscale=upscale)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=1, optimize=True)
    return buffer.getvalue(), image.size


def data_uri(payload, mime):
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default=os.path.join(VIZ_DIR, "trajectories.json"))
    parser.add_argument("--template", default=os.path.join(VIZ_DIR, "player_template.html"))
    parser.add_argument("--three", default=os.path.join(VIZ_DIR, "vendor", "three.module.min.js"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(PROJECT_ROOT),
                                                     "Simulation3D.html"))
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    for path in (args.trajectories, args.template, args.three):
        if not os.path.exists(path):
            raise SystemExit(f"missing required input: {path}")

    grid_static = np.load(GRID_STATIC_PATH).astype(np.float32)
    with open(GRID_META_PATH) as handle:
        meta = json.load(handle)

    elevation = grid_static[LAYER_INDEX["elevation"]]
    elevation_png = encode_elevation_png(elevation)
    basemap_jpeg, basemap_size = encode_basemap_jpeg(grid_static, meta, args.upscale, args.quality)

    with open(args.trajectories) as handle:
        trajectories = json.load(handle)
    with open(args.three, "rb") as handle:
        three_source = handle.read()
    with open(args.template) as handle:
        template = handle.read()

    build_meta = {
        "grid": {"width": meta["width"], "height": meta["height"],
                 "cell_size_m": float(meta["cell_size_m"])},
        "elevation": {"scale": ELEVATION_SCALE,
                      "min_m": float(np.min(elevation)), "max_m": float(np.max(elevation))},
        "basemap": {"width": basemap_size[0], "height": basemap_size[1]},
        "crs": meta["crs"],
    }

    replacements = {
        "__THREE_SOURCE_B64__": base64.b64encode(three_source).decode("ascii"),
        "__BASEMAP_URI__": data_uri(basemap_jpeg, "image/jpeg"),
        "__ELEVATION_URI__": data_uri(elevation_png, "image/png"),
        "__TRAJECTORIES_JSON__": json.dumps(trajectories, separators=(",", ":")),
        "__BUILD_META_JSON__": json.dumps(build_meta, separators=(",", ":")),
    }
    for token, value in replacements.items():
        if token not in template:
            raise SystemExit(f"template is missing placeholder {token}")
        template = template.replace(token, value)

    with open(args.out, "w") as handle:
        handle.write(template)

    print(f"[build] basemap  {basemap_size[0]}x{basemap_size[1]} jpeg "
          f"{len(basemap_jpeg) / 1024:.0f} KiB")
    print(f"[build] elevation png {len(elevation_png) / 1024:.0f} KiB "
          f"({build_meta['elevation']['min_m']:.0f}-{build_meta['elevation']['max_m']:.0f} m)")
    print(f"[build] three.js {len(three_source) / 1024:.0f} KiB")
    print(f"[build] trajectories {os.path.getsize(args.trajectories) / 1024:.0f} KiB "
          f"({len(trajectories['episodes'])} episodes)")
    print(f"[build] wrote {args.out} ({os.path.getsize(args.out) / 1024 / 1024:.2f} MiB, "
          f"self-contained)")


if __name__ == "__main__":
    main()
