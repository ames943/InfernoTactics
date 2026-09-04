"""Loading and validation for a versioned simulation world."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fingerprint import world_fingerprint


@dataclass(frozen=True)
class WorldBundle:
    """Static raster layers and their geospatial contract.

    Keeping this validation at the simulator boundary prevents a model from
    silently training against a grid whose layer order or dimensions changed.
    """

    grid: np.ndarray
    metadata: dict[str, Any]
    grid_path: Path
    metadata_path: Path
    fingerprint: str

    @classmethod
    def load(
        cls,
        grid_path: str | Path,
        metadata_path: str | Path,
        *,
        expected_layers: Sequence[str] | None = None,
    ) -> "WorldBundle":
        grid_file = Path(grid_path).resolve()
        metadata_file = Path(metadata_path).resolve()
        grid = np.load(grid_file, allow_pickle=False).astype(np.float32, copy=False)
        with metadata_file.open(encoding="utf-8") as source:
            metadata = json.load(source)

        if grid.ndim != 3:
            raise ValueError(f"World grid must have shape (layers, height, width), got {grid.shape}")
        required = {"layer_names", "height", "width", "crs", "transform", "cell_size_m"}
        missing = sorted(required.difference(metadata))
        if missing:
            raise ValueError(f"World metadata is missing required keys: {', '.join(missing)}")
        expected_shape = (len(metadata["layer_names"]), metadata["height"], metadata["width"])
        if grid.shape != expected_shape:
            raise ValueError(f"World grid shape {grid.shape} does not match metadata {expected_shape}")
        if expected_layers is not None and list(metadata["layer_names"]) != list(expected_layers):
            raise ValueError(
                "World layer order does not match the simulator contract: "
                f"{metadata['layer_names']} != {list(expected_layers)}"
            )
        if not np.isfinite(grid).all():
            raise ValueError("World grid contains NaN or infinite values")

        return cls(
            grid=grid,
            metadata=metadata,
            grid_path=grid_file,
            metadata_path=metadata_file,
            fingerprint=world_fingerprint(grid_file, metadata_file),
        )
