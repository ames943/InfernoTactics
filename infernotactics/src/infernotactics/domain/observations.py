"""Stable observation schema shared by environment and policy code."""

from collections.abc import Mapping

import numpy as np


SCALAR_KEYS = (
    "wind_speed_mph",
    "wind_direction_deg",
    "humidity_pct",
    "water_team_available",
    "trench_crew_available",
    "rescue_vehicle_available",
    "helicopter_available",
    "time_elapsed_ticks",
    "traffic_mean_load",
    "traffic_max_load",
    "active_ground_resources",
)


def flatten_scalars(scalars: Mapping[str, float]) -> np.ndarray:
    """Return an observation's scalar values in the model contract order."""
    missing = [key for key in SCALAR_KEYS if key not in scalars]
    if missing:
        raise ValueError(f"Observation scalars are missing: {', '.join(missing)}")
    return np.asarray([scalars[key] for key in SCALAR_KEYS], dtype=np.float32)
