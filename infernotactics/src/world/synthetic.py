"""Deterministic, explicitly illustrative inputs for offline world verification."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

import numpy as np
from pyproj import Transformer

from .contracts import Scenario


def make_synthetic(
    seed: int = 0, duration_s: int = 21600, size: int = 96,
    cell_size_m: float = 30.0,
) -> Scenario:
    """Create a flat grass landscape; these are not measured incident inputs.

    The grid is centered inside the Palisades study box. Meteorological bands
    have explicit hourly validity covering the horizon. A two-way
    directed road loop and all three illustrative units share one home base.
    """
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(duration_s, bool) or not isinstance(duration_s, int) or duration_s <= 0:
        raise ValueError("duration_s must be a positive integer")
    if isinstance(size, bool) or not isinstance(size, int) or size < 5:
        raise ValueError("size must be an integer of at least 5 cells")
    if isinstance(cell_size_m, bool) or not math.isfinite(cell_size_m) or cell_size_m <= 0:
        raise ValueError("cell_size_m must be finite and positive")
    seed = int(seed)
    cell_size_m = float(cell_size_m)
    longitude, latitude = -118.5225, 34.09
    cx, cy = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True).transform(longitude, latitude)
    half = size * cell_size_m / 2
    transform = (cell_size_m, 0.0, cx - half, 0.0, -cell_size_m, cy + half)
    shape = (size, size)
    landscape = {
        "elevation": np.full(shape, 200.0),
        "slope": np.zeros(shape),
        "aspect": np.zeros(shape),
        "fuel_model": np.full(shape, 101, dtype=np.int16),
        "canopy_cover": np.zeros(shape),
        "canopy_height": np.zeros(shape),
        "canopy_base_height": np.zeros(shape),
        "canopy_bulk_density": np.zeros(shape),
        "valid": np.ones(shape, dtype=bool),
        "asset_value": np.zeros(shape),
    }
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    iso = lambda value: value.isoformat().replace("+00:00", "Z")
    bands = math.ceil(duration_s / 3600)
    weather_values = {
        "wind_speed_m_s": 0.0, "upwind_direction_deg": 270.0,
        "temperature_c": 25.0, "fuel_moisture_dead_1hr": 0.06,
        "fuel_moisture_dead_10hr": 0.08, "fuel_moisture_dead_100hr": 0.10,
        "fuel_moisture_live_herbaceous": 0.60,
        "fuel_moisture_live_woody": 0.90, "foliar_moisture": 1.0,
    }
    weather = {key: np.full((bands, 1, 1), value) for key, value in weather_values.items()}
    # Put roads through cell centers, with a margin from the domain edge.
    offset = (size // 2 - 1) * cell_size_m
    points = [(cx - offset, cy - offset), (cx + offset, cy - offset),
              (cx + offset, cy + offset), (cx - offset, cy + offset)]
    nodes = [{"id": str(i), "x": x, "y": y} for i, (x, y) in enumerate(points)]
    edges = []
    for i in range(4):
        j = (i + 1) % 4
        for source, target in ((i, j), (j, i)):
            a, b = points[source], points[target]
            edges.append({"id": f"{source}-{target}", "source": str(source),
                          "target": str(target), "length_m": math.dist(a, b),
                          "speed_m_s": 10.0, "geometry": [list(a), list(b)],
                          "closed": False, "speed_source": "illustrative_not_measured"})
    resources = []
    for kind, capacity, speed in (("engine", 2000.0, 10.0),
                                  ("handcrew", 0.0, 10.0),
                                  ("helicopter", 1000.0, 40.0)):
        x, y = points[0]
        unit = {
            "id": f"synthetic-{kind}", "kind": kind, "base_id": "synthetic-base",
            "home_x": x, "home_y": y, "x": x, "y": y,
            "status": "available", "assignment": None, "route": [],
            "route_progress_m": 0.0, "supply_l": capacity, "capacity_l": capacity,
            "mobilization_s": 30.0, "speed_m_s": speed,
            "offroad_speed_m_s": 1.4 if kind == "handcrew" else 2.0,
            "max_access_m": half * 2 if kind != "engine" else 100.0,
            "work_duration_s": 60.0, "water_rate_l_s": 10.0 if capacity else 0.0,
            "line_rate_m_s": 0.2 if kind == "handcrew" else 0.0,
            "refill_rate_l_s": 20.0 if capacity else 0.0,
            "max_wind_m_s": 15.0, "shift_end_s": float(duration_s),
            "max_slope": 1.0, "min_ground_speed_m_s": 5.0,
            "air_corridor_available": True,
        }
        unit["parameter_sources"] = {key: "synthetic illustrative assumption; not operational data"
                                     for key in unit}
        resources.append(unit)
    config = {"seed": seed, "duration_s": duration_s, "size": size, "cell_size_m": cell_size_m}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    provenance = {
        "scenario_kind": "synthetic", "observation_mode": "oracle_current",
        "historical_validation": "unvalidated", "suppression_validation": "experimental",
        "source_hashes": {"synthetic_configuration": digest},
        "sources": {key: "illustrative synthetic constant, not measured" for key in
                    (*landscape, *weather)},
        "weather_intervals": [
            {"start_utc": iso(start + timedelta(seconds=i * 3600)),
             "end_utc": iso(start + timedelta(seconds=(i + 1) * 3600))}
            for i in range(bands)
        ],
        "ignition": {"longitude": longitude, "latitude": latitude},
        "moisture_source": "synthetic_illustrative",
        "suppression": {
            "k": 1.0, "decay_s": 600.0, "radius_m": 30.0,
            "application_rate_l_s": 10.0,
            "source": "illustrative experimental parameters; not calibrated",
        },
        "spotting": {"enabled": False, "source": "numerical fixture configuration"},
        "walking_speed_relation": "approximation; not validated for operational use",
        "base_service_slots": {"synthetic-base": 1},
    }
    scenario = Scenario(
        name="synthetic-flat-grass", crs="EPSG:5070", transform=transform,
        start_utc=iso(start), duration_s=duration_s, landscape=landscape,
        weather=weather, weather_step_s=3600, roads={"nodes": nodes, "edges": edges},
        resources=resources, provenance=provenance, seed=seed,
    )
    scenario.validate()
    return scenario
