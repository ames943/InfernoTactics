"""Fire-relative semantic target candidates for containment policies."""

import math

import numpy as np

from infernotactics.domain.fire import BLAZE, FUEL, THREAT
from infernotactics.domain.resources import RESOURCE_TYPES
from infernotactics.domain.weather import meteorological_wind_to_grid
from infernotactics.world.layers import LAYER_INDEX


TARGET_TYPES = (
    "active_fire",
    "downwind_fire_front",
    "adjacent_fuel",
    "threatened_population",
    "nearest_reachable_fire",
    "noop",
)
N_TARGET_TYPES = len(TARGET_TYPES)
NOOP_TARGET_INDEX = TARGET_TYPES.index("noop")
TARGET_FEATURE_DIM = 10


def _active_mask(fire_state):
    return np.isin(fire_state, (THREAT, BLAZE))


def _eight_connected_dilation(mask):
    result = np.zeros_like(mask, dtype=bool)
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            row_start = max(0, row_delta)
            row_stop = min(mask.shape[0], mask.shape[0] + row_delta)
            col_start = max(0, col_delta)
            col_stop = min(mask.shape[1], mask.shape[1] + col_delta)
            result[row_start:row_stop, col_start:col_stop] |= mask[
                row_start - row_delta:row_stop - row_delta,
                col_start - col_delta:col_stop - col_delta,
            ]
    return result


def _downwind_score(rows, cols, fire_state, wind_direction_deg):
    """Score candidate fuel cells by downwind alignment and distance."""
    active = np.argwhere(_active_mask(fire_state))
    if not len(active) or not len(rows):
        return np.zeros(len(rows), dtype=np.float32)
    wind = np.array(meteorological_wind_to_grid(wind_direction_deg), dtype=np.float32)
    points = np.column_stack((rows, cols)).astype(np.float32)
    best = np.zeros(len(points), dtype=np.float32)
    for fire_row, fire_col in active[:: max(1, len(active) // 256)]:
        delta = points - np.array([fire_row, fire_col], dtype=np.float32)
        distance = np.linalg.norm(delta, axis=1) + 1e-6
        alignment = (delta @ wind) / distance
        best = np.maximum(best, np.maximum(alignment, 0.0) / distance)
    return best


def _zone_stats(env, fire_state):
    active = _active_mask(fire_state)
    adjacent_fuel = _eight_connected_dilation(active) & (fire_state == FUEL)
    building = (
        env.grid_static[LAYER_INDEX["building_density"]]
        > env.building_presence_threshold
    )
    population = env.grid_static[LAYER_INDEX["population_density"]]
    threatened = building & active
    stats = []
    for zone in env.zones:
        row_start, row_stop = zone["row_range"]
        col_start, col_stop = zone["col_range"]
        state = fire_state[row_start:row_stop, col_start:col_stop]
        active_zone = active[row_start:row_stop, col_start:col_stop]
        fuel_zone = adjacent_fuel[row_start:row_stop, col_start:col_stop]
        threat_zone = threatened[row_start:row_stop, col_start:col_stop]
        pop_zone = population[row_start:row_stop, col_start:col_stop]
        stats.append({
            "active": int(active_zone.sum()),
            "adjacent_fuel": int(fuel_zone.sum()),
            "threatened_population": float(pop_zone[threat_zone].max()) if threat_zone.any() else 0.0,
            "fuel": float((state == FUEL).mean()),
            "building": float(building[row_start:row_stop, col_start:col_stop].mean()),
            "row": float(zone["centroid_row"]),
            "col": float(zone["centroid_col"]),
        })
    return stats, adjacent_fuel


def resolve_relative_targets(env, obs):
    """Return `(resource, semantic target)` zone IDs and feature vectors.

    Zone IDs have shape ``(resources, targets)`` and features have shape
    ``(resources, targets, 10)``. A zone ID of ``-1`` marks an unavailable
    candidate; the no-dispatch target deliberately retains that sentinel.
    """
    fire_state = np.asarray(obs["grid"][-1])
    wind_direction = float(obs["scalars"]["wind_direction_deg"])
    stats, adjacent_fuel = _zone_stats(env, fire_state)
    active_zones = [index for index, stat in enumerate(stats) if stat["active"] > 0]
    fuel_zones = [
        index for index, stat in enumerate(stats)
        if stat["adjacent_fuel"] > 0 and stat["active"] == 0
    ]
    threat_zones = [
        index for index, stat in enumerate(stats) if stat["threatened_population"] > 0
    ]

    fuel_rows, fuel_cols = np.where(adjacent_fuel)
    downwind = _downwind_score(fuel_rows, fuel_cols, fire_state, wind_direction)
    downwind_zone = -1
    if len(downwind):
        zone_map = np.empty(fire_state.shape, dtype=np.int16)
        for zone in env.zones:
            row_start, row_stop = zone["row_range"]
            col_start, col_stop = zone["col_range"]
            zone_map[row_start:row_stop, col_start:col_stop] = zone["zone_id"]
        scores = np.zeros(env.n_zones, dtype=np.float32)
        np.add.at(scores, zone_map[fuel_rows, fuel_cols], downwind)
        eligible = [
            zone_id for zone_id in np.argsort(scores)[::-1]
            if stats[int(zone_id)]["active"] == 0 and scores[zone_id] > 0
        ]
        downwind_zone = int(eligible[0]) if eligible else -1

    active_zone = max(active_zones, key=lambda zone_id: stats[zone_id]["active"], default=-1)
    adjacent_zone = max(
        fuel_zones, key=lambda zone_id: stats[zone_id]["adjacent_fuel"], default=-1
    )
    population_zone = max(
        threat_zones,
        key=lambda zone_id: stats[zone_id]["threatened_population"],
        default=-1,
    )

    zones = np.full((len(RESOURCE_TYPES), N_TARGET_TYPES), -1, dtype=np.int64)
    features = np.zeros(
        (len(RESOURCE_TYPES), N_TARGET_TYPES, TARGET_FEATURE_DIM), dtype=np.float32
    )
    for resource_index, resource_type in enumerate(RESOURCE_TYPES):
        nearest = min(
            active_zones,
            key=lambda zone_id: env.zone_travel_time_s[resource_type][zone_id],
            default=-1,
        )
        selected = (
            active_zone,
            downwind_zone,
            adjacent_zone,
            population_zone,
            nearest,
            -1,
        )
        for target_index, zone_id in enumerate(selected):
            zones[resource_index, target_index] = zone_id
            if zone_id < 0:
                continue
            stat = stats[zone_id]
            travel = env.zone_travel_time_s[resource_type][zone_id]
            if not math.isfinite(travel):
                zones[resource_index, target_index] = -1
                continue
            features[resource_index, target_index] = np.array([
                1.0,
                min(stat["active"] / 256.0, 1.0),
                min(stat["adjacent_fuel"] / 256.0, 1.0),
                stat["threatened_population"],
                stat["fuel"],
                stat["building"],
                min(travel / 3600.0, 1.0),
                stat["row"] / max(1.0, fire_state.shape[0]),
                stat["col"] / max(1.0, fire_state.shape[1]),
                1.0 if stat["active"] > 0 else 0.0,
            ], dtype=np.float32)
    return zones, features


def decode_action(resource_index, target_index, target_zones):
    """Convert a semantic policy choice to the environment's dispatch tuple."""
    if target_index == NOOP_TARGET_INDEX:
        return None
    zone_id = int(target_zones[resource_index, target_index])
    if zone_id < 0:
        return None
    return RESOURCE_TYPES[resource_index], zone_id
