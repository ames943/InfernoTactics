"""
Gym-style RL environment wrapper around fire_sim.py.

This module defines the INTERFACE the future CNN/MLP/actor-critic model code
will train against (observation format, action format, reward). It contains
no neural networks and no training loop -- reset()/step() just drive the
deterministic fire-spread cellular automaton (fire_sim.FireSim) plus simple,
explicit resource-dispatch/reward bookkeeping.

Action space
------------
action = (resource_type, target_zone) where:
  - resource_type: one of RESOURCE_TYPES ("water_team", "trench_crew",
    "rescue_vehicle"), or None/"noop" to dispatch nothing this tick.
  - target_zone: int zone id in [0, n_zones); ignored if resource_type is
    None/"noop". Zones are a coarse rectangular partition of the grid (see
    _build_zones) used ONLY to give the action space a manageable size and a
    real road-network travel time -- it does not change what the CNN branch
    sees, which is still the full-resolution grid.

Observation space
-----------------
obs = {
    "grid": float32 array (n_static_layers + 1, height, width) -- the 7
        static layers from grid_static.npy (see STATIC_LAYER_NAMES) stacked
        with the current per-cell fire state (fire_sim.FireSim.state, raw
        ordinal 0-4: Safe/Fuel/Threat/Blaze/Burned Out) as the last channel.
        This is the CNN branch's input.
    "scalars": OrderedDict, fixed key order given by SCALAR_KEYS -- wind
        speed/direction, humidity, resources available per type, elapsed
        ticks. This is the MLP branch's input. Use flatten_scalars(obs) to
        get a plain float32 vector in SCALAR_KEYS order once model code
        needs one.
}

Reward (see the project reward-formula spec)
---------------------------------------------
  +50  a dispatched water_team's arrival extinguishes >=1 active fire cell
 -100  a building cell is consumed by fire this tick (BUILDING_DESTROYED_PENALTY)
  -20  same, but the building was in a zone a rescue_vehicle had reached
       first (RESCUE_PENALTY_REDUCTION cuts the penalty, doesn't zero it)
  -10  a resource action has no effect: no unit of that type was available,
       OR the unit arrives but its effect can't apply (trench_crew targets a
       cell that's already on fire; water_team/rescue_vehicle target a zone
       with nothing to put out / no one threatened) (RESOURCE_WASTED_PENALTY)
  -lambda * travel_time_seconds, charged once per successful dispatch
       (LAMBDA_TRAVEL_TIME)
Congestion term intentionally omitted (Tier 3, per project plan).
Only water_team has an explicit positive reward term -- trench_crew and
rescue_vehicle create value indirectly, by preventing future -100/-10
events, matching the reward terms specified in the project plan (no extra
terms invented here).
"""

import json
import math
import os
import sys
from collections import OrderedDict

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR, ROADS_GRAPHML_PATH  # noqa: E402
from env.fire_sim import (  # noqa: E402
    BLAZE,
    BURNED_OUT,
    FUEL,
    SAFE,
    THREAT,
    FireSim,
)

GRID_STATIC_PATH = os.path.join(DATA_DIR, "grid_static.npy")
GRID_META_PATH = os.path.join(DATA_DIR, "grid_meta.json")

# Must match grid_builder.LAYER_NAMES -- duplicated here (rather than
# importing grid_builder, which pulls in geopandas/rasterio) the same way
# fire_sim.py duplicates the layer index convention.
STATIC_LAYER_NAMES = [
    "elevation", "slope", "building_density", "building_height",
    "road_mask", "fuel_density", "water_mask",
]
LAYER_INDEX = {name: i for i, name in enumerate(STATIC_LAYER_NAMES)}

# --- Macro-zones (action-space abstraction only, see module docstring) -----
ZONE_SIZE_CELLS = 80  # -> 4x8 = 32 zones on the current 316x595 grid, ~2.4km/side

# --- Local physical footprint of a single resource's effect -----------------
# Zones exist only to make the action space small + give a real travel-time
# target; the actual water/trench/rescue effect is a small local disk at the
# zone's centroid, not the whole (much larger) zone rectangle.
EFFECT_RADIUS_CELLS = 3

# --- Ignition-point sampling heuristic ---------------------------------------
# "Sensible" default ignition = flammable, not water, AND within this many
# cells of at least one building cell -- avoids spawning episodes deep in the
# Topanga wilderness core where there's nothing for the agent to protect and
# nothing interesting to learn.
WUI_PROXIMITY_RADIUS_CELLS = 15

BUILDING_PRESENCE_THRESHOLD = 0.10  # building_density above which a cell "is a building"

# --- Resources ---------------------------------------------------------------
RESOURCE_COUNTS = {"water_team": 3, "trench_crew": 3, "rescue_vehicle": 3}
RESOURCE_TYPES = tuple(RESOURCE_COUNTS)
DEPLOYED_BUSY_TICKS = 5  # ticks a unit stays tied up at the incident after arriving, before it can be redispatched

# --- Time / weather -----------------------------------------------------------
# PLACEHOLDER: maps sim ticks <-> wall-clock minutes so road travel times
# (real seconds) and fire_sim ticks share a consistent clock. Replace with a
# derivation from the real Jan 7-8 2025 timeline once that data is wired in.
TICK_DURATION_MINUTES = 2.0
MAX_TICKS = 150  # episode cap (~5 sim-hours at the placeholder tick duration)

# --- Depot (fixed resource staging point for travel-time routing) -----------
# Approximate real-world location: LAFD Fire Station 69, Pacific Palisades.
DEPOT_LONLAT = (-118.5253, 34.0489)

# --- Reward constants (see module docstring) ---------------------------------
FIRE_EXTINGUISHED_REWARD = 50.0
BUILDING_DESTROYED_PENALTY = 100.0
RESOURCE_WASTED_PENALTY = 10.0
RESCUE_PENALTY_REDUCTION = 0.8  # fraction of BUILDING_DESTROYED_PENALTY waived if evacuated first
LAMBDA_TRAVEL_TIME = 0.02  # reward penalty per second of dispatch travel time

SCALAR_KEYS = (
    "wind_speed_mph",
    "wind_direction_deg",
    "humidity_pct",
    "water_team_available",
    "trench_crew_available",
    "rescue_vehicle_available",
    "time_elapsed_ticks",
)


def flatten_scalars(scalars):
    """Convert the observation's 'scalars' OrderedDict into a fixed-order
    float32 vector, in SCALAR_KEYS order, for the future MLP branch."""
    return np.array([scalars[k] for k in SCALAR_KEYS], dtype=np.float32)


def _weather_schedule(tick):
    """PLACEHOLDER synthetic weather, standing in for real Jan 7-8 2025
    Palisades Fire hourly observations until that data is wired in. Ramps a
    Santa-Ana-style offshore wind up over the first 10 ticks then holds it,
    with humidity held low and constant -- deliberately simple/fixed per the
    current stage of the project, NOT meant to be realistic yet."""
    wind_speed_mph = min(10.0 + 3.5 * tick, 45.0)
    wind_direction_deg = 45.0  # from the NE, consistent with the validated test_fire_sim.py scenario
    humidity_pct = 8.0
    return wind_speed_mph, wind_direction_deg, humidity_pct


def _disk_slice(center_row, center_col, radius, height, width):
    """Bounding-box slice + local boolean disk mask (Euclidean, like
    FireSim.ignite) around (center_row, center_col), clipped to the grid."""
    r0, r1 = max(0, center_row - radius), min(height, center_row + radius + 1)
    c0, c1 = max(0, center_col - radius), min(width, center_col + radius + 1)
    rr, cc = np.ogrid[r0:r1, c0:c1]
    mask = (rr - center_row) ** 2 + (cc - center_col) ** 2 <= radius ** 2
    return r0, r1, c0, c1, mask


def _apply_water(sim, row, col):
    """Knock down active fire in a small disk around (row, col): Threat/Blaze
    cells are extinguished (-> SAFE, not BURNED_OUT, so they are never later
    miscounted as a "building destroyed" event). Returns cells extinguished."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    region = sim.state[r0:r1, c0:c1]
    active = disk & np.isin(region, (THREAT, BLAZE))
    n = int(active.sum())
    if n:
        region[active] = SAFE
    return n


def _apply_trench(sim, row, col):
    """Dig a permanent fire break in a small disk around (row, col). Fails
    (0 cells affected) if any cell in the footprint is already Threat/Blaze
    -- per the project plan, a trench crew can't place a line on ground
    that's already burning. Otherwise converts FUEL cells in the footprint
    to SAFE and permanently zeroes their ignitability."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    region_state = sim.state[r0:r1, c0:c1]
    if (disk & np.isin(region_state, (THREAT, BLAZE))).any():
        return 0
    fuel_cells = disk & (region_state == FUEL)
    n = int(fuel_cells.sum())
    if n:
        region_state[fuel_cells] = SAFE
        sim.ignitability[r0:r1, c0:c1][fuel_cells] = 0.0
    return n


def _apply_rescue(sim, row, col, evacuated_cells):
    """Mark threatened building cells in a small disk around (row, col) as
    evacuated: they still burn per fire_sim's normal physics, but the
    building-destroyed penalty is reduced later (see RESCUE_PENALTY_REDUCTION)
    instead of the reward branch changing fire_sim's state directly."""
    r0, r1, c0, c1, disk = _disk_slice(row, col, EFFECT_RADIUS_CELLS, sim.height, sim.width)
    building_region = sim.building_density[r0:r1, c0:c1]
    state_region = sim.state[r0:r1, c0:c1]
    threatened = disk & (building_region > BUILDING_PRESENCE_THRESHOLD) & np.isin(state_region, (THREAT, BLAZE))
    rows, cols = np.where(threatened)
    for lr, lc in zip(rows, cols):
        evacuated_cells.add((r0 + lr, c0 + lc))
    return len(rows)


def _effect_target_point(sim, zone):
    """Where a resource's small EFFECT_RADIUS_CELLS footprint should land
    within its (much larger) target zone.

    Zones are ~80 cells / ~2.4km per side while the physical effect disk is
    only ~7 cells / ~210m across, so pinning every arrival to the zone's
    fixed geometric centroid means water/rescue would need the fire to be
    sitting on that one exact point -- effectively never, even for a policy
    that has correctly identified the right zone. Instead, target the
    centroid of whatever active fire (Threat/Blaze) cells are currently in
    the zone, if any -- i.e. "help wherever the fire actually is within the
    zone I was sent to". Falls back to the zone's geometric centroid when
    there's no fire in the zone yet, which is exactly the case a trench crew
    digging a defensive line ahead of the fire needs."""
    r0, r1 = zone["row_range"]
    c0, c1 = zone["col_range"]
    region_state = sim.state[r0:r1, c0:c1]
    active_rows, active_cols = np.where(np.isin(region_state, (THREAT, BLAZE)))
    if len(active_rows):
        return r0 + int(round(active_rows.mean())), c0 + int(round(active_cols.mean()))
    return zone["centroid_row"], zone["centroid_col"]


def _near_building_mask(building_present, radius):
    """Vectorized "is there a building-present cell within `radius` cells"
    test for every cell, via a summed-area table (avoids adding a scipy/
    scikit-image dependency just for a box-window OR)."""
    h, w = building_present.shape
    integral = np.zeros((h + 1, w + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(building_present.astype(np.int64), axis=0), axis=1)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    r0 = np.broadcast_to(np.clip(rows - radius, 0, h), (h, w))
    r1 = np.broadcast_to(np.clip(rows + radius + 1, 0, h), (h, w))
    c0 = np.broadcast_to(np.clip(cols - radius, 0, w), (h, w))
    c1 = np.broadcast_to(np.clip(cols + radius + 1, 0, w), (h, w))
    total = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    return total > 0


def _new_resource_unit():
    return {"state": "available", "remaining_ticks": 0, "target_zone": None}


def _build_zones(grid_static, meta, zone_size_cells=ZONE_SIZE_CELLS):
    height, width = meta["height"], meta["width"]
    water = grid_static[LAYER_INDEX["water_mask"]] > 0.5
    building_density = grid_static[LAYER_INDEX["building_density"]]

    zones = []
    zone_id = 0
    for r0 in range(0, height, zone_size_cells):
        r1 = min(r0 + zone_size_cells, height)
        for c0 in range(0, width, zone_size_cells):
            c1 = min(c0 + zone_size_cells, width)
            zones.append({
                "zone_id": zone_id,
                "row_range": (r0, r1),
                "col_range": (c0, c1),
                "centroid_row": (r0 + r1) // 2,
                "centroid_col": (c0 + c1) // 2,
                "is_water": bool(water[r0:r1, c0:c1].mean() > 0.9),
                "building_cells": int((building_density[r0:r1, c0:c1] > BUILDING_PRESENCE_THRESHOLD).sum()),
            })
            zone_id += 1
    return zones


def _load_routing_graph(grid_crs):
    """Load the OSMnx road graph and project it into the grid's CRS so zone
    centroids (already in that CRS, from the grid's affine transform) can be
    used directly for nearest-node lookups without a second pyproj hop.
    Also adds OSMnx's imputed edge speeds/travel times, reused for real
    shortest-path ETAs (nx.single_source_dijkstra_path_length, weight=
    'travel_time', seconds) from a fixed depot to each zone."""
    graph = ox.load_graphml(ROADS_GRAPHML_PATH)
    graph = ox.project_graph(graph, to_crs=grid_crs)
    graph = ox.routing.add_edge_speeds(graph)
    graph = ox.routing.add_edge_travel_times(graph)
    return graph


class InfernoEnv:
    """Gym-style environment: reset(ignition_point=None) -> obs;
    step(action) -> (obs, reward, done, info). See the module docstring for
    the observation/action/reward formats."""

    def __init__(self, grid_static_path=GRID_STATIC_PATH, grid_meta_path=GRID_META_PATH, seed=None):
        self.grid_static = np.load(grid_static_path).astype(np.float32)
        with open(grid_meta_path) as f:
            self.meta = json.load(f)
        self.height, self.width = self.meta["height"], self.meta["width"]
        self.rng = np.random.default_rng(seed)

        self._building_mask = self.grid_static[LAYER_INDEX["building_density"]] > BUILDING_PRESENCE_THRESHOLD

        # Reuse FireSim's own ignitability/water logic as the single source
        # of truth for "is this cell a valid fuel cell" rather than
        # re-deriving the same physics constants here.
        probe_sim = FireSim(self.grid_static, self.meta, seed=0)
        fuel_cell_mask = probe_sim.state == FUEL
        near_building = _near_building_mask(self._building_mask, WUI_PROXIMITY_RADIUS_CELLS)
        self._ignition_candidates = np.argwhere(fuel_cell_mask & near_building)
        assert len(self._ignition_candidates) > 0, "no WUI-adjacent fuel cells found for ignition sampling"

        self.zones = _build_zones(self.grid_static, self.meta)
        self.n_zones = len(self.zones)

        print(f"[InfernoEnv] Loading + preparing road routing graph from {ROADS_GRAPHML_PATH} ...")
        self.road_graph = _load_routing_graph(self.meta["crs"])
        self._prepare_routing()

        self.sim = None
        self.tick_count = 0
        self.resources = None
        self.evacuated_cells = set()
        self._last_weather = None
        self.ignition_point = None

    def _prepare_routing(self):
        transform_coeffs = self.meta["transform"]  # (a, b, c, d, e, f), (col,row)->(x,y)
        a, b, c, d, e, f = transform_coeffs

        def pixel_to_xy(row, col):
            x = a * (col + 0.5) + b * (row + 0.5) + c
            y = d * (col + 0.5) + e * (row + 0.5) + f
            return x, y

        zone_xy = [pixel_to_xy(z["centroid_row"], z["centroid_col"]) for z in self.zones]
        xs = [xy[0] for xy in zone_xy]
        ys = [xy[1] for xy in zone_xy]
        zone_nodes = ox.distance.nearest_nodes(self.road_graph, X=xs, Y=ys)

        transformer = Transformer.from_crs("EPSG:4326", self.meta["crs"], always_xy=True)
        depot_x, depot_y = transformer.transform(*DEPOT_LONLAT)
        self.depot_node = ox.distance.nearest_nodes(self.road_graph, X=depot_x, Y=depot_y)

        travel_times_from_depot = nx.single_source_dijkstra_path_length(
            self.road_graph, self.depot_node, weight="travel_time"
        )

        self.zone_travel_time_s = []
        for zone, node in zip(self.zones, zone_nodes):
            zone["road_node"] = int(node)
            travel_s = travel_times_from_depot.get(node, math.inf)
            zone["travel_time_s"] = travel_s
            self.zone_travel_time_s.append(travel_s)

        n_unreachable = sum(1 for t in self.zone_travel_time_s if not math.isfinite(t))
        if n_unreachable:
            print(f"[InfernoEnv] Warning: {n_unreachable}/{self.n_zones} zones unreachable "
                  f"from depot via the road graph (dispatches there will always be wasted).")

    # --- Episode lifecycle ---------------------------------------------------

    def _sample_ignition_point(self):
        idx = self.rng.integers(len(self._ignition_candidates))
        row, col = self._ignition_candidates[idx]
        return int(row), int(col)

    def reset(self, ignition_point=None, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        sim_seed = int(self.rng.integers(0, 2 ** 31 - 1))
        self.sim = FireSim(self.grid_static, self.meta, seed=sim_seed)

        if ignition_point is None:
            ignition_point = self._sample_ignition_point()
        row, col = ignition_point
        self.sim.ignite(row, col, radius=1)
        self.ignition_point = (row, col)

        self.tick_count = 0
        self.resources = {
            rtype: [_new_resource_unit() for _ in range(RESOURCE_COUNTS[rtype])]
            for rtype in RESOURCE_TYPES
        }
        self.evacuated_cells = set()
        self._last_weather = _weather_schedule(0)

        return self._build_observation()

    def _parse_action(self, action):
        if action is None:
            return None, None
        resource_type, target_zone = action
        if resource_type is None or resource_type == "noop":
            return None, None
        if resource_type not in RESOURCE_TYPES:
            raise ValueError(f"Unknown resource_type {resource_type!r}; expected one of "
                              f"{RESOURCE_TYPES} or None/'noop'")
        if not (0 <= target_zone < self.n_zones):
            raise ValueError(f"target_zone {target_zone} out of range [0, {self.n_zones})")
        return resource_type, target_zone

    def _try_dispatch(self, resource_type, target_zone_id):
        travel_s = self.zone_travel_time_s[target_zone_id]
        if not math.isfinite(travel_s):
            return {"resource_type": resource_type, "target_zone": target_zone_id,
                    "status": "zone_unreachable", "reward_delta": -RESOURCE_WASTED_PENALTY}

        roster = self.resources[resource_type]
        unit = next((u for u in roster if u["state"] == "available"), None)
        if unit is None:
            return {"resource_type": resource_type, "target_zone": target_zone_id,
                    "status": "no_unit_available", "reward_delta": -RESOURCE_WASTED_PENALTY}

        eta_ticks = max(1, math.ceil(travel_s / (TICK_DURATION_MINUTES * 60.0)))
        unit["state"] = "traveling"
        unit["remaining_ticks"] = eta_ticks
        unit["target_zone"] = target_zone_id
        return {"resource_type": resource_type, "target_zone": target_zone_id, "status": "dispatched",
                "travel_time_s": travel_s, "eta_ticks": eta_ticks,
                "reward_delta": -LAMBDA_TRAVEL_TIME * travel_s}

    def _advance_resources(self):
        """Advance every non-available unit by one tick; apply effects for
        units that just finished traveling, and free up units that just
        finished their post-arrival busy period."""
        reward = 0.0
        events = []
        for rtype in RESOURCE_TYPES:
            for unit in self.resources[rtype]:
                if unit["state"] == "available":
                    continue
                unit["remaining_ticks"] -= 1
                if unit["remaining_ticks"] > 0:
                    continue

                if unit["state"] == "traveling":
                    zone = self.zones[unit["target_zone"]]
                    row, col = _effect_target_point(self.sim, zone)
                    if rtype == "water_team":
                        n_affected = _apply_water(self.sim, row, col)
                    elif rtype == "trench_crew":
                        n_affected = _apply_trench(self.sim, row, col)
                    else:
                        n_affected = _apply_rescue(self.sim, row, col, self.evacuated_cells)

                    success = n_affected > 0
                    if success and rtype == "water_team":
                        reward += FIRE_EXTINGUISHED_REWARD
                    elif not success:
                        reward -= RESOURCE_WASTED_PENALTY
                    events.append({"resource_type": rtype, "zone": unit["target_zone"],
                                   "cells_affected": n_affected, "success": success})
                    unit["state"] = "deployed"
                    unit["remaining_ticks"] = DEPLOYED_BUSY_TICKS
                elif unit["state"] == "deployed":
                    unit["state"] = "available"
                    unit["target_zone"] = None
        return reward, events

    def _score_building_destruction(self, before_state):
        newly_burned = self._building_mask & (before_state != BURNED_OUT) & (self.sim.state == BURNED_OUT)
        rows, cols = np.where(newly_burned)
        reward = 0.0
        n_evacuated = 0
        for r, c in zip(rows, cols):
            if (r, c) in self.evacuated_cells:
                reward -= BUILDING_DESTROYED_PENALTY * (1.0 - RESCUE_PENALTY_REDUCTION)
                n_evacuated += 1
                self.evacuated_cells.discard((r, c))
            else:
                reward -= BUILDING_DESTROYED_PENALTY
        return reward, int(len(rows)), n_evacuated

    def step(self, action):
        resource_type, target_zone = self._parse_action(action)

        # Advance units already en route/deployed BEFORE committing this
        # tick's new dispatch, so a freshly-dispatched unit's ETA countdown
        # starts on the *next* step() rather than losing a tick to this one.
        reward, resource_events = self._advance_resources()

        dispatch_info = None
        if resource_type is not None:
            dispatch_info = self._try_dispatch(resource_type, target_zone)
            reward += dispatch_info["reward_delta"]

        wind_speed, wind_direction, humidity = _weather_schedule(self.tick_count)
        self._last_weather = (wind_speed, wind_direction, humidity)

        before_state = self.sim.state.copy()
        self.sim.step(wind_speed_mph=wind_speed, wind_direction_deg=wind_direction, humidity_pct=humidity)
        destroy_reward, n_destroyed, n_destroyed_evacuated = self._score_building_destruction(before_state)
        reward += destroy_reward

        self.tick_count += 1
        counts = self.sim.state_counts()
        contained = counts["Threat"] == 0 and counts["Blaze"] == 0
        timed_out = self.tick_count >= MAX_TICKS
        done = contained or timed_out

        info = {
            "dispatch": dispatch_info,
            "resource_events": resource_events,
            "weather": {"wind_speed_mph": wind_speed, "wind_direction_deg": wind_direction, "humidity_pct": humidity},
            "state_counts": counts,
            "buildings_destroyed": n_destroyed,
            "buildings_destroyed_evacuated": n_destroyed_evacuated,
            "contained": contained,
            "timeout": timed_out and not contained,
            "tick": self.tick_count,
        }
        return self._build_observation(), reward, done, info

    # --- Observation -----------------------------------------------------------

    def _build_observation(self):
        grid = np.concatenate(
            [self.grid_static, self.sim.state[np.newaxis, :, :].astype(np.float32)], axis=0
        )
        wind_speed, wind_direction, humidity = self._last_weather
        available = {
            rtype: float(sum(1 for u in self.resources[rtype] if u["state"] == "available"))
            for rtype in RESOURCE_TYPES
        }
        scalars = OrderedDict([
            ("wind_speed_mph", wind_speed),
            ("wind_direction_deg", wind_direction),
            ("humidity_pct", humidity),
            ("water_team_available", available["water_team"]),
            ("trench_crew_available", available["trench_crew"]),
            ("rescue_vehicle_available", available["rescue_vehicle"]),
            ("time_elapsed_ticks", float(self.tick_count)),
        ])
        return {"grid": grid, "scalars": scalars}
