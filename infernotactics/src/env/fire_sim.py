"""
Deterministic (rule-based, not learned) cellular-automaton fire-spread model.

Runs directly on the static grid produced by grid_builder.py. Each tick, every
Fuel cell's chance of igniting is driven by: how many of its 8 neighbors are
currently Blaze, the cell's own fuel/building flammability, the local slope
along each spread direction (uphill spreads faster), current wind speed/
direction (downwind spreads faster, upwind slower), humidity (suppresses
spread), and whether the cell is a road (fuel break -- much harder to ignite,
not impossible).

This module has no ML/training in it -- wind_speed/wind_direction/humidity
are passed into step() each tick so this can later be driven by real weather
data (e.g. the Jan 7-8 2025 Palisades Fire Santa Ana event) or by an RL
agent's environment loop, without changing this file.
"""

import numpy as np

# --- Cell states -------------------------------------------------------
SAFE = 0        # non-flammable (negligible fuel and no structure)
FUEL = 1        # unburned, flammable (vegetation and/or structure)
THREAT = 2      # actively igniting this tick; becomes Blaze next tick
BLAZE = 3       # fully burning, can ignite neighbors
BURNED_OUT = 4  # fuel consumed, cannot reignite

STATE_NAMES = ["Safe", "Fuel", "Threat", "Blaze", "Burned Out"]
STATE_COLORS = ["#e8e8e8", "#2ca02c", "#ff7f0e", "#d62728", "#3a2a20"]

# --- Layer indices in grid_static.npy (must match grid_builder.LAYER_NAMES) -
_L_ELEVATION, _L_SLOPE, _L_BDENSITY, _L_BHEIGHT, _L_ROAD, _L_FUEL, _L_WATER = range(7)

# 8-connected Moore neighborhood offsets (dy, dx): row+dy, col+dx
_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),            (0, 1),
    (1, -1),  (1, 0),   (1, 1),
]

# --- Tunable physical constants -----------------------------------------
FUEL_MIN_THRESHOLD = 0.03      # below this effective flammability, a cell can never ignite
BUILDING_FLAMMABILITY_FACTOR = 0.35  # structures ignite less readily than dense brush
ROAD_RESISTANCE_FACTOR = 0.08  # multiplicative ignition penalty for road cells (fuel break)

BASE_SPREAD_PROB = 0.22        # per-neighbor, per-tick base ignition chance at full fuel, flat, no wind
SLOPE_COEFF = 2.5              # larger -> stronger uphill acceleration / downhill suppression
WIND_COEFF = 2.0               # larger -> stronger downwind acceleration / upwind suppression
REFERENCE_WIND_MPH = 40.0      # wind speed normalization reference (strong Santa Ana)
HUMIDITY_SUPPRESSION = 0.85    # at 100% humidity, spread prob is scaled by (1 - this)

SLOPE_FACTOR_CLIP = (0.15, 4.0)
WIND_FACTOR_CLIP = (0.15, 4.0)

BURN_DURATION_TICKS = 4        # ticks a cell stays Blaze before becoming Burned Out


def _shift(arr, dy, dx, fill):
    """Return an array where out[r, c] == arr[r + dy, c + dx], with `fill`
    used for positions that would fall outside the grid."""
    out = np.roll(arr, shift=(-dy, -dx), axis=(0, 1))
    if dy > 0:
        out[-dy:, :] = fill
    elif dy < 0:
        out[:-dy, :] = fill
    if dx > 0:
        out[:, -dx:] = fill
    elif dx < 0:
        out[:, :-dx] = fill
    return out


class FireSim:
    def __init__(self, grid_static, meta, seed=0):
        self.height = meta["height"]
        self.width = meta["width"]
        self.cell_size_m = meta["cell_size_m"]

        elevation = grid_static[_L_ELEVATION]
        building_density = grid_static[_L_BDENSITY]
        road_mask = grid_static[_L_ROAD]
        fuel_density = grid_static[_L_FUEL]
        water_mask = grid_static[_L_WATER]

        self.elevation = elevation
        self.building_density = building_density  # kept for visualization/context only
        self.road_mask = road_mask > 0
        self.water_mask = water_mask > 0.5

        # Static per-cell flammability: vegetation fuel, or (for built-up
        # cells) a reduced structure-ignition chance based on building
        # coverage -- whichever is higher. Roads are handled separately as a
        # per-tick multiplicative resistance rather than baked in here, so
        # its effect stays visible/tunable independent of the fuel layer.
        self.ignitability = np.clip(
            np.maximum(fuel_density, building_density * BUILDING_FLAMMABILITY_FACTOR),
            0.0, 1.0,
        )
        # Belt-and-suspenders: grid_builder already zeroes fuel/building
        # density over water, but force it here too so water cells can never
        # ignite regardless of upstream grid changes.
        self.ignitability = np.where(self.water_mask, 0.0, self.ignitability)
        self.road_resistance = np.where(self.road_mask, ROAD_RESISTANCE_FACTOR, 1.0)

        # Precompute the (static -- elevation never changes) slope factor for
        # each of the 8 spread directions once, instead of every tick.
        self._slope_factor_by_dir = {}
        for dy, dx in _NEIGHBOR_OFFSETS:
            run = self.cell_size_m * np.hypot(dy, dx)
            neighbor_elev = _shift(elevation, dy, dx, fill=np.nan)
            # Fire travels FROM the burning neighbor INTO this (target) cell,
            # so >0 (uphill, faster spread) means the target sits higher than
            # the source neighbor it's catching fire from.
            slope_ratio = (elevation - neighbor_elev) / run
            slope_ratio = np.nan_to_num(slope_ratio, nan=0.0)  # no bias off-grid
            factor = np.exp(SLOPE_COEFF * slope_ratio)
            self._slope_factor_by_dir[(dy, dx)] = np.clip(factor, *SLOPE_FACTOR_CLIP)

        self.state = np.where(self.ignitability >= FUEL_MIN_THRESHOLD, FUEL, SAFE).astype(np.uint8)
        self.blaze_age = np.zeros((self.height, self.width), dtype=np.int32)
        self.tick_count = 0
        self.rng = np.random.default_rng(seed)

    def ignite(self, row, col, radius=0):
        """Force-ignite a cell (or a small disk of cells) regardless of its
        current state, e.g. to set the initial fire origin."""
        rr, cc = np.ogrid[:self.height, :self.width]
        mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2
        self.state[mask] = BLAZE
        self.blaze_age[mask] = 1

    def state_counts(self):
        counts = np.bincount(self.state.ravel(), minlength=len(STATE_NAMES))
        return {name: int(c) for name, c in zip(STATE_NAMES, counts)}

    def step(self, wind_speed_mph=0.0, wind_direction_deg=0.0, humidity_pct=30.0):
        """Advance the simulation by one tick.

        wind_direction_deg: meteorological convention -- the direction the
            wind is blowing FROM (0=N, 90=E, 180=S, 270=W). e.g. a Santa Ana
            event is wind_direction_deg≈45 (from the NE).
        """
        wind_to_deg = (wind_direction_deg + 180.0) % 360.0
        theta = np.radians(wind_to_deg)
        wind_to_east, wind_to_north = np.sin(theta), np.cos(theta)
        wind_speed_norm = min(wind_speed_mph / REFERENCE_WIND_MPH, 1.5)

        humidity_factor = 1.0 - HUMIDITY_SUPPRESSION * np.clip(humidity_pct / 100.0, 0.0, 1.0)

        blaze_mask = self.state == BLAZE
        fuel_mask = self.state == FUEL

        p_no_ignite = np.ones((self.height, self.width), dtype=np.float32)

        for dy, dx in _NEIGHBOR_OFFSETS:
            neighbor_is_blaze = _shift(blaze_mask, dy, dx, fill=False)
            if not neighbor_is_blaze.any():
                continue

            dist = np.hypot(dy, dx)
            # Fire travel direction is FROM the neighbor at (dy, dx) INTO
            # this cell, i.e. the vector (-dy, -dx); row+ is south so
            # north = -row_delta.
            spread_east, spread_north = -dx / dist, dy / dist
            alignment = spread_east * wind_to_east + spread_north * wind_to_north
            wind_factor = np.clip(
                np.exp(WIND_COEFF * wind_speed_norm * alignment), *WIND_FACTOR_CLIP
            )

            per_neighbor_prob = (
                BASE_SPREAD_PROB
                * self.ignitability
                * self._slope_factor_by_dir[(dy, dx)]
                * wind_factor
                * self.road_resistance
                * humidity_factor
            )
            per_neighbor_prob = np.clip(per_neighbor_prob, 0.0, 1.0)
            contribution = np.where(neighbor_is_blaze, per_neighbor_prob, 0.0)
            p_no_ignite *= (1.0 - contribution)

        p_ignite = 1.0 - p_no_ignite
        draws = self.rng.random((self.height, self.width))
        newly_threat = fuel_mask & (draws < p_ignite)

        new_state = self.state.copy()

        # 1. Fuel cells that catch this tick become Threat (an ignition front
        #    that becomes Blaze next tick), based on last tick's Blaze cells.
        new_state[newly_threat] = THREAT

        # 2. Cells that were Threat last tick fully ignite this tick.
        was_threat = self.state == THREAT
        new_state[was_threat] = BLAZE
        self.blaze_age[was_threat] = 1

        # 3. Cells that were already Blaze age by one tick; once fuel is
        #    consumed (age >= BURN_DURATION_TICKS) they burn out.
        self.blaze_age[blaze_mask] += 1
        burned_out = blaze_mask & (self.blaze_age >= BURN_DURATION_TICKS)
        new_state[burned_out] = BURNED_OUT

        self.state = new_state
        self.tick_count += 1
        return self.state
