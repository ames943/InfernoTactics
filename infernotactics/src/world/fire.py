"""Pyretechnics 2026.10.20 adapter, not an atmosphere or structure-fire model.

The public grid is north-up, in metres and seconds. Pyretechnics uses a
south-to-north row axis, minutes, 10-m wind in km/h, and spread rate in m/min.
Only this boundary performs those conversions. Fire behavior rasters describe
behavior at arrival, not a live combustion or smoldering observation.

Treatments are experimental spread modifiers, held constant for each canonical
minute. Water strength is delivered litres/m²; completed line footprints are
cell-center rasterizations (radius zero selects the containing cell). Callers
supply the current treatment list, not incremental doses. No treatment erases
arrival history. Observation calls never integrate or draw random numbers.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from pyproj import Transformer
from pyretechnics import eulerian_level_set as els
from pyretechnics import fuel_models
from pyretechnics.space_time_cube import SpaceTimeCube

from .contracts import Event, Scenario, Treatment, event_sort_key

ENGINE_VERSION = "2026.10.20"
COUPLING_S = 60


def _finite(value: object, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def _cell(scenario: Scenario, x: float, y: float) -> tuple[int, int]:
    a, _, c, _, e, f = scenario.transform
    col, row = (x - c) / a, (y - f) / e
    height, width = scenario.landscape["valid"].shape
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("point is outside the fire domain")
    return math.floor(row), math.floor(col)


def treatment_mask(scenario: Scenario, treatment: Treatment) -> np.ndarray:
    """Rasterize an explicit projected footprint without clipping its target.

    Line-work callers must emit only completed increments, not a whole planned
    segment. The radius is the increment's width/footprint, not a work rate.
    Footprints may cross the grid edge; their outside portion has no effect.
    """
    valid = scenario.landscape["valid"]
    if treatment.radius_m == 0:
        mask = np.zeros(valid.shape, dtype=bool)
        try:
            row, col = _cell(scenario, treatment.x, treatment.y)
        except ValueError:
            return mask
        mask[row, col] = valid[row, col]
        return mask
    a, _, c, _, e, f = scenario.transform
    xs = c + (np.arange(valid.shape[1]) + 0.5) * a
    ys = f + (np.arange(valid.shape[0]) + 0.5) * e
    return (((xs[None, :] - treatment.x) ** 2 +
             (ys[:, None] - treatment.y) ** 2 <= treatment.radius_m ** 2) & valid)


class FireEngine:
    """Stateful physical spread on fixed 60-second stochastic substreams."""

    def __init__(self, scenario: Scenario, seed: int):
        scenario.validate()
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("fire seed must be a nonnegative integer")
        self.scenario = scenario
        self.seed = int(seed)
        self.time_s = 0
        self.domain_exited = False
        self.area_censored = False
        self._events: list[Event] = []
        self._activated: set[Treatment] = set()
        self._spot_ignitions: dict[float, set[tuple[int, int]]] = {}
        self._valid = scenario.landscape["valid"].copy()
        height, width = self._valid.shape
        self._shape = (scenario.weather["wind_speed_m_s"].shape[0], height, width)
        self._resolution = (scenario.weather_step_s / 60.0,
                            scenario.transform[0], scenario.transform[0])
        fuels = scenario.landscape["fuel_model"]
        supported = set(fuel_models.list_fuel_model_numbers())
        unknown = set(int(code) for code in np.unique(fuels[self._valid])) - supported
        if unknown:
            raise ValueError(f"unsupported Pyretechnics fuel codes: {sorted(unknown)}")
        self._burnable = self._valid & ~np.isin(fuels, [91, 92, 93, 98, 99])
        # Invalid sentinels must never reach upstream's unchecked fuel cache.
        self._fuel = np.where(self._valid, fuels, 99).astype(np.float32)
        self._cubes = self._build_cubes()
        self._state = els.SpreadState(self._shape)
        self._state.set_resolution(self._resolution)
        self._state.set_start_time(0.0)
        self._transformer = Transformer.from_crs("EPSG:4326", scenario.crs, always_xy=True)
        self._spot_config = self._spotting_config()
        self._suppression = self._suppression_config()
        self._breached_line = np.zeros(self._valid.shape, dtype=bool)
        # Only the outer fire-domain boundary censors area; interior nodata
        # remains an explicit nonburnable coverage mask, not a domain exit.
        self._coverage_edge = np.zeros(self._valid.shape, dtype=bool)
        self._coverage_edge[0, :] = self._valid[0, :]
        self._coverage_edge[-1, :] = self._valid[-1, :]
        self._coverage_edge[:, 0] = self._valid[:, 0]
        self._coverage_edge[:, -1] = self._valid[:, -1]

    def _cube(self, array: np.ndarray) -> SpaceTimeCube:
        # Copy once into contiguous float32; upstream retains the supplied buffer.
        return SpaceTimeCube(self._shape, np.ascontiguousarray(np.flip(array, axis=-2), dtype=np.float32))

    def _build_cubes(self) -> dict:
        cubes = {}
        for name in ("slope", "aspect", "canopy_cover", "canopy_height",
                     "canopy_base_height", "canopy_bulk_density"):
            cubes[name] = self._cube(np.where(self._valid, self.scenario.landscape[name], 0))
        cubes["fuel_model"] = self._cube(self._fuel)
        for name, array in self.scenario.weather.items():
            if array.shape[1:] == self._valid.shape:
                array = np.where(self._valid[None, :, :], array, 0)
            if name == "wind_speed_m_s":
                name, array = "wind_speed_10m", array * 3.6
            elif name == "upwind_direction_deg":
                name = "upwind_direction"
            elif name == "temperature_c":
                name = "temperature"
            # Scenario permits extra weather metadata arrays; only the contract
            # fields consumed by this engine are forwarded upstream.
            if name in {"wind_speed_10m", "upwind_direction", "temperature",
                        "fuel_moisture_dead_1hr", "fuel_moisture_dead_10hr",
                        "fuel_moisture_dead_100hr", "fuel_moisture_live_herbaceous",
                        "fuel_moisture_live_woody", "foliar_moisture"}:
                cubes[name] = self._cube(array)
        return cubes

    def _spotting_config(self) -> dict | None:
        config = self.scenario.provenance.get("spotting", {"enabled": False})
        if not isinstance(config, dict) or not isinstance(config.get("enabled"), bool):
            raise ValueError("provenance.spotting requires boolean enabled")
        if not config["enabled"]:
            return None
        if not isinstance(config.get("source"), str) or not config["source"].strip():
            raise ValueError("enabled spotting requires a provenance source")
        names = ("firebrands_per_unit_heat", "downwind_distance_mean",
                 "fireline_intensity_exponent", "wind_speed_exponent",
                 "downwind_variance_mean_ratio", "crosswind_distance_stdev", "decay_distance")
        values = {name: _finite(config.get(name), f"spotting.{name}", 0) for name in names}
        for name in ("downwind_distance_mean", "decay_distance"):
            if values[name] == 0:
                raise ValueError(f"spotting.{name} must be positive")
        return values

    def _suppression_config(self) -> dict | None:
        config = self.scenario.provenance.get("suppression")
        if config is None:
            return None  # Free spread does not require hypothetical effect parameters.
        if not isinstance(config, dict) or not isinstance(config.get("source"), str) or not config["source"].strip():
            raise ValueError("suppression parameters require a provenance source")
        values = {name: _finite(config.get(name), f"suppression.{name}", 0)
                  for name in ("k", "decay_s", "radius_m", "application_rate_l_s")}
        if values["decay_s"] == 0 or values["radius_m"] == 0 or values["application_rate_l_s"] == 0:
            raise ValueError("suppression decay_s, radius_m and application_rate_l_s must be positive")
        return values

    def _align_state_clock(self) -> None:
        """Preserve a silent interval without rewriting historical arrivals.

        Upstream can return at its start when there are no tracked cells. Its
        public clock setter is initialization-only. Rebase only in that case
        (or its subsecond termination tolerance), copying public state matrices;
        phi_star is synchronized by upstream at each call's entry.
        """
        minutes = self.time_s / 60.0
        if math.isclose(self._state.get_simulation_time(), minutes, rel_tol=0, abs_tol=1e-7):
            return
        old = self._state.get_full_matrices()
        state = els.SpreadState(self._shape)
        state.set_resolution(self._resolution)
        state.set_start_time(minutes)
        for name, target in state.get_full_matrices().items():
            target[:] = old[name]
        self._state = state

    def ignite(self, longitude: float, latitude: float) -> None:
        longitude = _finite(longitude, "ignition longitude")
        latitude = _finite(latitude, "ignition latitude")
        if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
            raise ValueError("ignition longitude/latitude is outside geographic bounds")
        if self.time_s >= self.scenario.duration_s:
            raise ValueError("cannot ignite after scenario horizon")
        x, y = self._transformer.transform(longitude, latitude)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("ignition coordinates cannot be projected")
        row, col = _cell(self.scenario, x, y)
        if not self._valid[row, col]:
            raise ValueError("ignition is on a nodata cell")
        if not self._burnable[row, col]:
            raise ValueError("ignition is on a nonburnable fuel cell")
        self._align_state_clock()
        upstream_row = self._valid.shape[0] - 1 - row
        arrival = self._state.get_full_matrices(["time_of_arrival"])["time_of_arrival"]
        details = {"longitude": longitude, "latitude": latitude, "x": x, "y": y,
                   "row": row, "col": col}
        if np.isfinite(arrival[upstream_row, col]):
            kind = "ignition_ignored"
        else:
            self._state.ignite_cell((upstream_row, col))
            kind = "ignition"
        self._events.append(Event(float(self.time_s), kind, None, details))

    def _modifiers(self, treatments: list[Treatment]) -> tuple[np.ndarray, np.ndarray, list[Event]]:
        adjustment = np.ones(self._valid.shape, dtype=np.float32)
        lines = np.zeros(self._valid.shape, dtype=bool)
        events = []
        for treatment in dict.fromkeys(treatments):
            if not (treatment.time_s <= self.time_s < treatment.expires_s):
                continue
            if self._suppression is None:
                raise ValueError("treatments require provenance-tagged suppression parameters")
            footprint = treatment_mask(self.scenario, treatment) & self._burnable
            if treatment.kind == "water":
                age = self.time_s - treatment.time_s
                attenuation = math.exp(-self._suppression["k"] * treatment.strength *
                                       math.exp(-age / self._suppression["decay_s"]))
                adjustment[footprint] *= attenuation
            elif treatment.strength > 0:
                lines |= footprint & ~self._breached_line
            if treatment not in self._activated:
                self._activated.add(treatment)
                events.append(Event(float(self.time_s), "treatment_activated", None, {
                    "kind": treatment.kind, "x": treatment.x, "y": treatment.y,
                    "radius_m": treatment.radius_m, "strength": treatment.strength,
                    "completed_s": treatment.time_s,
                    "coupling_delay_s": self.time_s - treatment.time_s,
                    "suppression_validation": "experimental",
                }))
        adjustment[lines | ~self._valid] = 0
        return adjustment, lines, events

    def advance(self, until_s: int, treatments: list[Treatment]) -> list[Event]:
        if isinstance(until_s, (bool, np.bool_)) or not isinstance(until_s, Integral):
            raise ValueError("fire advance requires an integer elapsed time")
        if not self.time_s <= until_s <= self.scenario.duration_s:
            raise ValueError("fire advance must be monotonic and inside the scenario horizon")
        if until_s != self.scenario.duration_s and until_s % COUPLING_S:
            raise ValueError("fire advances require canonical 60-second boundaries")
        if not isinstance(treatments, list) or any(not isinstance(t, Treatment) for t in treatments):
            raise ValueError("fire treatments must be a list of Treatment records")
        # Validate before any integration or event-queue mutation.
        if treatments and self._suppression is None:
            raise ValueError("treatments require provenance-tagged suppression parameters")
        events = self._events
        self._events = []
        while self.time_s < until_s:
            end_s = min(self.time_s + COUPLING_S, self.scenario.duration_s)
            self._align_state_clock()
            adjustment, lines, activation_events = self._modifiers(treatments)
            events.extend(activation_events)
            cubes = dict(self._cubes)
            cubes["fuel_spread_adjustment"] = self._cube(adjustment)
            spot_config = None
            if self._spot_config is not None:
                spot_config = dict(self._spot_config)
                spot_config["random_seed"] = int(np.random.SeedSequence(
                    [self.seed, self.time_s // COUPLING_S]).generate_state(1)[0])
            result = els.spread_fire_with_phi_field(
                space_time_cubes=cubes, spread_state=self._state,
                start_time=self.time_s / 60.0,
                max_duration=(end_s - self.time_s) / 60.0,
                max_cells_per_timestep=0.4,
                spot_config=spot_config,
                spot_ignitions={time: set(cells) for time, cells in self._spot_ignitions.items()},
                cube_refresh_rates={},
            )
            self._state = result["spread_state"]
            self._spot_ignitions = {float(time): set(cells)
                                    for time, cells in result.get("spot_ignitions", {}).items()}
            interval_start = self.time_s
            self.time_s = end_s  # Never inherit an early upstream stop as world time.
            arrivals = np.flip(self._state.get_full_matrices(["time_of_arrival"])["time_of_arrival"], 0)
            newly_burned = np.isfinite(arrivals) & (arrivals * 60.0 >= interval_start) & self._burnable
            breached = lines & newly_burned
            if breached.any():
                self._breached_line |= breached
                events.append(Event(float(end_s), "line_breached", None, {
                    "cells": np.argwhere(breached).tolist(),
                    "cause": "modeled_ignition_in_completed_line",
                    "suppression_validation": "experimental",
                }))
            reached = np.isfinite(arrivals) & self._burnable & self._coverage_edge
            if not self.domain_exited and reached.any():
                self.domain_exited = self.area_censored = True
                events.append(Event(float(np.nanmin(arrivals[reached]) * 60.0), "domain_exit", None, {
                    "area_censored": True, "reason": "fire reached landscape coverage boundary",
                    "cells": np.argwhere(reached).tolist(),
                }))
        return sorted(events, key=event_sort_key)

    def snapshot(self) -> dict[str, np.ndarray]:
        matrices = self._state.get_full_matrices()
        arrivals = np.flip(matrices["time_of_arrival"], 0)
        burned = np.isfinite(arrivals) & self._burnable
        # Burned-side, four-neighbor frontier on supported burnable cells. Domain
        # exits are tracked separately: an array edge is not permanent containment.
        frontier = np.zeros(self._valid.shape, dtype=bool)
        unburned = self._burnable & ~burned
        frontier[1:] |= burned[1:] & unburned[:-1]
        frontier[:-1] |= burned[:-1] & unburned[1:]
        frontier[:, 1:] |= burned[:, 1:] & unburned[:, :-1]
        frontier[:, :-1] |= burned[:, :-1] & unburned[:, 1:]
        result = {"burned": burned, "frontier": frontier}
        for public, upstream, scale, fill in (
            ("arrival_time_s", "time_of_arrival", 60.0, np.nan),
            ("fire_type", "fire_type", 1.0, 0),
            ("fireline_intensity", "fireline_intensity", 1.0, 0),
            ("spread_rate_m_s", "spread_rate", 1 / 60.0, 0),
            ("spread_direction_deg", "spread_direction", 1.0, np.nan),
            ("flame_length_m", "flame_length", 1.0, 0),
        ):
            values = np.flip(matrices[upstream], 0).copy()
            if scale != 1.0:
                values *= scale
            values[~self._burnable] = fill
            result[public] = values
        for array in result.values():
            array.setflags(write=False)
        return result
