"""Canonical minute coupling and owned, present-only policy observations.

Completed resource effects enter the *next* fire interval, never the interval
in which they were earned. A boundary/domain exit censors the episode; a quiet
minute is only a no-new-growth duration, not a containment declaration.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .contracts import Action, Scenario, event_sort_key


class World:
    def __init__(self, scenario: Scenario, seed: int = 0):
        scenario.validate()
        from .fire import FireEngine
        from .resources import ResourceEngine
        self.scenario = deepcopy(scenario)
        self.seed = seed
        self.fire = FireEngine(self.scenario, seed)
        self.resources = ResourceEngine(self.scenario)
        self.time_s = 0
        self.events = []
        self.treatments = []
        self.no_new_growth_s = 0
        self._domain_exit = False
        fuels = self.scenario.landscape["fuel_model"]
        self.valid_burnable_m2 = float(np.count_nonzero(
            self.scenario.landscape["valid"] & (fuels >= 101) & (fuels <= 204)
        ) * self.scenario.transform[0] ** 2)
        self.total_asset_m2 = float(self.scenario.landscape["asset_value"][
            self.scenario.landscape["valid"]].sum())

    @property
    def done(self) -> bool:
        return self._domain_exit or self.time_s >= self.scenario.duration_s

    def ignite(self, longitude: float, latitude: float) -> None:
        if self.done:
            raise ValueError("episode already ended")
        self.fire.ignite(longitude, latitude)

    def step(self, actions: list[Action]) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise ValueError("episode already ended")
        # submit prepares and validates the complete batch before any mutation.
        events = self.resources.submit(actions, self.time_s)
        before = self.fire.snapshot()["burned"]
        until = min(self.time_s + 60, self.scenario.duration_s)
        events += self.fire.advance(until, self.treatments)
        resource_events, completed = self.resources.advance(until)
        events += resource_events
        self.treatments.extend(completed)
        self.treatments.sort(key=lambda item: (item.time_s, item.kind, item.x, item.y))
        new = self.fire.snapshot()["burned"] & ~before
        dt = until - self.time_s
        self.no_new_growth_s = 0 if new.any() else self.no_new_growth_s + dt
        burned = float(new.sum() * self.scenario.transform[0] ** 2)
        assets = float(self.scenario.landscape["asset_value"][new].sum())
        # These audit integrals account for every nonavailable status, including
        # partial travel/work, FIFO waits, and suspended/blocked units.
        resource_time_s = sum(float(event.details["resource_time_s"]) for event in resource_events
                              if event.kind == "resource_time")
        reward = -(burned / self.valid_burnable_m2) if self.valid_burnable_m2 else 0.0
        reward -= assets / max(self.total_asset_m2, 1.0)
        reward -= .01 * resource_time_s / max(len(self.scenario.resources) * self.scenario.duration_s, 1)
        events.sort(key=event_sort_key)
        self.events.extend(events)
        self.events.sort(key=event_sort_key)
        self.time_s = until
        self._domain_exit |= any(event.kind == "domain_exit" for event in events)
        info = {"events": deepcopy(events), "new_burned_m2": burned,
                "new_asset_exposure_m2": assets, "resource_time_s": resource_time_s,
                "domain_exit": self._domain_exit, "censored": self._domain_exit,
                "no_new_growth_s": self.no_new_growth_s}
        return self.observe(), float(reward), self.done, info

    def observe(self) -> dict:
        from .actions import _candidate_data
        band = min(self.time_s // self.scenario.weather_step_s,
                   next(iter(self.scenario.weather.values())).shape[0] - 1)
        fire = self.fire.snapshot()
        # All arrays and nested operational values are independently owned by
        # this observation. No retained physics state or future bands escape.
        grid = {key: value.copy() for key, value in self.scenario.landscape.items()}
        grid.update({key: value.copy() for key, value in fire.items()})
        suppression = self.scenario.provenance.get("suppression", {})
        provenance = self.scenario.provenance
        public_provenance = {
            "sources": {key: value for key, value in provenance.get("sources", {}).items()
                        if isinstance(value, str)},
            "source_hashes": deepcopy(provenance.get("source_hashes", {})),
            "resolution_m": self.scenario.transform[0],
            "observation_mode": "oracle_current",
            "suppression_validation": "experimental",
        }
        for key in ("historical_validation", "spread_validation", "synthetic"):
            value = provenance.get(key)
            if isinstance(value, (str, bool)):
                public_provenance[key] = value
        observation = {
            "time_s": self.time_s, "scenario": self.scenario.name, "seed": self.seed,
            "grid": grid, "fire": fire,
            "weather": {key: value[band].copy() for key, value in self.scenario.weather.items()},
            "provenance": public_provenance,
            "units": self.resources.snapshot(), "roads": deepcopy(self.scenario.roads),
            "operations": {"water_radius_m": suppression.get("radius_m"),
                           "air_corridor_available": self.scenario.provenance.get(
                               "air_corridor_available", True)},
            "events": deepcopy(self.events),
            "no_new_growth_s": self.no_new_growth_s,
            "burned_area_m2": float(fire["burned"].sum() * self.scenario.transform[0] ** 2),
            "asset_exposure_m2": float(self.scenario.landscape["asset_value"][fire["burned"]].sum()),
            "domain_exit": self._domain_exit, "censored": self._domain_exit,
            "done": self.done,
        }
        candidates, metadata = _candidate_data(observation, self.scenario)
        observation["candidates"] = candidates
        observation["candidate_mask"] = [True] * len(candidates)
        observation["candidate_metadata"] = metadata
        return observation
