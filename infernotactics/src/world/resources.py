"""Timestamped illustrative resource operations, independent of a policy.

Travel and work use elapsed seconds; effects are earned, never dispatch-time
edits to fire state. Line footprints rasterize only completed segment portions.
These configured research constraints are not operational dispatch authority.
"""
from __future__ import annotations

import copy
import math
from numbers import Real

import numpy as np

from .contracts import Action, Event, Scenario, Treatment, event_sort_key
from .contracts import validate_actions as validate_batch
from .routing import (check_air_constraints, line_access_point, plan_route,
                      route_invalid_reason, validate_ground_segment)

_EPS = 1e-9


def _time(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class ResourceEngine:
    """Own unit state and shared-base FIFO refill service.

    ``advance`` returns newly completed treatments, not the cumulative list.
    ``resource_time`` audit events integrate every nonavailable status, including
    suspended work and FIFO waits. Observation and validation do not advance time.
    """

    def __init__(self, scenario: Scenario):
        scenario.validate()
        self.scenario = scenario
        self.time_s = 0.0
        self._units = {unit["id"]: copy.deepcopy(unit) for unit in sorted(scenario.resources, key=lambda unit: unit["id"])}
        self._queues: dict[str, list[str]] = {}
        self._serving: dict[str, str] = {}
        for unit in sorted(self._units.values(), key=lambda unit: unit["id"]):
            unit.setdefault("capacity_l", 0.0)
            unit.setdefault("supply_l", unit["capacity_l"])
            unit.setdefault("water_rate_l_s", 0.0)
            unit.setdefault("line_rate_m_s", 0.0)
            unit.setdefault("refill_rate_l_s", 0.0)
            unit.setdefault("route_progress_m", 0.0)
            unit.setdefault("assignment", None)
            unit.setdefault("route", [])
            unit.update(remaining_work_s=0.0, work_progress_s=0.0, water_delivered_l=0.0,
                        line_progress_m=0.0, blocked=False, suspended=False,
                        refill_waiting=False, event_timestamps={})
            unit["_route_index"] = 0
            unit["_segment_elapsed_s"] = 0.0
            unit["_segment_start"] = [unit["x"], unit["y"]]
            # Scenario resources describe initial availability, not resumable jobs.
            if unit["status"] != "available" or unit["route"] or unit["assignment"] is not None:
                raise ValueError("resources must start available without a route or assignment")

    @classmethod
    def from_observation(cls, observation: dict, scenario: Scenario) -> "ResourceEngine":
        """Build a nonadvancing validator from public current operational state.

        No queues, hidden work clocks, future weather, or effect-law parameters
        are needed to validate a dispatch. This uses the same ``_prepare`` path
        as live submission, rather than a second policy feasibility model.
        """
        engine = cls.__new__(cls)
        engine.scenario = copy.copy(scenario)
        engine.scenario.landscape = observation["grid"]
        engine.scenario.weather = {
            key: np.asarray(value)[None].copy()
            for key, value in observation["weather"].items()
        }
        engine.scenario.weather_step_s = scenario.duration_s + 1
        operations = observation["operations"]
        engine.scenario.provenance = {
            "suppression": {"source": "public configured work footprint",
                            "radius_m": operations["water_radius_m"]},
            "air_corridor_available": operations["air_corridor_available"],
        }
        engine.scenario.roads = observation["roads"]
        engine.time_s = observation["time_s"]
        engine._units = {unit["id"]: copy.deepcopy(unit) for unit in observation["units"]}
        return engine

    def feasible_eta(self, action: Action, time_s: float) -> float | None:
        """Seconds until work/return arrival under current observed conditions."""
        prepared = self._prepare([action], time_s)
        if action.operation == "wait":
            return None
        unit = self._units[action.unit_id]
        mobilization = unit.get("mobilization_s", 0.0) if action.operation != "return" else 0.0
        return float(mobilization + sum(segment["seconds"] for segment in prepared[0][1]["route"]))

    def _at_time(self, time_s: float) -> float:
        time_s = _time(time_s, "time_s")
        if time_s != self.time_s:
            raise ValueError("actions must be submitted at the current resource time")
        if time_s >= self.scenario.duration_s:
            raise ValueError("cannot submit actions at or after scenario horizon")
        return time_s

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        a, _, c, _, e, f = self.scenario.transform
        row, col = math.floor((y - f) / e), math.floor((x - c) / a)
        valid = self.scenario.landscape["valid"]
        if not (0 <= row < valid.shape[0] and 0 <= col < valid.shape[1]) or not valid[row, col]:
            raise ValueError("work point is outside the valid landscape")
        return row, col

    def _line_cells(self, x0: float, y0: float, x1: float, y1: float) -> list[dict]:
        """Exact grid crossings; each cell becomes complete at its segment exit."""
        a, _, c, _, e, f = self.scenario.transform
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance <= _EPS:
            return []
        cuts = {0.0, 1.0}
        for v0, v1, origin, spacing in ((x0, x1, c, a), (y0, y1, f, e)):
            p0, p1 = (v0 - origin) / spacing, (v1 - origin) / spacing
            if p0 != p1:
                for grid in range(math.floor(min(p0, p1)) + 1, math.ceil(max(p0, p1))):
                    t = (grid - p0) / (p1 - p0)
                    if 0 < t < 1:
                        cuts.add(t)
        cuts = sorted(cuts)
        cells = []
        for start, end in zip(cuts, cuts[1:]):
            mid = (start + end) / 2
            x, y = x0 + (x1 - x0) * mid, y0 + (y1 - y0) * mid
            row, col = self._cell(x, y)
            fuel = self.scenario.landscape["fuel_model"][row, col]
            if fuel in (91, 92, 93, 98, 99):
                raise ValueError("line segment crosses nonburnable or water terrain")
            cells.append({"x": x, "y": y, "distance_m": end * distance,
                          "row": row, "col": col})
        return cells

    def _water_footprint(self, x: float, y: float) -> tuple[float, float, int]:
        config = self.scenario.provenance.get("suppression", {})
        if not isinstance(config.get("source"), str) or not config["source"].strip():
            raise ValueError("water work requires provenance-tagged suppression parameters")
        radius = _time(config.get("radius_m"), "suppression.radius_m")
        if radius <= 0:
            raise ValueError("suppression.radius_m must be positive")
        a, _, c, _, e, f = self.scenario.transform
        valid = self.scenario.landscape["valid"]
        xs = c + (np.arange(valid.shape[1]) + 0.5) * a
        ys = f + (np.arange(valid.shape[0]) + 0.5) * e
        footprint = ((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2 <= radius ** 2)
        footprint &= valid & ~np.isin(self.scenario.landscape["fuel_model"], [91, 92, 93, 98, 99])
        count = int(footprint.sum())
        if count == 0:
            raise ValueError("water footprint contains no valid burnable cells")
        return radius, count * abs(a * e), count

    def _prepare(self, actions: list[Action], time_s: float) -> list[tuple[Action, dict]]:
        self._at_time(time_s)
        validate_batch(actions, self._units)
        prepared = []
        for action in actions:
            if action.operation == "wait":
                continue
            unit = self._units[action.unit_id]
            if action.operation == "return":
                if unit["status"] in {"available", "refilling"}:
                    raise ValueError("return requires an active unit away from refill service")
                route = plan_route(self.scenario, copy.deepcopy(unit), unit["home_x"], unit["home_y"], time_s)
                prepared.append((action, {"route": route}))
                continue
            if unit["status"] != "available":
                raise ValueError(f"unit {unit['id']} is busy")
            if time_s >= unit["shift_end_s"]:
                raise ValueError("unit shift has ended")
            if action.operation == "line" and unit["kind"] != "handcrew":
                raise ValueError("only handcrews can construct line")
            if action.operation == "water" and unit["kind"] == "handcrew":
                raise ValueError("handcrew supply is not water")
            rate = unit["line_rate_m_s"] if action.operation == "line" else unit["water_rate_l_s"]
            if rate <= 0 or unit["work_duration_s"] <= 0:
                raise ValueError("operation requires positive work rate and duration")
            if action.operation == "water" and unit["supply_l"] <= 0:
                raise ValueError("water supply is depleted")
            self._cell(action.x, action.y)
            data = {}
            x, y = action.x, action.y
            if action.operation == "line":
                x, y = line_access_point(self.scenario, copy.deepcopy(unit), x, y, time_s)
                length = math.hypot(action.x - x, action.y - y)
                if length <= _EPS:
                    raise ValueError("line target must define a nonzero segment from road access")
                length = min(length, rate * unit["work_duration_s"])
                ratio = length / math.hypot(action.x - x, action.y - y)
                end_x, end_y = x + ratio * (action.x - x), y + ratio * (action.y - y)
                validate_ground_segment(self.scenario, unit, x, y, end_x, end_y)
                cells = self._line_cells(x, y, end_x, end_y)
                data.update(line_start=[x, y], line_end=[end_x, end_y], line_length_m=length,
                            line_cells=cells, duration_s=length / rate)
            else:
                data["duration_s"] = min(unit["work_duration_s"], unit["supply_l"] / rate)
                radius, area, count = self._water_footprint(x, y)
                data.update(radius_m=radius, area_m2=area, footprint_cells=count)
            route = plan_route(self.scenario, copy.deepcopy(unit), x, y, time_s)
            data["route"] = route
            prepared.append((action, data))
        return prepared

    def validate_actions(self, actions: list[Action], time_s: float) -> None:
        self._prepare(actions, time_s)

    def _event(self, unit: dict, kind: str, details: dict | None = None) -> Event:
        unit["event_timestamps"][kind] = self.time_s
        return Event(float(self.time_s), kind, unit["id"], details or {})

    def _set_route(self, unit: dict, route: list[dict]) -> None:
        unit["route"] = copy.deepcopy(route)
        unit["route_progress_m"] = 0.0
        unit["_route_index"] = 0
        unit["_segment_elapsed_s"] = 0.0
        unit["_segment_start"] = [unit["x"], unit["y"]]
        unit["blocked"] = False

    def submit(self, actions: list[Action], time_s: float) -> list[Event]:
        prepared = self._prepare(actions, time_s)  # All feasibility before any mutation.
        events = []
        for action, data in prepared:
            unit = self._units[action.unit_id]
            if action.operation == "return":
                # Publish accumulated water at interruption, never lose earned work.
                if unit["status"] == "working":
                    unit["_publish_water"] = True
                unit["assignment"] = {"operation": "return", "x": unit["home_x"], "y": unit["home_y"]}
                unit["status"] = "returning"
                unit["suspended"] = False
                self._set_route(unit, data["route"])
                events.append(self._event(unit, "return_started", {"interrupted": True, "x": unit["x"], "y": unit["y"]}))
            else:
                unit["assignment"] = {"operation": action.operation, "x": action.x, "y": action.y}
                unit["_work"] = data
                unit["_work_elapsed_s"] = 0.0
                unit["_water_earned_l"] = 0.0
                unit["_line_next_cell"] = 0
                unit["remaining_work_s"] = data["duration_s"]
                unit["work_progress_s"] = 0.0
                unit["water_delivered_l"] = 0.0
                unit["line_progress_m"] = 0.0
                unit["status"] = "mobilizing"
                unit["_mobilization_end_s"] = self.time_s + unit["mobilization_s"]
                unit["suspended"] = False
                self._set_route(unit, data["route"])
                events.append(self._event(unit, "dispatch", dict(unit["assignment"])))
        return sorted(events, key=event_sort_key)

    def _safety_reason(self, unit: dict) -> str | None:
        if self.time_s >= unit["shift_end_s"]:
            return "shift_ended"
        if unit["kind"] == "helicopter":
            try:
                check_air_constraints(self.scenario, unit, self.time_s, unit["x"], unit["y"])
            except ValueError as exc:
                return str(exc)
        return None

    def _remaining_route(self, unit: dict) -> list[dict]:
        route = copy.deepcopy(unit["route"][unit["_route_index"]:])
        if route:
            route[0]["start_x"], route[0]["start_y"] = unit["x"], unit["y"]
            seconds = route[0]["seconds"]
            remaining = max(0.0, seconds - unit["_segment_elapsed_s"])
            route[0]["length_m"] *= remaining / seconds if seconds else 0.0
            route[0]["seconds"] = remaining
        return route

    def _boundary(self, events: list[Event]) -> None:
        for unit in self._units.values():
            if unit["status"] in {"traveling", "returning", "mobilizing"}:
                remaining = self._remaining_route(unit)
                reason = route_invalid_reason(self.scenario, unit, remaining, self.time_s)
                if reason or unit["blocked"]:
                    if unit["status"] == "returning":
                        x, y = unit["home_x"], unit["home_y"]
                    else:
                        work = unit["_work"]
                        x, y = (work["line_start"] if "line_start" in work else
                                (unit["assignment"]["x"], unit["assignment"]["y"]))
                    try:
                        route = plan_route(self.scenario, unit, x, y, self.time_s)
                    except ValueError as exc:
                        if not unit["blocked"]:
                            events.append(self._event(unit, "route_blocked", {"reason": str(exc), "x": unit["x"], "y": unit["y"]}))
                        unit["blocked"] = True
                    else:
                        self._set_route(unit, route)
                        events.append(self._event(unit, "route_replanned", {"reason": reason or "route_available"}))
            elif unit["status"] == "working":
                reason = self._safety_reason(unit)
                if reason and not unit["suspended"]:
                    unit["suspended"] = True
                    events.append(self._event(unit, "work_suspended", {"reason": reason}))
                elif not reason and unit["suspended"]:
                    unit["suspended"] = False
                    events.append(self._event(unit, "work_resumed"))

    def _water(self, unit: dict, treatments: list[Treatment], events: list[Event]) -> None:
        earned = unit.get("_water_earned_l", 0.0)
        if earned > _EPS:
            work = unit["_work"]
            x, y = work["water_point"]
            treatment = Treatment(float(x), float(y), work["radius_m"], "water",
                                  float(earned / work["area_m2"]),
                                  float(self.scenario.duration_s), float(self.time_s))
            treatments.append(treatment)
            events.append(self._event(unit, "treatment_completed", {
                "kind": "water", "litres": earned, "delivered_l": earned,
                "strength_l_m2": treatment.strength, "radius_m": treatment.radius_m,
                "footprint_cells": work["footprint_cells"], "area_m2": work["area_m2"],
                "x": x, "y": y}))
            unit["_water_earned_l"] = 0.0
        unit["_publish_water"] = False

    def _start_return(self, unit: dict, events: list[Event]) -> None:
        unit["assignment"] = {"operation": "return", "x": unit["home_x"], "y": unit["home_y"]}
        unit["status"] = "returning"
        unit["suspended"] = False
        try:
            route = plan_route(self.scenario, unit, unit["home_x"], unit["home_y"], self.time_s)
        except ValueError as exc:
            self._set_route(unit, [])
            unit["blocked"] = True
            events.append(self._event(unit, "route_blocked", {"reason": str(exc), "x": unit["x"], "y": unit["y"]}))
        else:
            self._set_route(unit, route)
            events.append(self._event(unit, "return_started", {"interrupted": False}))

    def _available(self, unit: dict, events: list[Event]) -> None:
        unit["status"] = "available"
        unit["assignment"] = None
        unit["remaining_work_s"] = 0.0
        unit["refill_waiting"] = False
        self._set_route(unit, [])
        events.append(self._event(unit, "available"))

    def _settle(self, events: list[Event], treatments: list[Treatment]) -> None:
        # Each loop consumes a route segment, work completion, or service state;
        # no zero-time cycle can dispatch an available unit again.
        changed = True
        while changed:
            changed = False
            for unit in self._units.values():
                if unit.get("_publish_water"):
                    self._water(unit, treatments, events)
                if unit["status"] == "mobilizing" and not unit["blocked"] and self.time_s + _EPS >= unit["_mobilization_end_s"]:
                    unit["status"] = "traveling"
                    events.append(self._event(unit, "travel_started"))
                    changed = True
                if unit["status"] in {"traveling", "returning"} and not unit["blocked"]:
                    route = unit["route"]
                    while unit["_route_index"] < len(route):
                        segment = route[unit["_route_index"]]
                        if unit["_segment_elapsed_s"] + _EPS < segment["seconds"]:
                            break
                        unit["x"], unit["y"] = segment["x"], segment["y"]
                        unit["_route_index"] += 1
                        unit["_segment_elapsed_s"] = 0.0
                        unit["_segment_start"] = [unit["x"], unit["y"]]
                        changed = True
                    if unit["_route_index"] == len(route):
                        if unit["status"] == "traveling":
                            unit["status"] = "working"
                            unit["_work"]["water_point"] = [unit["assignment"]["x"], unit["assignment"]["y"]]
                            events.append(self._event(unit, "arrival", {"x": unit["x"], "y": unit["y"]}))
                            events.append(self._event(unit, "work_started"))
                            reason = self._safety_reason(unit)
                            if reason:
                                unit["suspended"] = True
                                events.append(self._event(unit, "work_suspended", {"reason": reason}))
                        else:
                            events.append(self._event(unit, "home_arrival"))
                            if unit["capacity_l"] - unit["supply_l"] <= _EPS:
                                self._available(unit, events)
                            else:
                                unit["status"] = "refilling"
                                unit["refill_waiting"] = True
                                self._queues.setdefault(unit["base_id"], []).append(unit["id"])
                                events.append(self._event(unit, "refill_queued"))
                        changed = True
                if unit["status"] == "working" and not unit["suspended"] and unit["remaining_work_s"] <= _EPS:
                    operation = unit["assignment"]["operation"]
                    if operation == "water":
                        self._water(unit, treatments, events)
                    events.append(self._event(unit, "work_completed", {"operation": operation, "elapsed_work_s": unit["_work_elapsed_s"]}))
                    self._start_return(unit, events)
                    changed = True
                if unit["status"] == "refilling" and not unit["refill_waiting"] and unit["capacity_l"] - unit["supply_l"] <= _EPS:
                    unit["supply_l"] = unit["capacity_l"]
                    self._serving.pop(unit["base_id"], None)
                    events.append(self._event(unit, "refill_completed"))
                    self._available(unit, events)
                    changed = True
            for base, queue in self._queues.items():
                if queue and base not in self._serving:
                    # Arrival order first; simultaneous arrivals use stable unit ID.
                    unit_id = queue.pop(0)
                    unit = self._units[unit_id]
                    self._serving[base] = unit_id
                    unit["refill_waiting"] = False
                    events.append(self._event(unit, "refill_started"))
                    changed = True

    def _duration(self, unit: dict) -> float:
        status = unit["status"]
        if status == "mobilizing" and not unit["blocked"]:
            return max(0.0, unit["_mobilization_end_s"] - self.time_s)
        if status in {"traveling", "returning"} and not unit["blocked"]:
            if unit["_route_index"] < len(unit["route"]):
                return max(0.0, unit["route"][unit["_route_index"]]["seconds"] - unit["_segment_elapsed_s"])
        if status == "working" and not unit["suspended"]:
            return unit["remaining_work_s"]
        if status == "refilling" and not unit["refill_waiting"] and unit["refill_rate_l_s"] > 0:
            return (unit["capacity_l"] - unit["supply_l"]) / unit["refill_rate_l_s"]
        return math.inf

    def _elapse(self, unit: dict, dt: float, treatments: list[Treatment], events: list[Event]) -> None:
        status = unit["status"]
        if status in {"traveling", "returning"} and not unit["blocked"]:
            segment = unit["route"][unit["_route_index"]]
            unit["_segment_elapsed_s"] += dt
            fraction = min(1.0, unit["_segment_elapsed_s"] / segment["seconds"]) if segment["seconds"] else 1.0
            x0, y0 = unit["_segment_start"]
            unit["x"] = x0 + fraction * (segment["x"] - x0)
            unit["y"] = y0 + fraction * (segment["y"] - y0)
            unit["route_progress_m"] += segment["length_m"] * dt / segment["seconds"] if segment["seconds"] else 0.0
        elif status == "working" and not unit["suspended"]:
            unit["remaining_work_s"] = max(0.0, unit["remaining_work_s"] - dt)
            old_elapsed = unit["_work_elapsed_s"]
            unit["_work_elapsed_s"] += dt
            unit["work_progress_s"] = unit["_work_elapsed_s"]
            if unit["assignment"]["operation"] == "water":
                litres = min(unit["supply_l"], dt * unit["water_rate_l_s"])
                unit["supply_l"] = max(0.0, unit["supply_l"] - litres)
                unit["_water_earned_l"] += litres
                unit["water_delivered_l"] += litres
            else:
                work = unit["_work"]
                distance = min(work["line_length_m"], unit["_work_elapsed_s"] * unit["line_rate_m_s"])
                unit["line_progress_m"] = distance
                fraction = distance / work["line_length_m"]
                unit["x"] = work["line_start"][0] + fraction * (work["line_end"][0] - work["line_start"][0])
                unit["y"] = work["line_start"][1] + fraction * (work["line_end"][1] - work["line_start"][1])
                while unit["_line_next_cell"] < len(work["line_cells"]):
                    cell = work["line_cells"][unit["_line_next_cell"]]
                    if cell["distance_m"] > distance + _EPS:
                        break
                    completed = self.time_s + cell["distance_m"] / unit["line_rate_m_s"] - old_elapsed
                    treatment = Treatment(float(cell["x"]), float(cell["y"]), 0.0, "line", 1.0,
                                          float(self.scenario.duration_s), float(completed))
                    treatments.append(treatment)
                    events.append(Event(float(completed), "treatment_completed", unit["id"],
                                        {"kind": "line", "distance_m": cell["distance_m"], "x": cell["x"], "y": cell["y"]}))
                    unit["event_timestamps"]["treatment_completed"] = float(completed)
                    unit["_line_next_cell"] += 1
        elif status == "refilling" and not unit["refill_waiting"]:
            unit["supply_l"] = min(unit["capacity_l"], unit["supply_l"] + dt * unit["refill_rate_l_s"])

    def advance(self, until_s: float) -> tuple[list[Event], list[Treatment]]:
        until_s = _time(until_s, "until_s")
        if until_s < self.time_s or until_s > self.scenario.duration_s:
            raise ValueError("resource advance must be monotone and within scenario horizon")
        events: list[Event] = []
        treatments: list[Treatment] = []
        busy = {unit_id: 0.0 for unit_id in self._units}
        self._settle(events, treatments)
        # A changed road/weather input is observed at canonical boundaries only.
        if self.time_s % 60 == 0:
            self._boundary(events)
            self._settle(events, treatments)
        while self.time_s < until_s:
            boundary = min(until_s, (math.floor(self.time_s / 60) + 1) * 60.0)
            dt = min(boundary - self.time_s, *(self._duration(unit) for unit in self._units.values())) if self._units else boundary - self.time_s
            if dt <= _EPS:
                # Numerical residue is completion, not another physical interval.
                dt = min(boundary - self.time_s, _EPS)
            for unit_id, unit in self._units.items():
                if unit["status"] != "available":
                    busy[unit_id] += dt
                self._elapse(unit, dt, treatments, events)
            self.time_s += dt
            self._settle(events, treatments)
            if self.time_s % 60 == 0:
                self._boundary(events)
                self._settle(events, treatments)
        for unit_id, duration in busy.items():
            if duration:
                events.append(Event(float(self.time_s), "resource_time", unit_id, {"resource_time_s": duration}))
        return sorted(events, key=event_sort_key), sorted(treatments, key=lambda treatment: (treatment.time_s, treatment.kind, treatment.x, treatment.y))

    def snapshot(self) -> list[dict]:
        return [{key: copy.deepcopy(value) for key, value in unit.items() if not key.startswith("_")}
                for unit in sorted(self._units.values(), key=lambda unit: unit["id"])]
