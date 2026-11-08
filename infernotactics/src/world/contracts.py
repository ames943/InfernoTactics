"""Schema-one contracts in projected metres, elapsed seconds, and UTC.

Validity is explicit: nodata cells may contain sentinels, but valid landscape
cells and their weather must be finite. Weather intervals are half-open UTC
intervals in ``provenance['weather_intervals']``; the final interval may extend
past the scenario horizon. Provenance labels do not certify scientific validity.
"""

from collections.abc import Collection
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
from numbers import Real

import numpy as np
from pyproj import CRS

SCHEMA_VERSION = 1
LANDSCAPE_KEYS = (
    "elevation", "slope", "aspect", "fuel_model", "canopy_cover",
    "canopy_height", "canopy_base_height", "canopy_bulk_density", "valid",
    "asset_value",
)
WEATHER_KEYS = (
    "wind_speed_m_s", "upwind_direction_deg", "temperature_c",
    "fuel_moisture_dead_1hr", "fuel_moisture_dead_10hr",
    "fuel_moisture_dead_100hr", "fuel_moisture_live_herbaceous",
    "fuel_moisture_live_woody", "foliar_moisture",
)
OPERATIONS = frozenset({"wait", "water", "line", "return"})
UNIT_STATUSES = frozenset({
    "available", "mobilizing", "traveling", "working", "returning", "refilling",
})


def _text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _number(value: object, name: str, minimum: float | None = None) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")


def _integer(value: object, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _json(value: object, name: str) -> None:
    """Require actual JSON types, not Python objects silently coerced by JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
    elif isinstance(value, list):
        for item in value:
            _json(item, name)
        return
    elif isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json(item, name)
        return
    raise ValueError(f"{name} must contain only finite JSON values with string keys")


def _utc(value: object, name: str) -> datetime:
    _text(value, name)
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp") from exc
    if result.tzinfo is None or result.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must explicitly specify UTC")
    return result


@dataclass(frozen=True)
class Action:
    unit_id: str | None = None
    operation: str = "wait"
    x: float | None = None
    y: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.operation, str) or self.operation not in OPERATIONS:
            raise ValueError(f"unknown operation: {self.operation!r}")
        if self.unit_id is not None:
            _text(self.unit_id, "action.unit_id")
        if self.operation != "wait" and self.unit_id is None:
            raise ValueError(f"{self.operation} requires a unit_id")
        for name in ("x", "y"):
            value = getattr(self, name)
            if value is not None:
                _number(value, f"action.{name}")
        if self.operation in {"water", "line"} and (self.x is None or self.y is None):
            raise ValueError(f"{self.operation} requires finite projected x and y")


def validate_actions(actions: list[Action], unit_ids: Collection[str] | None = None) -> None:
    """Validate an entire batch without mutation; engines add feasibility checks."""
    if not isinstance(actions, list):
        raise ValueError("actions must be a list of Action objects")
    assigned = set()
    for action in actions:
        if not isinstance(action, Action):
            raise ValueError("each action must be an Action")
        action.validate()
        if action.unit_id is not None:
            if unit_ids is not None and action.unit_id not in unit_ids:
                raise ValueError(f"unknown unit: {action.unit_id}")
            if action.unit_id in assigned:
                raise ValueError(f"duplicate unit assignment: {action.unit_id}")
            assigned.add(action.unit_id)


@dataclass(frozen=True)
class Event:
    time_s: float
    kind: str
    unit_id: str | None
    details: dict

    def __post_init__(self) -> None:
        _number(self.time_s, "event.time_s", 0)
        _text(self.kind, "event.kind")
        if self.unit_id is not None:
            _text(self.unit_id, "event.unit_id")
        if not isinstance(self.details, dict):
            raise ValueError("event.details must be a dict")
        _json(self.details, "event.details")
        # Check scalar fields as well: audit records must be directly JSON-safe.
        _json(self.time_s, "event.time_s")


def event_sort_key(event: Event) -> tuple[float, str, str]:
    """Canonical event ordering; global (None-unit) events sort first."""
    return event.time_s, event.unit_id or "", event.kind


@dataclass(frozen=True)
class Treatment:
    x: float
    y: float
    radius_m: float
    kind: str
    strength: float
    expires_s: float
    time_s: float

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            _number(getattr(self, name), f"treatment.{name}")
        for name in ("radius_m", "strength", "expires_s", "time_s"):
            _number(getattr(self, name), f"treatment.{name}", 0)
        if not isinstance(self.kind, str) or self.kind not in {"water", "line"}:
            raise ValueError("treatment.kind must be water or line")
        if self.expires_s < self.time_s:
            raise ValueError("treatment expires before its completion time")
        for name in ("x", "y", "radius_m", "strength", "expires_s", "time_s"):
            _json(getattr(self, name), f"treatment.{name}")


@dataclass
class Scenario:
    name: str
    crs: str
    transform: tuple[float, float, float, float, float, float]
    start_utc: str
    duration_s: int
    landscape: dict[str, np.ndarray]
    weather: dict[str, np.ndarray]
    weather_step_s: int
    roads: dict
    resources: list[dict]
    provenance: dict
    seed: int = 0

    def validate(self) -> None:
        _text(self.name, "scenario.name")
        _text(self.crs, "scenario.crs")
        try:
            crs = CRS.from_user_input(self.crs)
        except Exception as exc:
            raise ValueError("scenario.crs is not a recognized CRS") from exc
        if not crs.is_projected or len(crs.axis_info) < 2 or any(
            not math.isclose(axis.unit_conversion_factor, 1.0, rel_tol=0, abs_tol=1e-12)
            or axis.unit_name.lower() not in {"metre", "meter", "metres", "meters"}
            for axis in crs.axis_info[:2]
        ):
            raise ValueError("scenario.crs must be projected with metre units")
        if not isinstance(self.transform, tuple) or len(self.transform) != 6:
            raise ValueError("transform must be a six-element affine (a,b,c,d,e,f) tuple")
        for value in self.transform:
            _number(value, "transform")
        a, b, _, d, e, _ = self.transform
        if a <= 0 or e >= 0 or b != 0 or d != 0 or not math.isclose(a, -e, rel_tol=1e-12):
            raise ValueError("transform must describe square north-up cells")
        start = _utc(self.start_utc, "start_utc")
        _integer(self.duration_s, "duration_s", 1)
        _integer(self.weather_step_s, "weather_step_s", 1)
        _integer(self.seed, "seed", 0)
        if not isinstance(self.provenance, dict) or not self.provenance:
            raise ValueError("provenance must be a nonempty source-provenance dict")
        _json(self.provenance, "provenance")
        sources = self.provenance.get("sources")
        if not isinstance(sources, dict) or not sources:
            raise ValueError("provenance.sources must document the input sources")
        for key, source in sources.items():
            if not source:
                raise ValueError(f"provenance.sources.{key} must identify a source")
        hashes = self.provenance.get("source_hashes", {})
        if not isinstance(hashes, dict):
            raise ValueError("provenance.source_hashes must be a dict")
        for key, digest in hashes.items():
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                raise ValueError(f"provenance source hash {key} must be SHA-256 hex")
        self._validate_landscape()
        self._validate_weather(start)
        validate_roads(self.roads)
        validate_resources(self.resources)

    def _validate_landscape(self) -> None:
        if not isinstance(self.landscape, dict) or not set(LANDSCAPE_KEYS).issubset(self.landscape):
            raise ValueError(f"landscape requires keys {LANDSCAPE_KEYS}")
        valid = self.landscape["valid"]
        if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.ndim != 2 or 0 in valid.shape:
            raise ValueError("landscape.valid must be a nonempty two-dimensional bool array")
        if not valid.any():
            raise ValueError("landscape has no valid cells")
        for key, array in self.landscape.items():
            _text(key, "landscape array name")
            if not isinstance(array, np.ndarray) or array.shape != valid.shape or array.dtype.kind not in "biuf":
                raise ValueError(f"landscape.{key} must be a numeric/bool array aligned with valid")
            if not np.isfinite(array[valid]).all():
                raise ValueError(f"landscape.{key} has nonfinite data in valid cells")
        fuel = self.landscape["fuel_model"]
        if fuel.dtype.kind not in "iu" or (fuel[valid] <= 0).any():
            raise ValueError("fuel_model must contain positive integer fuel codes in valid cells")
        for key in ("slope", "canopy_height", "canopy_base_height", "canopy_bulk_density", "asset_value"):
            if (self.landscape[key][valid] < 0).any():
                raise ValueError(f"landscape.{key} cannot be negative in valid cells")
        for key, maximum, inclusive in (("aspect", 360, False), ("canopy_cover", 1, True)):
            values = self.landscape[key][valid]
            if (values < 0).any() or ((values > maximum) if inclusive else (values >= maximum)).any():
                raise ValueError(f"landscape.{key} is outside its physical range")
        if (self.landscape["canopy_base_height"][valid] > self.landscape["canopy_height"][valid]).any():
            raise ValueError("canopy_base_height cannot exceed canopy_height")

    def _validate_weather(self, start: datetime) -> None:
        if not isinstance(self.weather, dict) or not set(WEATHER_KEYS).issubset(self.weather):
            raise ValueError(f"weather requires keys {WEATHER_KEYS}")
        valid = self.landscape["valid"]
        bands = None
        for key, array in self.weather.items():
            _text(key, "weather array name")
            if not isinstance(array, np.ndarray) or array.dtype.kind not in "iuf" or array.ndim != 3:
                raise ValueError(f"weather.{key} must be a three-dimensional numeric array")
            if array.shape[0] < 1 or array.shape[1:] not in (valid.shape, (1, 1)):
                raise ValueError(f"weather.{key} must have shape (time,height,width) or (time,1,1)")
            if bands is None:
                bands = array.shape[0]
            elif bands != array.shape[0]:
                raise ValueError("weather arrays must share the same time dimension")
            values = array[:, valid] if array.shape[1:] == valid.shape else array.ravel()
            if not np.isfinite(values).all():
                raise ValueError(f"weather.{key} is nonfinite over valid cells")
            if key != "temperature_c" and (values < 0).any():
                raise ValueError(f"weather.{key} cannot be negative")
            if key == "upwind_direction_deg" and (values >= 360).any():
                raise ValueError("upwind_direction_deg must be in [0,360)")
        intervals = self.provenance.get("weather_intervals")
        if not isinstance(intervals, list) or len(intervals) != bands:
            raise ValueError("provenance.weather_intervals must contain one UTC interval per band")
        previous = start
        for index, interval in enumerate(intervals):
            if not isinstance(interval, dict):
                raise ValueError("weather interval must contain start_utc and end_utc")
            begin = _utc(interval.get("start_utc"), f"weather interval {index} start_utc")
            end = _utc(interval.get("end_utc"), f"weather interval {index} end_utc")
            if begin != previous or (end - begin).total_seconds() != self.weather_step_s:
                raise ValueError("weather intervals must be contiguous, start at start_utc, and match weather_step_s")
            previous = end
        if (previous - start).total_seconds() < self.duration_s:
            raise ValueError("weather does not cover the full scenario duration")


def validate_roads(roads: dict) -> None:
    if not isinstance(roads, dict) or not isinstance(roads.get("nodes"), list) or not isinstance(roads.get("edges"), list):
        raise ValueError("roads must contain nodes and directed edges lists")
    _json(roads, "roads")
    nodes = {}
    for node in roads["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("road node must be a dict")
        node_id = node.get("id")
        _text(node_id, "road node id")
        if node_id in nodes:
            raise ValueError(f"duplicate road node: {node_id}")
        _number(node.get("x"), "road node x")
        _number(node.get("y"), "road node y")
        nodes[node_id] = (node["x"], node["y"])
    edge_ids = set()
    for edge in roads["edges"]:
        if not isinstance(edge, dict):
            raise ValueError("road edge must be a dict")
        edge_id = edge.get("id")
        _text(edge_id, "road edge id")
        if edge_id in edge_ids:
            raise ValueError(f"duplicate road edge: {edge_id}")
        edge_ids.add(edge_id)
        for key in ("source", "target"):
            _text(edge.get(key), f"road edge {key}")
            if edge[key] not in nodes:
                raise ValueError(f"road edge {edge_id} references unknown {key}")
        for key in ("length_m", "speed_m_s"):
            _number(edge.get(key), f"road edge {key}", 0)
            if edge[key] == 0:
                raise ValueError(f"road edge {key} must be positive")
        if not isinstance(edge.get("closed"), bool):
            raise ValueError("road edge closed must be bool")
        _text(edge.get("speed_source"), "road edge speed_source")
        geometry = edge.get("geometry")
        if not isinstance(geometry, list) or len(geometry) < 2:
            raise ValueError("road edge geometry requires at least two coordinate pairs")
        for point in geometry:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("road geometry coordinates must be [x,y]")
            for coordinate in point:
                _number(coordinate, "road geometry coordinate")
        for point, node_id in ((geometry[0], edge["source"]), (geometry[-1], edge["target"])):
            if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6) for a, b in zip(point, nodes[node_id])):
                raise ValueError(f"road edge {edge_id} geometry disagrees with {node_id}")


def validate_resources(resources: list[dict]) -> None:
    if not isinstance(resources, list):
        raise ValueError("resources must be a list")
    _json(resources, "resources")
    ids = set()
    bases = {}
    common = ("home_x", "home_y", "mobilization_s", "speed_m_s", "offroad_speed_m_s",
              "max_access_m", "work_duration_s", "max_wind_m_s", "shift_end_s")
    optional_rates = ("capacity_l", "supply_l", "water_rate_l_s", "line_rate_m_s", "refill_rate_l_s",
                      "route_progress_m", "remaining_work_s", "max_slope", "min_ground_speed_m_s")
    for unit in resources:
        if not isinstance(unit, dict):
            raise ValueError("resource must be a dict")
        unit_id = unit.get("id")
        _text(unit_id, "resource id")
        if unit_id in ids:
            raise ValueError(f"duplicate resource id: {unit_id}")
        ids.add(unit_id)
        kind = unit.get("kind")
        if not isinstance(kind, str) or kind not in {"engine", "handcrew", "helicopter"}:
            raise ValueError(f"unknown resource kind: {kind!r}")
        _text(unit.get("base_id"), f"{unit_id}.base_id")
        required = list(common)
        required += (["line_rate_m_s", "max_slope"] if kind == "handcrew" else
                     ["capacity_l", "water_rate_l_s", "refill_rate_l_s"])
        if kind == "helicopter":
            required.append("min_ground_speed_m_s")
            if not isinstance(unit.get("air_corridor_available"), bool):
                raise ValueError(f"{unit_id}.air_corridor_available must be bool")
        sources = unit.get("parameter_sources")
        if not isinstance(sources, dict):
            raise ValueError(f"{unit_id}.parameter_sources must label operational parameters")
        for key in required:
            _number(unit.get(key), f"{unit_id}.{key}", None if key in {"home_x", "home_y"} else 0)
            _text(sources.get(key), f"{unit_id}.parameter_sources.{key}")
        if kind == "helicopter":
            _text(sources.get("air_corridor_available"), f"{unit_id}.parameter_sources.air_corridor_available")
        for key in optional_rates:
            if key in unit:
                _number(unit[key], f"{unit_id}.{key}", 0)
                _text(sources.get(key), f"{unit_id}.parameter_sources.{key}")
        if kind == "helicopter" and unit["min_ground_speed_m_s"] <= 0:
            raise ValueError(f"{unit_id}.min_ground_speed_m_s must be positive")
        if kind == "handcrew":
            if any(unit.get(key, 0) != 0 for key in ("capacity_l", "supply_l", "water_rate_l_s", "refill_rate_l_s")):
                raise ValueError("handcrew supplies are not water; water fields must be absent or zero")
        elif unit.get("line_rate_m_s", 0) != 0:
            raise ValueError("only handcrews can perform line work")
        if unit.get("supply_l", unit.get("capacity_l", 0)) > unit.get("capacity_l", 0):
            raise ValueError(f"{unit_id} supply exceeds capacity")
        if "status" in unit and (not isinstance(unit["status"], str) or unit["status"] not in UNIT_STATUSES):
            raise ValueError(f"{unit_id} has unknown status")
        for key in ("x", "y"):
            if key in unit:
                _number(unit[key], f"{unit_id}.{key}")
        if "assignment" in unit and unit["assignment"] is not None and not isinstance(unit["assignment"], dict):
            raise ValueError(f"{unit_id}.assignment must be a dict or null")
        if "route" in unit and not isinstance(unit["route"], list):
            raise ValueError(f"{unit_id}.route must be a list")
        home = (unit["home_x"], unit["home_y"])
        if unit["base_id"] in bases and bases[unit["base_id"]] != home:
            raise ValueError("units sharing base_id must agree on home coordinates")
        bases[unit["base_id"]] = home
