"""Illustrative travel models in projected metres and elapsed seconds.

Road costs use directed polyline geometry and the lesser of road/unit speed.
Off-road crew speeds use the approximate Tobler grade relation, normalized to
unit.offroad_speed_m_s on flat ground; these are research travel assumptions,
not operational access or aviation authorization. Every returned segment starts
at the previous endpoint (or the unit's actual position), so interruption never
snaps a unit to a road node. Durations exclude mobilization and work.
"""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
import weakref

from .contracts import Scenario

_EPS = 1e-7
_GRAPH_CACHE: dict[int, tuple[weakref.ReferenceType, tuple, dict]] = {}


def _distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _cell(scenario, x, y):
    a, _, c, _, e, f = scenario.transform
    row, col = math.floor((y - f) / e), math.floor((x - c) / a)
    height, width = scenario.landscape["valid"].shape
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError("route leaves landscape coverage")
    return row, col


def _center(scenario, cell):
    row, col = cell
    a, _, c, _, e, f = scenario.transform
    return c + (col + .5) * a, f + (row + .5) * e


def _terrain_cell(scenario, unit, point):
    row, col = _cell(scenario, *point)
    land = scenario.landscape
    if not land["valid"][row, col]:
        raise ValueError("route crosses nodata")
    if land["fuel_model"][row, col] == 98:
        raise ValueError("route crosses water")
    if land["slope"][row, col] > unit.get("max_slope", math.inf):
        raise ValueError("route exceeds slope limit")
    return row, col


def _breakpoints(scenario, start, end):
    """Split a straight segment at every raster boundary, not sample intervals."""
    a, _, c, _, e, f = scenario.transform
    result = {0.0, 1.0}
    for origin, delta, offset, scale in (
        (start[0], end[0] - start[0], c, a),
        (start[1], end[1] - start[1], f, e),
    ):
        if abs(delta) <= _EPS:
            continue
        lo, hi = sorted(((origin - offset) / scale,
                         (origin + delta - offset) / scale))
        for boundary in range(math.floor(lo) + 1, math.ceil(hi)):
            t = (offset + boundary * scale - origin) / delta
            if 0 < t < 1:
                result.add(t)
    return sorted(result)


def _interpolate(start, end, fraction):
    return (start[0] + (end[0] - start[0]) * fraction,
            start[1] + (end[1] - start[1]) * fraction)


def validate_ground_segment(scenario, unit, start_x, start_y, x, y):
    """Reject terrain-incompatible straight travel or incremental line work."""
    start, end = (start_x, start_y), (x, y)
    samples = [start, end]
    breaks = _breakpoints(scenario, start, end)
    samples.extend(_interpolate(start, end, (a + b) / 2)
                   for a, b in zip(breaks, breaks[1:]))
    previous = None
    for point in samples:
        _terrain_cell(scenario, unit, point)
    # Elevation grade is checked in travel order, separately from raster slope.
    ordered = [start] + [_interpolate(start, end, (a + b) / 2)
                         for a, b in zip(breaks, breaks[1:])] + [end]
    for point in ordered:
        cell = _cell(scenario, *point)
        if previous is not None and previous[1] != cell:
            _, old_cell = previous
            # Cell-center spacing avoids arbitrarily steep grades at cell edges.
            length = _distance(_center(scenario, old_cell), _center(scenario, cell))
            grade = abs(float(scenario.landscape["elevation"][cell]) -
                        float(scenario.landscape["elevation"][old_cell])) / length
            if grade > unit.get("max_slope", math.inf):
                raise ValueError("route exceeds DEM grade limit")
        previous = point, cell


def _segment(start, end, seconds, **metadata):
    return {"start_x": float(start[0]), "start_y": float(start[1]),
            "x": float(end[0]), "y": float(end[1]),
            "seconds": float(seconds), "length_m": _distance(start, end), **metadata}


def _crew_speed(unit, grade):
    # Normalize Tobler's exp(-3.5*abs(grade+.05)) to the configured flat speed.
    return unit["offroad_speed_m_s"] * math.exp(-3.5 * (abs(grade + .05) - .05))


def _offroad(scenario, unit, start, end):
    if _distance(start, end) <= _EPS:
        return []
    limit = unit["max_access_m"]
    if _distance(start, end) > limit + _EPS:
        raise ValueError("destination exceeds bounded last-mile access")
    if unit["offroad_speed_m_s"] <= 0:
        raise ValueError("unit has no off-road travel speed")
    if unit["kind"] != "handcrew":
        validate_ground_segment(scenario, unit, *start, *end)
        return [_segment(start, end, _distance(start, end) / unit["offroad_speed_m_s"],
                         mode="offroad")]
    start_cell = _terrain_cell(scenario, unit, start)
    end_cell = _terrain_cell(scenario, unit, end)
    if start_cell == end_cell:
        validate_ground_segment(scenario, unit, *start, *end)
        return [_segment(start, end, _distance(start, end) / _crew_speed(unit, 0),
                         mode="offroad")]
    # Nondominated time/distance labels preserve feasible bounded paths even
    # when a faster grade-adjusted path is longer than a slower alternative.
    # A* uses the maximum possible Tobler speed (at grade -.05), so its
    # Euclidean lower bound is admissible on every DEM and preserves optimality.
    maximum_speed = unit["offroad_speed_m_s"] * math.exp(.175)
    def lower_bound(cell):
        return _distance(_center(scenario, cell), end) / maximum_speed
    initial = _distance(start, _center(scenario, start_cell))
    initial_cost = initial / _crew_speed(unit, 0)
    labels = {start_cell: [(initial_cost, initial, None)]}
    queue = [(initial_cost + lower_bound(start_cell), initial_cost, initial, start_cell, 0)]
    finish = None
    while queue:
        _, cost, length, cell, index = heapq.heappop(queue)
        label = labels[cell][index]
        if label is None:
            continue
        if cell == end_cell:
            tail = _distance(_center(scenario, cell), end)
            if length + tail <= limit + _EPS:
                finish = cell, index
                break
        row, col = cell
        for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                       (0, 1), (1, -1), (1, 0), (1, 1)):
            other = row + dr, col + dc
            point = _center(scenario, other)
            try:
                _terrain_cell(scenario, unit, point)
                # No diagonal corner cutting between water/nodata/steep cells.
                if dr and dc:
                    _terrain_cell(scenario, unit, _center(scenario, (row + dr, col)))
                    _terrain_cell(scenario, unit, _center(scenario, (row, col + dc)))
            except ValueError:
                continue
            distance = _distance(_center(scenario, cell), point)
            grade = (float(scenario.landscape["elevation"][other]) -
                     float(scenario.landscape["elevation"][cell])) / distance
            if abs(grade) > unit.get("max_slope", math.inf):
                continue
            new_length = length + distance
            if new_length + _distance(point, end) > limit + _EPS:
                continue
            new_cost = cost + distance / _crew_speed(unit, grade)
            entries = labels.setdefault(other, [])
            if any(entry is not None and entry[0] <= new_cost + _EPS
                   and entry[1] <= new_length + _EPS for entry in entries):
                continue
            # Keep prior labels intact: descendants can reference their parents.
            entries.append((new_cost, new_length, (cell, index)))
            heapq.heappush(queue, (new_cost + lower_bound(other), new_cost, new_length, other, len(entries) - 1))
    if finish is None:
        raise ValueError("no reachable DEM eight-neighbor last-mile path")
    cells = []
    current = finish
    while current is not None:
        cell, index = current
        cells.append(cell)
        current = labels[cell][index][2]
    points = [start] + [_center(scenario, cell) for cell in reversed(cells)] + [end]
    result = []
    for a, b in zip(points, points[1:]):
        distance = _distance(a, b)
        if distance <= _EPS:
            continue
        ac, bc = _cell(scenario, *a), _cell(scenario, *b)
        grade = (float(scenario.landscape["elevation"][bc]) -
                 float(scenario.landscape["elevation"][ac])) / distance
        result.append(_segment(a, b, distance / _crew_speed(unit, grade), mode="offroad"))
    return result


def _road_key(scenario):
    return (scenario.roads.get("version", 0), tuple(
        (edge["id"], edge["source"], edge["target"], bool(edge["closed"]),
         float(edge["speed_m_s"]), tuple(tuple(point) for point in edge["geometry"]))
        for edge in scenario.roads["edges"]))


@dataclass
class _Graph:
    edges: list
    adjacency: dict
    revision: str


def _graph(scenario, speed):
    key = _road_key(scenario)
    identity = id(scenario)
    entry = _GRAPH_CACHE.get(identity)
    if entry is None or entry[0]() is not scenario or entry[1] != key:
        def remove(reference):
            cached = _GRAPH_CACHE.get(identity)
            if cached is not None and cached[0] is reference:
                _GRAPH_CACHE.pop(identity, None)
        entry = weakref.ref(scenario, remove), key, {}
        _GRAPH_CACHE[identity] = entry
    cache = entry[2]
    if speed not in cache:
        edges, adjacency = [], defaultdict(list)
        revision = hashlib.sha256(repr(key).encode()).hexdigest()
        for raw in scenario.roads["edges"]:
            if raw["closed"]:
                continue
            geometry = [tuple(point) for point in raw["geometry"]]
            cumulative = _cumulative(geometry)
            edge = {"id": raw["id"], "u": raw["source"], "v": raw["target"],
                    "geometry": geometry, "cumulative": cumulative,
                    "length": cumulative[-1], "speed": min(speed, raw["speed_m_s"])}
            if edge["speed"] <= 0:
                continue
            edges.append(edge)
            adjacency[edge["u"]].append(edge)
        cache[speed] = _Graph(edges, adjacency, revision)
    return cache[speed]


def _project(point, edge):
    best = None
    for index, (a, b) in enumerate(zip(edge["geometry"], edge["geometry"][1:])):
        length = _distance(a, b)
        if length <= _EPS:
            continue
        t = max(0.0, min(1.0, ((point[0] - a[0]) * (b[0] - a[0]) +
                               (point[1] - a[1]) * (b[1] - a[1])) / length ** 2))
        projection = _interpolate(a, b, t)
        candidate = (_distance(point, projection), edge["cumulative"][index] + t * length,
                     projection)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _road_segments(edge, start_distance, end_distance, revision):
    def at(distance):
        for index, (a, b) in enumerate(zip(edge["geometry"], edge["geometry"][1:])):
            lo, hi = edge["cumulative"][index:index + 2]
            if distance <= hi + _EPS and hi > lo:
                return _interpolate(a, b, max(0, min(1, (distance - lo) / (hi - lo))))
        return edge["geometry"][-1]
    points = [at(start_distance)]
    points.extend(point for point, distance in zip(edge["geometry"], edge["cumulative"])
                  if start_distance + _EPS < distance < end_distance - _EPS)
    points.append(at(end_distance))
    return [_segment(a, b, _distance(a, b) / edge["speed"], mode="road",
                     edge_id=edge["id"], road_revision=revision, road_speed_m_s=edge["speed"])
            for a, b in zip(points, points[1:]) if _distance(a, b) > _EPS]


def _cumulative(geometry):
    distances = [0.0]
    for start, end in zip(geometry, geometry[1:]):
        distances.append(distances[-1] + _distance(start, end))
    return distances


def _attachments(scenario, unit, graph, point, outbound):
    projections = [(edge, _project(point, edge)) for edge in graph.edges]
    # A position on a closed edge must not become an off-road teleport onto
    # some nearby open edge. Shared open endpoints remain legitimate exits.
    on_road = any(projected and projected[0] <= _EPS for _, projected in projections) or any(
        projected is not None and projected[0] <= _EPS
        for projected in (_project(point, {"geometry": edge["geometry"],
            "cumulative": _cumulative(edge["geometry"])})
            for edge in scenario.roads["edges"] if edge["closed"]))
    result = []
    for edge, projected in projections:
        if projected is None:
            continue
        distance, along, snapped = projected
        if distance > unit["max_access_m"] + _EPS or (on_road and distance > _EPS):
            continue
        try:
            connector = _offroad(scenario, unit, point, snapped) if outbound else _offroad(
                scenario, unit, snapped, point)
        except ValueError:
            continue
        result.append((edge, along, snapped, connector))
    return result


def _ground_route(scenario, unit, end, access_only=False):
    start = float(unit["x"]), float(unit["y"])
    _terrain_cell(scenario, unit, start)
    _terrain_cell(scenario, unit, end)
    graph = _graph(scenario, unit["speed_m_s"])
    starts = _attachments(scenario, unit, graph, start, True)
    goals = _attachments(scenario, unit, graph, end, False)
    best, best_cost, best_access = None, math.inf, None
    # Crews may walk directly within the configured bounded last-mile range.
    if unit["kind"] == "handcrew" and not access_only:
        try:
            direct = _offroad(scenario, unit, start, end)
            best, best_cost = direct, sum(segment["seconds"] for segment in direct)
        except ValueError:
            pass
    distances, paths, queue = {}, {}, []
    for edge, along, _, connector in starts:
        route = connector + _road_segments(edge, along, edge["length"], graph.revision)
        cost = sum(segment["seconds"] for segment in route)
        if cost < distances.get(edge["v"], math.inf):
            distances[edge["v"]], paths[edge["v"]] = cost, route
            heapq.heappush(queue, (cost, edge["v"]))
        # Source and target can both lie inside a directed edge, without a node.
        for goal, goal_along, access, tail in goals:
            if goal["id"] == edge["id"] and goal_along >= along - _EPS:
                candidate = connector + _road_segments(edge, along, goal_along, graph.revision)
                full = candidate if access_only else candidate + tail
                score = sum(segment["seconds"] for segment in full)
                if score < best_cost:
                    best, best_cost, best_access = full, score, access
        # An edge endpoint is also a valid graph node in either direction.
        if along <= _EPS:
            cost = sum(segment["seconds"] for segment in connector)
            if cost < distances.get(edge["u"], math.inf):
                distances[edge["u"]], paths[edge["u"]] = cost, connector
                heapq.heappush(queue, (cost, edge["u"]))
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances[node]:
            continue
        for edge in graph.adjacency.get(node, ()):
            new_cost = cost + edge["length"] / edge["speed"]
            if new_cost < distances.get(edge["v"], math.inf):
                distances[edge["v"]] = new_cost
                paths[edge["v"]] = paths[node] + _road_segments(edge, 0, edge["length"], graph.revision)
                heapq.heappush(queue, (new_cost, edge["v"]))
    for edge, along, access, tail in goals:
        if edge["u"] in paths:
            candidate = paths[edge["u"]] + _road_segments(edge, 0, along, graph.revision)
            full = candidate if access_only else candidate + tail
            score = sum(segment["seconds"] for segment in full)
            if score < best_cost:
                best, best_cost, best_access = full, score, access
        if along >= edge["length"] - _EPS and edge["v"] in paths:
            candidate = paths[edge["v"]]
            full = candidate if access_only else candidate + tail
            score = sum(segment["seconds"] for segment in full)
            if score < best_cost:
                best, best_cost, best_access = full, score, access
    if best is None:
        raise ValueError("destination is unreachable on directed roads and bounded last-mile access")
    return best_access if access_only else best


def check_air_constraints(scenario, unit, time_s, x, y):
    """Check configured illustrative corridor/weather limits, raising ValueError."""
    row, col = _cell(scenario, x, y)
    if not scenario.landscape["valid"][row, col]:
        raise ValueError("air route crosses nodata weather coverage")
    corridor = scenario.landscape.get("air_corridor_available")
    if (not unit.get("air_corridor_available", True) or
            not scenario.provenance.get("air_corridor_available", True) or
            (corridor is not None and not corridor[row, col])):
        raise ValueError("air corridor is unavailable")
    if not math.isfinite(time_s) or time_s < 0 or time_s > scenario.duration_s:
        raise ValueError("air route time is outside scenario weather coverage")
    index = min(int(time_s // scenario.weather_step_s),
                scenario.weather["wind_speed_m_s"].shape[0] - 1)
    wind = scenario.weather["wind_speed_m_s"]
    speed = float(wind[index] if wind.ndim == 1 else
                  wind[index, 0 if wind.shape[1] == 1 else row,
                       0 if wind.shape[2] == 1 else col])
    if not math.isfinite(speed) or speed > unit["max_wind_m_s"]:
        raise ValueError("air operation exceeds configured wind limit")
    return index, row, col, speed


def _air_route(scenario, unit, end, time_s):
    start = float(unit["x"]), float(unit["y"])
    check_air_constraints(scenario, unit, time_s, *start)
    distance = _distance(start, end)
    if distance <= _EPS:
        return []
    minimum = unit.get("min_ground_speed_m_s", 0.1)
    if minimum <= 0 or not math.isfinite(minimum):
        raise ValueError("minimum helicopter ground speed must be positive")
    ux, uy = (end[0] - start[0]) / distance, (end[1] - start[1]) / distance
    result = []
    breaks = _breakpoints(scenario, start, end)
    for lo, hi in zip(breaks, breaks[1:]):
        position = _interpolate(start, end, lo)
        finish = _interpolate(start, end, hi)
        midpoint = _interpolate(start, end, (lo + hi) / 2)
        # Dispatch can use only weather observed at this coupling boundary.
        # ResourceEngine refreshes unfinished routes when the current band changes.
        index, row, col, wind = check_air_constraints(scenario, unit, time_s, *midpoint)
        direction = scenario.weather["upwind_direction_deg"]
        angle = math.radians(float(direction[index] if direction.ndim == 1 else
            direction[index, 0 if direction.shape[1] == 1 else row,
                      0 if direction.shape[2] == 1 else col]))
        projected = -wind * (math.sin(angle) * ux + math.cos(angle) * uy)
        speed = max(minimum, unit["speed_m_s"] + projected)
        result.append(_segment(position, finish, _distance(position, finish) / speed, mode="air"))
    return result


def plan_route(scenario: Scenario, unit: dict, x: float, y: float, time_s: float) -> list[dict]:
    """Plan from actual unit position; raise ValueError rather than teleport.

    Segment fields are x/y (endpoint), start_x/start_y, seconds, length_m, mode;
    road segments also identify edge_id, road_revision and road_speed_m_s.
    A route has no mobilization/work duration. An empty route means co-location.
    """
    if not all(math.isfinite(value) for value in (x, y, unit["x"], unit["y"], time_s)):
        raise ValueError("route coordinates and time must be finite")
    if unit["kind"] == "helicopter":
        return _air_route(scenario, unit, (x, y), time_s)
    if unit["kind"] not in {"engine", "handcrew"}:
        raise ValueError("unsupported routing unit kind")
    if _distance((unit["x"], unit["y"]), (x, y)) <= _EPS:
        _terrain_cell(scenario, unit, (x, y))
        return []
    return _ground_route(scenario, unit, (x, y))


def line_access_point(scenario, unit, x, y, time_s):
    """Reachable road/last-mile transition for incremental line work."""
    if unit["kind"] == "helicopter":
        plan_route(scenario, unit, x, y, time_s)
        return float(x), float(y)
    return _ground_route(scenario, unit, (x, y), access_only=True)


def route_invalid_reason(scenario, unit, remaining_segments, time_s):
    """Return a boundary invalidation reason without moving or rerouting a unit."""
    try:
        if unit["kind"] == "helicopter":
            check_air_constraints(scenario, unit, time_s, unit["x"], unit["y"])
            if remaining_segments:
                end = remaining_segments[-1]
                # The first segment may be untrimmed after interpolation.
                old_remaining = 0.0
                start = unit["x"], unit["y"]
                for segment in remaining_segments:
                    length = _distance(start, (segment["x"], segment["y"]))
                    if segment["length_m"] > _EPS:
                        old_remaining += segment["seconds"] * length / segment["length_m"]
                    start = segment["x"], segment["y"]
                new_route = _air_route(scenario, unit, (end["x"], end["y"]), time_s)
                if not math.isclose(old_remaining, sum(s["seconds"] for s in new_route),
                                    rel_tol=1e-8, abs_tol=1e-6):
                    return "air wind/time travel cost changed"
        else:
            graph = _graph(scenario, unit["speed_m_s"])
            for segment in remaining_segments:
                if segment.get("mode") == "road" and segment.get("road_revision") != graph.revision:
                    return "road closure, geometry, or speed version changed"
                if segment.get("mode") == "road":
                    edge = next((edge for edge in graph.edges if edge["id"] == segment["edge_id"]), None)
                    if edge is None or not math.isclose(segment["road_speed_m_s"], edge["speed"]):
                        return "road speed or unit speed cap changed"
                if segment.get("mode") == "offroad":
                    start = (unit["x"], unit["y"]) if segment is remaining_segments[0] else (
                        segment["start_x"], segment["start_y"])
                    validate_ground_segment(scenario, unit, *start, segment["x"], segment["y"])
    except ValueError as exc:
        return str(exc)
    return None
