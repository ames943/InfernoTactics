"""Deterministic observed-frontier actions with shared dispatch feasibility.

No fire-engine state, future weather, provenance, or randomized physics is read.
ETA means time until work starts (mobilization plus current feasible travel), not
an estimate of suppression effectiveness or eventual containment.
"""
from __future__ import annotations

import math
import hashlib
import weakref
import numpy as np

from .contracts import Action, Scenario
from .resources import ResourceEngine

_CANDIDATE_CACHE: dict[int, tuple[weakref.ReferenceType, bytes, list[Action], list[dict]]] = {}
_UNIT_CANDIDATE_CACHE: dict[tuple[int, str], tuple[bytes, list[Action], list[dict]]] = {}


def _public_key(observation: dict, scenario: Scenario) -> bytes:
    """Fingerprint current feasibility, not future bands or physics parameters."""
    digest = hashlib.blake2b(digest_size=20)
    for values in (observation["fire"]["burned"], observation["fire"]["frontier"],
                   *(observation["grid"][key] for key in
                     ("valid", "fuel_model", "elevation", "slope")),
                   *observation["weather"].values()):
        array = np.asarray(values)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    corridor = observation["grid"].get("air_corridor_available")
    if corridor is not None:
        digest.update(np.asarray(corridor).tobytes())
    digest.update(repr((scenario.transform, scenario.duration_s, observation["roads"],
                        observation["operations"], observation["units"],
                        [observation["time_s"] >= unit["shift_end_s"]
                         for unit in observation["units"]])).encode())
    return digest.digest()



def _frontier_samples(observation: dict, limit: int = 16) -> list[tuple[int, int]]:
    """Sample equal arc lengths of oriented, observed burned-side cell edges.

    Trace each boundary deterministically, including disconnected islands, then
    concatenate contours in lexicographic order. Duplicate cells at corners are
    retained only once; an observation therefore yields at most ``limit`` sites.
    """
    fire = observation["fire"]
    burned, frontier = fire["burned"], fire["frontier"]
    valid = observation["grid"]["valid"]
    fuel = observation["grid"]["fuel_model"]
    unburned = valid & (fuel >= 101) & (fuel <= 204) & ~burned
    edges = {}
    height, width = burned.shape
    for r, c in np.argwhere(frontier):
        r, c = int(r), int(c)
        sides = [((r, c), (r, c + 1), r - 1, c),
                 ((r, c + 1), (r + 1, c + 1), r, c + 1),
                 ((r + 1, c + 1), (r + 1, c), r + 1, c),
                 ((r + 1, c), (r, c), r, c - 1)]
        for start, end, nr, nc in sides:
            if 0 <= nr < height and 0 <= nc < width and unburned[nr, nc]:
                edges[(start, end)] = (r, c)
    if not edges:
        return []
    outgoing = {}
    for edge in edges:
        outgoing.setdefault(edge[0], []).append(edge)
    remaining = set(edges)
    arc = []
    while remaining:
        edge = min(remaining)
        while edge in remaining:
            remaining.remove(edge)
            arc.append(edges[edge])
            following = [item for item in outgoing.get(edge[1], ()) if item in remaining]
            if not following:
                break
            edge = min(following)
    count = min(limit, int(np.count_nonzero(frontier)))
    indices = np.floor((np.arange(count) + .5) * len(arc) / count).astype(int)
    return list(dict.fromkeys(arc[index] for index in indices))


def _point(scenario: Scenario, cell: tuple[int, int]) -> tuple[float, float]:
    a, _, c, _, e, f = scenario.transform
    row, col = cell
    return float(c + (col + .5) * a), float(f + (row + .5) * e)


def _road_projections(roads: dict, point: tuple[float, float]):
    """Nearest geometric projections first, ties resolved by edge/segment ID."""
    choices = []
    for edge in sorted(roads["edges"], key=lambda edge: edge["id"]):
        if edge["closed"]:
            continue
        for index, (start, end) in enumerate(zip(edge["geometry"], edge["geometry"][1:])):
            dx, dy = end[0] - start[0], end[1] - start[1]
            length2 = dx * dx + dy * dy
            t = 0.0 if length2 == 0 else max(0.0, min(1.0,
                ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length2))
            x, y = float(start[0] + t * dx), float(start[1] + t * dy)
            choices.append((math.hypot(x - point[0], y - point[1]), edge["id"], index, x, y))
    return [(x, y) for _, _, _, x, y in sorted(choices)]


def _candidate_data(observation: dict, scenario: Scenario) -> tuple[list[Action], list[dict]]:
    if observation["time_s"] >= scenario.duration_s or observation.get("done", False):
        return [Action()], [{"eta_s": None}]
    key = _public_key(observation, scenario)
    cached = _CANDIDATE_CACHE.get(id(scenario))
    if cached is not None and cached[0]() is scenario and cached[1] == key:
        return list(cached[2]), [dict(item) for item in cached[3]]
    validator = ResourceEngine.from_observation(observation, scenario)
    actions = [Action()]
    metadata = [{"eta_s": None}]
    samples = _frontier_samples(observation)
    burned = observation["fire"]["burned"]
    grid = observation["grid"]
    burnable = grid["valid"] & (grid["fuel_model"] >= 101) & (grid["fuel_model"] <= 204)
    burned_cells = np.argwhere(burned)
    center = burned_cells.mean(axis=0) if len(burned_cells) else np.zeros(2)
    height, width = burned.shape

    def add(action):
        if action in actions:
            return True
        try:
            eta = validator.feasible_eta(action, observation["time_s"])
        except ValueError:
            return False
        actions.append(action)
        metadata.append({"eta_s": eta})
        return True

    for unit in sorted(observation["units"], key=lambda unit: unit["id"]):
        unit_key = _public_key({**observation, "units": [unit]}, scenario)
        unit_identity = (id(scenario), unit["id"])
        unit_cached = _UNIT_CANDIDATE_CACHE.get(unit_identity)
        if unit_cached is not None and unit_cached[0] == unit_key:
            actions.extend(unit_cached[1])
            metadata.extend(dict(item) for item in unit_cached[2])
            continue
        start_index = len(actions)
        if unit["status"] != "available":
            # A busy unit's public position/route changes each step; computing
            # its one return candidate is cheap and must stay current.
            add(Action(unit["id"], "return"))
            continue
        for row, col in samples:
            if unit["kind"] == "handcrew":
                ahead = [(row + dr, col + dc) for dr, dc in
                         ((-1, 0), (0, -1), (0, 1), (1, 0))]
                ahead = [(r, c) for r, c in ahead if 0 <= r < height and 0 <= c < width
                         and burnable[r, c] and not burned[r, c]]
                ahead.sort(key=lambda cell: (-float(np.sum((np.asarray(cell) - center) ** 2)), cell))
                for cell in ahead:
                    if add(Action(unit["id"], "line", *_point(scenario, cell))):
                        break
            elif unit["kind"] == "helicopter":
                add(Action(unit["id"], "water", *_point(scenario, (row, col))))
            else:
                for point in _road_projections(observation["roads"], _point(scenario, (row, col))):
                    if add(Action(unit["id"], "water", *point)):
                        break
        _UNIT_CANDIDATE_CACHE[unit_identity] = (
            unit_key, list(actions[start_index:]),
            [dict(item) for item in metadata[start_index:]])
    identity = id(scenario)
    def discard(reference):
        cached = _CANDIDATE_CACHE.get(identity)
        if cached is not None and cached[0] is reference:
            _CANDIDATE_CACHE.pop(identity, None)
            for unit_identity in [item for item in _UNIT_CANDIDATE_CACHE if item[0] == identity]:
                _UNIT_CANDIDATE_CACHE.pop(unit_identity, None)
    _CANDIDATE_CACHE[identity] = (weakref.ref(scenario, discard), key,
                                  list(actions), [dict(item) for item in metadata])
    return actions, metadata


def candidate_actions(obs: dict, scenario: Scenario) -> list[Action]:
    """Wait first, sorted units, <=16 feasible observed targets per unit."""
    return _candidate_data(obs, scenario)[0]


def heuristic_action(obs: dict, scenario: Scenario) -> Action:
    """Choose earliest feasible work ETA, preserving deterministic tie order."""
    if "candidates" in obs and "candidate_metadata" in obs:
        actions, metadata = obs["candidates"], obs["candidate_metadata"]
    else:
        actions, metadata = _candidate_data(obs, scenario)
    mask = obs.get("candidate_mask", [True] * len(actions))
    feasible = [(data["eta_s"], index, action)
                for index, (action, data, valid) in enumerate(zip(actions, metadata, mask))
                if valid and action.operation in {"water", "line"} and data["eta_s"] is not None]
    return min(feasible, key=lambda item: (item[0], item[1]))[2] if feasible else Action()
