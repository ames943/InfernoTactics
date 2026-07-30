"""Run the canonical v8 relative-action policy and export a replayable trajectory.

This is the bridge between the model and the live simulation view.  It runs a
real deterministic rollout of ``RelativeInfernoModel`` against a real
``InfernoEnv`` -- the same environment, weather, routing and reward the training
loop used -- and records everything the player needs to redraw it frame by
frame:

  * per-tick fire-state diffs (only the cells that changed, so the file stays
    small even for a 150-tick episode on a 316x595 grid)
  * every resource unit's position each tick, reconstructed from the unit's real
    road route or air path, so vehicles physically follow the OSM road network
  * every suppression/trench/rescue effect with the exact cell the physics hit
  * every building loss, with its real population-density penalty multiplier
  * the real weather series, reward stream and semantic action the policy chose

Nothing here re-simulates or approximates the fire.  The player is a renderer;
this file is the only place the model and environment actually run.

Checkpoint compatibility
------------------------
v8 checkpoints were trained before the v9 traffic scalars existed, so they carry
an 8-column MLP input weight while the current env emits 11 scalars.  The three
v9 scalars are appended *last* in ``SCALAR_KEYS``, so widening the first layer
and zeroing the new columns is function-identical to the original v8 network on
the eight scalars it was actually trained on.  ``--traffic-mode legacy``
additionally reproduces the v8-era dynamics; ``synthetic`` runs the current
traffic/delay model, which those weights never saw.
"""

import argparse
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.fire_sim import BLAZE, BURNED_OUT, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    AIR_RESOURCE_TYPES,
    DEPLOYED_BUSY_TICKS,
    FIRE_START_UTC,
    GROUND_RESOURCE_TYPES,
    HELICOPTER_RELOAD_TICKS,
    LAYER_INDEX,
    MAX_TICKS,
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_TYPES,
    SCALAR_KEYS,
    TICK_DURATION_MINUTES,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    _effect_target_point,
)
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import TARGET_TYPES, decode_action  # noqa: E402
from train.train_relative import MAX_DISPATCH_SLOTS, _forward  # noqa: E402


# Fire states the player needs to draw.  Extinguished cells go back to SAFE and
# simply stop being drawn, so they are absent from the diff stream by design.
DRAWN_STATES = (THREAT, BLAZE, BURNED_OUT)

SCENARIOS = {
    "anchor": {
        "label": "Skull Rock trailhead",
        "sublabel": "Real documented Palisades Fire origin - the training ignition point",
        "kind": "training",
        "reset": {"ignition_point": TRAINING_IGNITION_POINT},
    },
    "mandeville": {
        "label": "Mandeville Canyon",
        "sublabel": "Held-out validation ignition - never trained on",
        "kind": "held-out",
        "reset": {"ignition_point": VALIDATION_IGNITION_POINTS["mandeville_canyon"]},
    },
    "getty": {
        "label": "Getty View Park",
        "sublabel": "Held-out validation ignition - never trained on",
        "kind": "held-out",
        "reset": {"ignition_point": VALIDATION_IGNITION_POINTS["getty_view_park"]},
    },
    "multi": {
        "label": "Three simultaneous ignitions",
        "sublabel": "Topanga ridge + Sullivan Canyon + Stone Canyon - exceeds fleet capacity",
        "kind": "stress",
        "reset": {"scenario": "multi"},
    },
}


# --- geometry ---------------------------------------------------------------

class PixelTransform:
    """Grid affine transform and its inverse, in cell/pixel space.

    ``grid_meta.json``'s transform maps (col, row) -> projected (x, y).  The
    player works in cell coordinates, so routes and station locations coming
    from the projected road graph have to come back the other way.
    """

    def __init__(self, meta):
        a, b, c, d, e, f = meta["transform"]
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f
        det = a * e - b * d
        if abs(det) < 1e-12:
            raise ValueError("grid transform is singular")
        self.inv = (e / det, -b / det, -d / det, a / det)

    def cell_to_xy(self, row, col):
        x = self.a * (col + 0.5) + self.b * (row + 0.5) + self.c
        y = self.d * (col + 0.5) + self.e * (row + 0.5) + self.f
        return x, y

    def xy_to_cell(self, x, y):
        """Return fractional (col, row) -- the player's drawing coordinates."""
        ia, ib, ic, id_ = self.inv
        dx, dy = x - self.c, y - self.f
        col = ia * dx + ib * dy - 0.5
        row = ic * dx + id_ * dy - 0.5
        return col, row


def _round(value, digits=2):
    return float(np.round(float(value), digits))


def _edge_polyline(graph, u, v, transform):
    """Real geometry of one road edge as fractional (col, row) points."""
    records = graph.get_edge_data(u, v) or {}
    best = None
    for _key, data in records.items():
        length = float(data.get("travel_time", data.get("length", 1.0)))
        if best is None or length < best[0]:
            best = (length, data)
    points = []
    if best is not None and best[1].get("geometry") is not None:
        coords = list(best[1]["geometry"].coords)
        points = [transform.xy_to_cell(x, y) for x, y in coords]
    if not points:
        for node in (u, v):
            attrs = graph.nodes[node]
            points.append(transform.xy_to_cell(attrs["x"], attrs["y"]))
    return points


def _polyline_from_nodes(graph, nodes, transform):
    """Stitch per-edge geometry into one polyline, orienting each edge to match."""
    if len(nodes) < 2:
        if nodes:
            attrs = graph.nodes[nodes[0]]
            return [transform.xy_to_cell(attrs["x"], attrs["y"])]
        return []
    polyline = []
    for u, v in zip(nodes, nodes[1:]):
        points = _edge_polyline(graph, u, v, transform)
        head = graph.nodes[u]
        head_cell = transform.xy_to_cell(head["x"], head["y"])
        if points and math.dist(points[0], head_cell) > math.dist(points[-1], head_cell):
            points = points[::-1]  # OSM stores some edges against travel direction
        if polyline and points and math.dist(polyline[-1], points[0]) < 1e-6:
            points = points[1:]
        polyline.extend(points)
    return polyline


def _cumulative_lengths(polyline):
    lengths = [0.0]
    for previous, current in zip(polyline, polyline[1:]):
        lengths.append(lengths[-1] + math.dist(previous, current))
    return lengths


def _point_along(polyline, lengths, fraction):
    """Interpolate a point at ``fraction`` of the polyline's arc length."""
    if not polyline:
        return None
    if len(polyline) == 1 or lengths[-1] <= 1e-9:
        return polyline[-1]
    target = max(0.0, min(1.0, fraction)) * lengths[-1]
    index = int(np.searchsorted(lengths, target, side="right")) - 1
    index = max(0, min(index, len(polyline) - 2))
    span = lengths[index + 1] - lengths[index]
    local = 0.0 if span <= 1e-9 else (target - lengths[index]) / span
    x0, y0 = polyline[index]
    x1, y1 = polyline[index + 1]
    return (x0 + (x1 - x0) * local, y0 + (y1 - y0) * local)


def _simplify(polyline, tolerance=0.6):
    """Drop near-collinear points; route lines are drawn, not measured."""
    if len(polyline) <= 2:
        return list(polyline)
    kept = [polyline[0]]
    for point in polyline[1:-1]:
        if math.dist(point, kept[-1]) >= tolerance:
            kept.append(point)
    kept.append(polyline[-1])
    return kept


# --- unit motion ------------------------------------------------------------

class UnitTracker:
    """Reconstruct where each resource unit is, tick by tick.

    The environment models a dispatch as a state machine plus a tick countdown;
    it never stores a position.  Every position here is derived from data the
    env does hold -- the unit's home station, its target zone, its real route
    and its remaining ticks -- so the animation follows the same roads and the
    same clock the reward function was computed against.

    One rendering-only liberty is documented in the player UI: during the
    post-effect ``deployed`` phase (a plain busy timer in the env, which is a
    12-tick reload for helicopters) the unit is drawn working at the target and
    then travelling home, because a unit that teleported back would read as a
    bug.  The env's dynamics are untouched; only the drawn position is invented.
    """

    WORK_FRACTION = 0.35  # of the busy timer spent on-scene before heading home

    def __init__(self, env, transform, graph):
        self.env = env
        self.transform = transform
        self.graph = graph
        self.station_cell = {}
        for station in env.stations:
            x, y = self._station_xy(station)
            self.station_cell[station["station_id"]] = self.transform.xy_to_cell(x, y)
        self.uid = {}
        for rtype in RESOURCE_TYPES:
            for index, unit in enumerate(env.resources[rtype]):
                self.uid[id(unit)] = f"{rtype}:{index}"
        self.tracks = {}       # uid -> current dispatch geometry
        self.previous = {}     # uid -> previous tick's raw state
        self._route_cache = {}

    def _station_xy(self, station):
        from pyproj import Transformer
        if not hasattr(self, "_to_grid"):
            self._to_grid = Transformer.from_crs(
                "EPSG:4326", self.env.meta["crs"], always_xy=True
            )
        return self._to_grid.transform(station["lon"], station["lat"])

    def _ground_route(self, station_id, zone_id):
        key = (station_id, zone_id)
        if key in self._route_cache:
            return self._route_cache[key]
        import networkx as nx
        station = next(s for s in self.env.stations if s["station_id"] == station_id)
        source = int(station.get("road_node", -1))
        target = int(self.env.zones[zone_id]["road_node"])
        polyline = []
        if source >= 0:
            try:
                nodes = nx.shortest_path(self.graph, source, target, weight="travel_time")
                polyline = _polyline_from_nodes(self.graph, nodes, self.transform)
            except Exception:
                polyline = []
        if not polyline:
            zone = self.env.zones[zone_id]
            polyline = [self.station_cell[station_id],
                        (float(zone["centroid_col"]), float(zone["centroid_row"]))]
        result = _simplify(polyline)
        self._route_cache[key] = result
        return result

    def _begin(self, uid, rtype, unit):
        """Record geometry for a dispatch that started on this tick."""
        station_id = unit["station_id"]
        zone_id = unit["target_zone"]
        aim_row, aim_col = _effect_target_point(self.env.sim, self.env.zones[zone_id])
        aim = (float(aim_col), float(aim_row))
        if rtype in GROUND_RESOURCE_TYPES:
            polyline = list(self._ground_route(station_id, zone_id))
            # The road graph ends at the zone's nearest node; walk the last leg
            # to the actual fire so the vehicle stops where the effect lands.
            if polyline and math.dist(polyline[-1], aim) > 1.0:
                polyline.append(aim)
        else:
            polyline = [self.station_cell[station_id], aim]
        self.tracks[uid] = {
            "route": polyline,
            "lengths": _cumulative_lengths(polyline),
            "travel_total": max(1, int(unit.get("pending_travel_ticks") or 1)),
            "prep_total": max(0, int(unit["remaining_ticks"] if unit["state"] == "preparing" else 0)),
            "aim": aim,
            "station": self.station_cell[station_id],
            "busy_total": max(1, int(self._busy_ticks(rtype))),
            "zone": zone_id,
        }

    def _busy_ticks(self, rtype):
        if self.env.traffic_mode == "synthetic":
            return self.env.delay_config[rtype]["post_effect_busy_ticks"]
        return HELICOPTER_RELOAD_TICKS if rtype == "helicopter" else DEPLOYED_BUSY_TICKS

    def snapshot(self):
        """Positions and phases for every unit, for the tick just completed."""
        out = []
        for rtype in RESOURCE_TYPES:
            for unit in self.env.resources[rtype]:
                uid = self.uid[id(unit)]
                state = unit["state"]
                was = self.previous.get(uid)
                if state != "available" and (was is None or was == "available"):
                    self._begin(uid, rtype, unit)
                self.previous[uid] = state
                track = self.tracks.get(uid)

                if state == "available" or track is None:
                    station = self.station_cell.get(unit["station_id"], (0.0, 0.0))
                    out.append({"u": uid, "s": "idle", "x": _round(station[0], 1),
                                "y": _round(station[1], 1)})
                    continue

                remaining = int(unit["remaining_ticks"])
                if state == "preparing":
                    point, phase, progress = track["station"], "prep", self._phase_progress(
                        remaining, track["prep_total"])
                elif state == "traveling":
                    progress = self._phase_progress(remaining, track["travel_total"])
                    point = _point_along(track["route"], track["lengths"], progress) or track["station"]
                    phase = "move"
                elif state == "arrival_setup":
                    point, phase, progress = track["aim"], "setup", 0.0
                else:  # deployed: on-scene, then the busy/reload timer's return leg
                    progress = self._phase_progress(remaining, track["busy_total"])
                    if progress <= self.WORK_FRACTION:
                        point, phase = track["aim"], "work"
                    else:
                        span = (progress - self.WORK_FRACTION) / max(1e-6, 1.0 - self.WORK_FRACTION)
                        point = _point_along(track["route"], track["lengths"], 1.0 - span) \
                            or track["station"]
                        phase = "return"

                entry = {"u": uid, "s": phase, "x": _round(point[0], 1), "y": _round(point[1], 1)}
                if track["zone"] is not None:
                    entry["z"] = int(track["zone"])
                if phase in ("move", "return"):
                    entry["p"] = _round(progress, 3)
                out.append(entry)
        return out

    @staticmethod
    def _phase_progress(remaining, total):
        total = max(1, int(total))
        return max(0.0, min(1.0, (total - remaining) / total))


# --- model loading ----------------------------------------------------------

def load_model(checkpoint_path, obs, env, device):
    """Load a relative-action checkpoint, widening v8's MLP input if needed."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RelativeInfernoModel(
        n_grid_channels=obs["grid"].shape[0],
        n_scalars=len(SCALAR_KEYS),
        n_resources=len(RESOURCE_TYPES),
        n_zones=env.n_zones,
    ).to(device)

    key = "mlp.net.0.weight"
    adapted = None
    if key in state:
        checkpoint_width = state[key].shape[1]
        expected_width = model.state_dict()[key].shape[1]
        if checkpoint_width < expected_width:
            widened = torch.zeros_like(model.state_dict()[key])
            widened[:, :checkpoint_width] = state[key]
            state[key] = widened
            adapted = (checkpoint_width, expected_width)
        elif checkpoint_width > expected_width:
            raise SystemExit(
                f"checkpoint expects {checkpoint_width} scalars but the env emits "
                f"{expected_width}; it was trained on a newer observation space"
            )
    model.load_state_dict(state)
    model.eval()
    return model, adapted


def choose_action(model, obs, env, device):
    """The deterministic policy from eval_relative.py: argmax, masked by roster."""
    available = {rtype: int(obs["scalars"][f"{rtype}_available"]) for rtype in RESOURCE_TYPES}
    ready = dict(available)   # what the policy could actually see when it chose
    actions, semantics = [], []
    critic_value = 0.0
    for slot in range(MAX_DISPATCH_SLOTS):
        logits, value, _classification, target_zones = _forward(model, obs, env, device)
        if slot == 0:
            # The critic's estimate of this state, taken from the same forward
            # pass as the first decision rather than a second pass over the grid.
            critic_value = float(value.item())
        resource_logits = logits["resource_type"][0].clone()
        mask = torch.tensor([available[rtype] > 0 for rtype in RESOURCE_TYPES],
                            dtype=torch.bool, device=device)
        resource_logits[~mask] = -1e9
        if not bool(mask.any()):
            break
        resource_idx = int(torch.argmax(resource_logits))
        target_idx = int(torch.argmax(logits["target"][0, resource_idx]))
        action = decode_action(resource_idx, target_idx, target_zones)
        if action is None:
            semantics.append({"r": RESOURCE_TYPES[resource_idx], "t": TARGET_TYPES[target_idx],
                              "held": True})
            break
        available[RESOURCE_TYPES[resource_idx]] -= 1
        actions.append(action)
        semantics.append({"r": RESOURCE_TYPES[resource_idx], "t": TARGET_TYPES[target_idx],
                          "z": int(action[1])})
    return actions, semantics, critic_value, ready


# --- rollout ----------------------------------------------------------------

def rollout(model, env, scenario_key, device, seed, post_ticks=0):
    scenario = SCENARIOS[scenario_key]
    transform = PixelTransform(env.meta)
    obs = env.reset(seed=seed, use_real_weather=True, **scenario["reset"])
    tracker = UnitTracker(env, transform, env.road_graph)

    width = env.width

    # Fire is diffed as a flat int8 array (drawn state per cell, 0 = not drawn)
    # and compared vectorised each tick.  A per-cell Python loop here would run
    # into the tens of millions of iterations once a large fire has burned.
    def drawn_array():
        flat = env.sim.state.reshape(-1)
        return np.where(np.isin(flat, DRAWN_STATES), flat, 0).astype(np.int8)

    previous_drawn = drawn_array()

    ticks = []
    cumulative = 0.0
    total_destroyed = 0
    info = {}
    done = False

    ignitions = [{"row": int(r), "col": int(c)} for r, c in env.ignition_points]
    initial_indices = np.flatnonzero(previous_drawn)
    initial = [[int(index), int(previous_drawn[index])] for index in initial_indices]

    # The episode itself ends the instant the fire is contained, which means the
    # units that just put it out are still mid-cycle -- their post-effect busy /
    # reload timers never play back. Recording a short, dispatch-free aftermath
    # lets the view show the fleet standing down. These ticks are flagged
    # "post" and are excluded from the scored outcome below.
    scored = {"tick": 0, "reward": 0.0, "destroyed": 0, "contained": False, "timeout": False}
    remaining_post = 0

    with torch.no_grad():
        while True:
            if done:
                if remaining_post <= 0:
                    break
                remaining_post -= 1
                ready = {rtype: int(obs["scalars"][f"{rtype}_available"]) for rtype in RESOURCE_TYPES}
                actions, semantics, value, is_post = [], [], 0.0, True
            else:
                actions, semantics, value, ready = choose_action(model, obs, env, device)
                is_post = False
            obs, reward, done, info = env.step(actions)
            cumulative += reward
            total_destroyed += info["buildings_destroyed"]

            if not is_post:
                # Everything scored comes from the episode proper; the aftermath
                # ticks below are for the animation only.
                scored = {"tick": int(info["tick"]), "reward": cumulative,
                          "destroyed": int(total_destroyed),
                          "contained": bool(info["contained"]),
                          "timeout": bool(info["timeout"])}
                if done and post_ticks:
                    remaining_post = post_ticks

            current_drawn = drawn_array()
            changed = np.flatnonzero(current_drawn != previous_drawn)
            changed_states = current_drawn[changed]
            added = [[int(index), int(value)]
                     for index, value in zip(changed[changed_states != 0],
                                             changed_states[changed_states != 0])]
            removed = [int(index) for index in changed[changed_states == 0]]
            previous_drawn = current_drawn

            effects = [{
                "r": event["resource_type"],
                "z": int(event["zone"]),
                "row": int(event.get("row", -1)),
                "col": int(event.get("col", -1)),
                "n": int(event["cells_affected"]),
                "ok": bool(event["success"]),
            } for event in info["resource_events"]]

            dispatches = []
            for item in info["dispatch"]:
                entry = {"r": item["resource_type"], "z": int(item["target_zone"]),
                         "st": item["status"]}
                if item["status"] == "dispatched":
                    entry["eta"] = int(item["eta_ticks"])
                    entry["travel_s"] = _round(item["travel_time_s"], 1)
                    entry["station"] = item["station_id"]
                dispatches.append(entry)

            losses = [{"row": int(e["row"]), "col": int(e["col"]),
                       "mult": _round(e["multiplier"]), "evac": bool(e["evacuated"])}
                      for e in info["building_destruction_events"]]

            weather = info["weather"]
            counts = info["state_counts"]
            record = {
                "t": int(info["tick"]),
                "wind": _round(weather["wind_speed_mph"], 1),
                "dir": _round(weather["wind_direction_deg"], 1),
                "rh": _round(weather["humidity_pct"], 2),
                "reward": _round(reward, 1),
                "cum": _round(cumulative, 1),
                "value": _round(value, 2),
                "destroyed": int(info["buildings_destroyed"]),
                "destroyed_total": int(total_destroyed),
                "active": int(counts["Threat"] + counts["Blaze"]),
                "blaze": int(counts["Blaze"]),
                "burned": int(counts["Burned Out"]),
                "fire": {"add": added, "del": removed},
                "units": tracker.snapshot(),
                # Roster state as the policy saw it when it chose, which is not
                # the same as the post-step state the unit snapshot reflects.
                "ready": {rtype: int(ready[rtype]) for rtype in RESOURCE_TYPES},
                "actions": semantics,
                "dispatch": dispatches,
                "effects": effects,
                "losses": losses,
            }
            if is_post:
                record["post"] = 1
            ticks.append(record)

    roster = {rtype: len(env.resources[rtype]) for rtype in RESOURCE_TYPES}
    stations = [{
        "id": station["station_id"],
        "name": station["station_name"],
        "mode": station["travel_mode"],
        "roster": station["roster"],
        "x": _round(tracker.station_cell[station["station_id"]][0], 1),
        "y": _round(tracker.station_cell[station["station_id"]][1], 1),
    } for station in env.stations]

    zones = [{
        "id": zone["zone_id"],
        "r0": zone["row_range"][0], "r1": zone["row_range"][1],
        "c0": zone["col_range"][0], "c1": zone["col_range"][1],
        "water": bool(zone["is_water"]),
        "buildings": int(zone["building_cells"]),
    } for zone in env.zones]

    routes = {}
    for uid, track in tracker.tracks.items():
        routes[uid] = [[_round(x, 1), _round(y, 1)] for x, y in track["route"]]

    return {
        "scenario": {
            "key": scenario_key,
            "label": scenario["label"],
            "sublabel": scenario["sublabel"],
            "kind": scenario["kind"],
            "ignitions": ignitions,
        },
        "outcome": {
            "ticks": scored["tick"],
            "reward": _round(scored["reward"], 1),
            "destroyed": scored["destroyed"],
            "contained": scored["contained"],
            "timeout": scored["timeout"],
            "frames": len(ticks),
        },
        "roster": roster,
        "stations": stations,
        "zones": zones,
        "routes": routes,
        "initial_fire": initial,
        "ticks": ticks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint",
                        default=os.path.join(os.path.dirname(PROJECT_ROOT),
                                             "best_model", "inferno_best_model.pt"))
    parser.add_argument("--scenarios", default="anchor,mandeville,getty,multi",
                        help="comma-separated: " + ",".join(SCENARIOS))
    parser.add_argument("--traffic-mode", default="legacy", choices=("legacy", "synthetic"),
                        help="legacy reproduces the dynamics the v8 weights were trained on")
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--post-ticks", type=int, default=14,
                        help="dispatch-free aftermath ticks recorded after containment, so the "
                             "view can show units standing down; excluded from the scored outcome")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "src", "viz", "trajectories.json"))
    args = parser.parse_args()

    keys = [key.strip() for key in args.scenarios.split(",") if key.strip()]
    unknown = [key for key in keys if key not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario(s): {unknown}; choose from {list(SCENARIOS)}")

    device = torch.device("cpu")
    env = InfernoEnv(seed=args.seed, traffic_mode=args.traffic_mode)
    obs = env.reset(seed=args.seed)
    model, adapted = load_model(args.checkpoint, obs, env, device)
    if adapted:
        print(f"[export] adapted checkpoint MLP input {adapted[0]} -> {adapted[1]} scalars "
              f"(v9 traffic scalars zero-initialised; identical function on the v8 inputs)")

    episodes = []
    for key in keys:
        print(f"[export] rolling out {key} ...", flush=True)
        episode = rollout(model, env, key, device, args.seed, post_ticks=args.post_ticks)
        outcome = episode["outcome"]
        print(f"[export]   {key}: {outcome['ticks']} ticks  reward={outcome['reward']:.1f}  "
              f"destroyed={outcome['destroyed']}  contained={outcome['contained']}  "
              f"({outcome['frames']} frames incl. aftermath)")
        episodes.append(episode)

    payload = {
        "meta": {
            "checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_name": os.path.basename(args.checkpoint),
            "model": "RelativeInfernoModel (v8 fire-relative actions)",
            "scalar_adapter": None if not adapted else f"{adapted[0]}->{adapted[1]}",
            "traffic_mode": args.traffic_mode,
            "seed": args.seed,
            "grid": {"width": env.width, "height": env.height,
                     "cell_size_m": float(env.meta["cell_size_m"])},
            "tick_minutes": TICK_DURATION_MINUTES,
            "max_ticks": MAX_TICKS,
            "fire_start_utc": FIRE_START_UTC.isoformat(),
            "states": {"threat": int(THREAT), "blaze": int(BLAZE), "burned": int(BURNED_OUT)},
            "resource_types": list(RESOURCE_TYPES),
            "ground_types": list(GROUND_RESOURCE_TYPES),
            "air_types": list(AIR_RESOURCE_TYPES),
            "target_types": list(TARGET_TYPES),
        },
        "episodes": episodes,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    print(f"[export] wrote {args.out} ({os.path.getsize(args.out) / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
