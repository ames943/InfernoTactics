"""
Fixed-policy resource-TYPE sweep -- environment diagnostic ONLY, no training,
no gradient updates, no model of any kind. Answers: holding zone-selection
(the existing HeuristicPolicy's per-type rule, see heuristic_policy.py) and
everything else constant, how much does the choice of resource TYPE alone
matter to outcome on the frozen v5 8-station environment?

Run tag "diag_restype_v1" -- writes only to logs/diag_restype_v1_*.{csv,json}.
Does not touch models/checkpoints_multi_v5/, models/checkpoints_bisect/,
train_log_*.csv/eval_log_*.csv, or any path a training run reads/writes, so
it's safe to run alongside the live bisect job.

Four single-type policies (water_team-only, trench_crew-only,
rescue_vehicle-only, helicopter-only) plus a random-type baseline (resource
type sampled uniformly per tick, same per-type zone rule as the single-type
runs) are each rolled out for N_EPISODES episodes on scenario='single'
(TRAINING_IGNITION_POINT / Skull Rock), real weather, deterministic seed set
(seed=SEED_BASE+episode_index, matching eval_policy()'s convention). No
learning anywhere -- every "policy" here is a fixed rule, not a model.

Per-type zone-selection logic is lifted directly from
HeuristicPolicy._decide() (heuristic_policy.py) so the type-restricted
policies use the exact same "where would this type go" rule the full
heuristic already uses -- only WHICH types are eligible each tick differs
across the 5 runs.

    python -m src.train.diagnostic_resource_type_sweep
"""

import json
import math
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.fire_sim import BLAZE, FUEL, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    BUILDING_PRESENCE_THRESHOLD,
    LAYER_INDEX,
    RESOURCE_TYPES,
    TRAINING_IGNITION_POINT,
    InfernoEnv,
)

RUN_TAG = "diag_restype_v1"
N_EPISODES = 5
SEED_BASE = 0
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")

_DILATION_STRUCTURE = np.ones((3, 3), dtype=bool)  # matches heuristic_policy.py's 8-connected Moore neighborhood


def _masks(fire_state, building_density):
    active_mask = np.isin(fire_state, (THREAT, BLAZE))
    fuel_mask = fire_state == FUEL
    threatened_building_mask = (building_density > BUILDING_PRESENCE_THRESHOLD) & active_mask
    fuel_adjacent_to_fire = binary_dilation(active_mask, structure=_DILATION_STRUCTURE) & fuel_mask
    return active_mask, fuel_adjacent_to_fire, threatened_building_mask


def _type_candidate(rtype, env, available, active_mask, fuel_adjacent_to_fire, threatened_building_mask,
                     population_density):
    """Same per-type rule as HeuristicPolicy._decide, restricted to a single
    resource_type: water_team/helicopter -> nearest zone with an active fire;
    trench_crew -> nearest zone with fire-adjacent Fuel but no active fire yet;
    rescue_vehicle -> zone maximizing threatened-population-density/travel_time.
    Returns (zone_id, travel_time_s) or None if no valid target this tick."""
    if available[rtype] < 1:
        return None

    best, best_score = None, None
    for zone in env.zones:
        zid = zone["zone_id"]
        r0, r1 = zone["row_range"]
        c0, c1 = zone["col_range"]
        t = env.zone_travel_time_s[rtype][zid]
        if not math.isfinite(t):
            continue

        if rtype in ("water_team", "helicopter"):
            if not active_mask[r0:r1, c0:c1].any():
                continue
            if best is None or t < best[1]:
                best = (zid, t)
        elif rtype == "trench_crew":
            if active_mask[r0:r1, c0:c1].any() or not fuel_adjacent_to_fire[r0:r1, c0:c1].any():
                continue
            if best is None or t < best[1]:
                best = (zid, t)
        else:  # rescue_vehicle
            zone_threatened = threatened_building_mask[r0:r1, c0:c1]
            if not zone_threatened.any():
                continue
            pop = float(population_density[r0:r1, c0:c1][zone_threatened].max())
            score = pop / max(t, 1.0)
            if best_score is None or score > best_score:
                best, best_score = (zid, t), score
    return best


def run_episode(env, episode_seed, mode, rng):
    """mode: one of RESOURCE_TYPES (single-type-only for the whole episode)
    or 'random' (resource type resampled uniformly every tick, same per-type
    zone rule as the single-type runs)."""
    obs = env.reset(seed=episode_seed, scenario="single", ignition_point=TRAINING_IGNITION_POINT,
                     use_real_weather=True)

    total_reward = 0.0
    buildings_destroyed = 0
    dispatch_attempts = 0
    dispatch_ok = 0
    dispatch_wasted_no_unit = 0
    dispatch_wasted_unreachable = 0
    effect_success_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}
    effect_wasted_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}

    done = False
    info = None
    while not done:
        fire_state = obs["grid"][-1]
        building_density = obs["grid"][LAYER_INDEX["building_density"]]
        population_density = obs["grid"][LAYER_INDEX["population_density"]]
        available = {rtype: obs["scalars"][f"{rtype}_available"] for rtype in RESOURCE_TYPES}

        active_mask, fuel_adjacent_to_fire, threatened_building_mask = _masks(fire_state, building_density)

        rtype = mode if mode in RESOURCE_TYPES else rng.choice(RESOURCE_TYPES)
        candidate = _type_candidate(rtype, env, available, active_mask, fuel_adjacent_to_fire,
                                     threatened_building_mask, population_density)
        action = (rtype, candidate[0]) if candidate is not None else None

        obs, reward, done, info = env.step(action)
        total_reward += reward

        if info["dispatch"] is not None:
            dispatch_attempts += 1
            status = info["dispatch"]["status"]
            if status == "dispatched":
                dispatch_ok += 1
            elif status == "no_unit_available":
                dispatch_wasted_no_unit += 1
            elif status == "zone_unreachable":
                dispatch_wasted_unreachable += 1

        for ev in info["resource_events"]:
            if ev["success"]:
                effect_success_by_type[ev["resource_type"]] += 1
            else:
                effect_wasted_by_type[ev["resource_type"]] += 1

        buildings_destroyed += info["buildings_destroyed"]

    return {
        "total_reward": total_reward,
        "buildings_destroyed": buildings_destroyed,
        "contained": info["contained"],
        "dispatch_attempts": dispatch_attempts,
        "dispatch_ok": dispatch_ok,
        "dispatch_wasted_no_unit": dispatch_wasted_no_unit,
        "dispatch_wasted_unreachable": dispatch_wasted_unreachable,
        "effect_success_by_type": effect_success_by_type,
        "effect_wasted_by_type": effect_wasted_by_type,
    }


def _aggregate(episodes):
    n = len(episodes)
    agg_effect_success = {rtype: sum(e["effect_success_by_type"][rtype] for e in episodes) for rtype in RESOURCE_TYPES}
    agg_effect_wasted = {rtype: sum(e["effect_wasted_by_type"][rtype] for e in episodes) for rtype in RESOURCE_TYPES}
    return {
        "avg_reward": sum(e["total_reward"] for e in episodes) / n,
        "avg_buildings_destroyed": sum(e["buildings_destroyed"] for e in episodes) / n,
        "containment_rate": sum(1 for e in episodes if e["contained"]) / n,
        "dispatch_attempts": sum(e["dispatch_attempts"] for e in episodes),
        "dispatch_ok": sum(e["dispatch_ok"] for e in episodes),
        "dispatch_wasted_no_unit": sum(e["dispatch_wasted_no_unit"] for e in episodes),
        "dispatch_wasted_unreachable": sum(e["dispatch_wasted_unreachable"] for e in episodes),
        "effect_success_by_type": agg_effect_success,
        "effect_wasted_by_type": agg_effect_wasted,
        "episode_rewards": [e["total_reward"] for e in episodes],
    }


def main():
    print(f"[{RUN_TAG}] Building InfernoEnv (scenario='single', Skull Rock, real weather)...")
    env = InfernoEnv(seed=0)
    env.reset(seed=0, ignition_point=TRAINING_IGNITION_POINT, use_real_weather=True)  # warm up routing

    modes = list(RESOURCE_TYPES) + ["random"]
    results = {}
    for mode_idx, mode in enumerate(modes):
        rng = np.random.default_rng(9000 + mode_idx)  # fixed per-mode seed, not Python hash() (PYTHONHASHSEED-unstable)
        episodes = [run_episode(env, SEED_BASE + ep, mode, rng) for ep in range(N_EPISODES)]
        results[mode] = _aggregate(episodes)

    print(f"\n{'mode':18s} {'avg_reward':>14s} {'avg_bldgs_destroyed':>20s} {'containment_rate':>18s}")
    print("-" * 74)
    for mode in modes:
        r = results[mode]
        label = f"{mode}-only" if mode in RESOURCE_TYPES else "random-type"
        print(f"{label:18s} {r['avg_reward']:14.1f} {r['avg_buildings_destroyed']:20.1f} {r['containment_rate']:17.0%}")

    print(f"\n{'mode':18s} " + " ".join(f"{rtype:>16s}" for rtype in RESOURCE_TYPES) + "  (effect success / (success+wasted), per type, aggregated over 5 episodes)")
    print("-" * (18 + 17 * len(RESOURCE_TYPES)))
    for mode in modes:
        r = results[mode]
        label = f"{mode}-only" if mode in RESOURCE_TYPES else "random-type"
        cells = []
        for rtype in RESOURCE_TYPES:
            ok = r["effect_success_by_type"][rtype]
            waste = r["effect_wasted_by_type"][rtype]
            total = ok + waste
            cells.append(f"{ok}/{total} ({ok/total:.0%})" if total else "0/0 (n/a)")
        print(f"{label:18s} " + " ".join(f"{c:>16s}" for c in cells))

    os.makedirs(LOG_DIR, exist_ok=True)
    out_path = os.path.join(LOG_DIR, f"{RUN_TAG}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results written to {out_path}")

    return results


if __name__ == "__main__":
    main()
