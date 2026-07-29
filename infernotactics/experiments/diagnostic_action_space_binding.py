"""
Read-only diagnostic: is the one-dispatch-per-tick action space a binding
constraint, or is fleet availability / travel time the actual bottleneck?

NO training, NO gradient updates, NO model loaded -- runs HeuristicPolicy
(the most "eager" dispatcher available: it commits to whichever qualifying
resource type reaches its target soonest, every single tick something
qualifies) against InfernoEnv via plain step() calls, and records per-tick
dispatch outcomes directly from step()'s info dict.

Deliberately does NOT touch models/checkpoints_multiscenario_v7 or any file
the concurrent v7 training run reads/writes -- this only reads env/heuristic
code and writes to logs/diag_actionspace_v1_results.json.

    python -m src.train.diagnostic_action_space_binding
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import (  # noqa: E402
    LAYER_INDEX,
    MULTI_IGNITION_TRAINING_SCENARIO,
    TRAINING_IGNITION_POINT,
    InfernoEnv,
    RESOURCE_TYPES,
    flatten_scalars,
)
from train.heuristic_policy import HeuristicPolicy  # noqa: E402

STONE_CANYON_IGNITION_POINT = MULTI_IGNITION_TRAINING_SCENARIO[2]  # (159, 483)

N_EPISODES = 5


def _run_episode(env, policy, ignition_point, seed):
    obs = env.reset(ignition_point=ignition_point, scenario="single", seed=seed, use_real_weather=True)
    done = False
    tick = 0
    dispatch_attempts = 0
    status_counts = {"dispatched": 0, "zone_unreachable": 0, "no_unit_available": 0}
    effect_failed = 0  # dispatched, arrived, but n_affected == 0 (e.g. trench hit active fire)
    effect_ok = 0
    blaze_series = []
    helicopter_active_series = []  # count of helicopter units not "available" each tick

    # Per-resource-type busy-cycle tracking (generalizes the old
    # helicopter-only version to all 4 types): for each unit, the tick it
    # left "available" and the tick it returned to "available", plus the
    # eta_ticks the dispatch itself reported (the travel-out leg) so the
    # remainder (cycle_length - eta_ticks) isolates the fixed on-scene +
    # return/reload leg (DEPLOYED_BUSY_TICKS or HELICOPTER_RELOAD_TICKS).
    dispatch_tick_by_unit = {rtype: {} for rtype in RESOURCE_TYPES}
    cycle_lengths = {rtype: [] for rtype in RESOURCE_TYPES}
    eta_ticks_by_type = {rtype: [] for rtype in RESOURCE_TYPES}
    available_series = {rtype: [] for rtype in RESOURCE_TYPES}

    while not done:
        # HeuristicPolicy leaves both action heads all-zero for a true noop;
        # argmax of an all-zero vector is index 0, indistinguishable from
        # "choosing resource 0". Call its internal _decide() directly rather
        # than trusting argmax-decoded logits for this one policy.
        fire_state = obs["grid"][-1]
        building_density = obs["grid"][LAYER_INDEX["building_density"]]
        population_density = obs["grid"][LAYER_INDEX["population_density"]]
        available = {rtype: obs["scalars"][f"{rtype}_available"] for rtype in RESOURCE_TYPES}
        rtype, zone_id = policy._decide(fire_state, building_density, population_density, available)

        for r in RESOURCE_TYPES:
            available_series[r].append(available[r])

        # snapshot every type's unit states before stepping, to track cycle length
        before_states = {r: [u["state"] for u in env.resources[r]] for r in RESOURCE_TYPES}

        action = (rtype, zone_id) if rtype is not None else None
        obs, reward, done, info = env.step(action)
        tick += 1

        if rtype is not None:
            dispatch_attempts += 1
            status = info["dispatch"]["status"]
            status_counts[status] += 1
            if status == "dispatched":
                eta_ticks_by_type[rtype].append(info["dispatch"]["eta_ticks"])

        for ev in info["resource_events"]:
            if ev["success"]:
                effect_ok += 1
            else:
                effect_failed += 1

        counts = info["state_counts"]
        blaze_series.append(counts["Blaze"])

        for r in RESOURCE_TYPES:
            after_r = [u["state"] for u in env.resources[r]]
            if r == "helicopter":
                helicopter_active_series.append(sum(1 for s in after_r if s != "available"))
            for i, (before_s, after_s) in enumerate(zip(before_states[r], after_r)):
                if before_s == "available" and after_s != "available":
                    dispatch_tick_by_unit[r][i] = tick
                if before_s != "available" and after_s == "available" and i in dispatch_tick_by_unit[r]:
                    cycle_lengths[r].append(tick - dispatch_tick_by_unit[r].pop(i))

    return {
        "ticks": tick,
        "dispatch_attempts": dispatch_attempts,
        "status_counts": status_counts,
        "effect_ok": effect_ok,
        "effect_failed": effect_failed,
        "blaze_series": blaze_series,
        "max_blaze": max(blaze_series) if blaze_series else 0,
        "helicopter_max_concurrent": max(helicopter_active_series) if helicopter_active_series else 0,
        "cycle_lengths": cycle_lengths,
        "eta_ticks_by_type": eta_ticks_by_type,
        "available_series": available_series,
        "contained": info["contained"],
    }


def _summarize(episodes, label):
    ticks = [e["ticks"] for e in episodes]
    dispatches = [e["dispatch_attempts"] for e in episodes]
    dispatched = [e["status_counts"]["dispatched"] for e in episodes]
    unreachable = [e["status_counts"]["zone_unreachable"] for e in episodes]
    no_unit = [e["status_counts"]["no_unit_available"] for e in episodes]
    effect_failed = [e["effect_failed"] for e in episodes]
    effect_ok = [e["effect_ok"] for e in episodes]
    heli_max = [e["helicopter_max_concurrent"] for e in episodes]

    # ticks-to-uncontainable proxy: first tick where Blaze count exceeds the
    # combined water_team+helicopter fleet size (the two suppression-capable
    # types) -- i.e. more simultaneously active fire than could in principle
    # be addressed even with a perfect one-unit-per-fire-front assignment.
    # Computed per-episode in main() and passed in via "first_exceed_tick".
    first_exceed_ticks = [e.get("first_exceed_tick") for e in episodes]

    per_type = {}
    for rtype in RESOURCE_TYPES:
        all_cycles = [c for e in episodes for c in e["cycle_lengths"][rtype]]
        all_etas = [t for e in episodes for t in e["eta_ticks_by_type"][rtype]]
        all_avail = [a for e in episodes for a in e["available_series"][rtype]]
        mean_cycle = float(np.mean(all_cycles)) if all_cycles else None
        mean_eta = float(np.mean(all_etas)) if all_etas else None
        per_type[rtype] = {
            "n_dispatches": len(all_etas),
            "mean_busy_ticks_total": mean_cycle,
            "mean_travel_out_ticks": mean_eta,
            # remainder after travel-out: on-scene effect (instantaneous, 0
            # ticks by construction -- see step()/_advance_resources) plus
            # the fixed post-arrival return/reload leg (DEPLOYED_BUSY_TICKS
            # or HELICOPTER_RELOAD_TICKS).
            "mean_onscene_plus_return_ticks": (mean_cycle - mean_eta) if (mean_cycle is not None and mean_eta is not None) else None,
            "mean_available_units_per_tick": float(np.mean(all_avail)) if all_avail else None,
        }

    summary = {
        "label": label,
        "n_episodes": len(episodes),
        "mean_ticks": float(np.mean(ticks)),
        "mean_dispatch_attempts": float(np.mean(dispatches)),
        "mean_dispatched": float(np.mean(dispatched)),
        "mean_zone_unreachable": float(np.mean(unreachable)),
        "mean_no_unit_available": float(np.mean(no_unit)),
        "mean_effect_ok": float(np.mean(effect_ok)),
        "mean_effect_failed": float(np.mean(effect_failed)),
        "dispatched_fraction_of_ticks": float(np.mean([d / t for d, t in zip(dispatched, ticks)])),
        "no_unit_available_fraction_of_ticks": float(np.mean([n / t for n, t in zip(no_unit, ticks)])),
        "helicopter_max_concurrent_mean": float(np.mean(heli_max)),
        "helicopter_max_concurrent_max": int(max(heli_max)) if heli_max else 0,
        "contained_rate": float(np.mean([e["contained"] for e in episodes])),
        "first_exceed_fleet_tick": first_exceed_ticks,
        "per_resource_type": per_type,
    }
    return summary


def main():
    from env.inferno_env import DEPLOYED_BUSY_TICKS, HELICOPTER_RELOAD_TICKS, TICK_DURATION_MINUTES

    print(f"TICK_DURATION_MINUTES (inferno_env.py) = {TICK_DURATION_MINUTES} min/tick "
          f"({TICK_DURATION_MINUTES * 60:.0f} s/tick). "
          f"fire_sim.py itself has no real-time-per-tick constant of its own -- "
          f"it's tick-native (e.g. BURN_DURATION_TICKS=4); only inferno_env.py's "
          f"TICK_DURATION_MINUTES maps ticks to real seconds/minutes, used for real "
          f"road-travel-time -> eta_ticks conversion and the real-weather lookup.")
    print(f"DEPLOYED_BUSY_TICKS (ground on-scene+return) = {DEPLOYED_BUSY_TICKS} ticks; "
          f"HELICOPTER_RELOAD_TICKS (return+reload) = {HELICOPTER_RELOAD_TICKS} ticks.")

    env = InfernoEnv(seed=0)
    env.reset(seed=0)
    policy = HeuristicPolicy(env)

    water_heli_fleet = len(env.resources["water_team"]) + len(env.resources["helicopter"])

    scenarios = {
        "single": TRAINING_IGNITION_POINT,
        "stone_canyon": STONE_CANYON_IGNITION_POINT,
    }

    all_results = {}
    for name, point in scenarios.items():
        episodes = []
        for ep in range(N_EPISODES):
            result = _run_episode(env, policy, point, seed=ep)
            # first tick Blaze count exceeds combined water+helicopter fleet size
            first_exceed = next((i + 1 for i, b in enumerate(result["blaze_series"]) if b > water_heli_fleet), None)
            result["first_exceed_tick"] = first_exceed
            episodes.append(result)
        summary = _summarize(episodes, name)
        summary["water_helicopter_fleet_size"] = water_heli_fleet
        all_results[name] = summary
        print(f"\n=== {name} ===")
        for k, v in summary.items():
            if k not in ("first_exceed_fleet_tick", "per_resource_type"):
                print(f"  {k}: {v}")
        print(f"  first_exceed_fleet_tick per episode: {summary['first_exceed_fleet_tick']}")
        print("  per_resource_type:")
        for rtype, stats in summary["per_resource_type"].items():
            print(f"    {rtype}: {stats}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "logs", "diag_actionspace_v2_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
