"""
Mechanics/perf harness for inferno_env.InfernoEnv -- NOT training. Takes
RANDOM actions for a handful of episodes to validate that reset/step/reward/
done behave sanely, and reports wall-clock timing so we have real numbers on
simulation speed before making any performance decisions.

    python -m src.env.test_inferno_env
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import (  # noqa: E402
    RESOURCE_COUNTS,
    RESOURCE_TYPES,
    InfernoEnv,
)

N_EPISODES = 5
DISPATCH_PROB = 0.7  # fraction of ticks the random policy attempts a dispatch (rest are noop)


def random_action(rng, n_zones):
    if rng.random() < DISPATCH_PROB:
        rtype = rng.choice(RESOURCE_TYPES)
        zone = int(rng.integers(n_zones))
        return (rtype, zone)
    return None


def run_episode(env, episode_seed):
    rng = np.random.default_rng(episode_seed)
    obs = env.reset(seed=episode_seed)

    total_reward = 0.0
    rewards = []
    dispatch_attempts = 0
    dispatch_ok = 0
    dispatch_wasted_no_unit = 0
    dispatch_wasted_unreachable = 0
    resource_effect_success = 0
    resource_effect_wasted = 0
    effect_success_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}
    effect_wasted_by_type = {rtype: 0 for rtype in RESOURCE_TYPES}
    fires_extinguished_events = 0
    buildings_destroyed = 0
    buildings_destroyed_evacuated = 0
    availability_snapshots = []  # (tick, {rtype: n_available})
    tick_wall_times = []

    done = False
    t = 0
    while not done:
        action = random_action(rng, env.n_zones)

        tick_t0 = time.perf_counter()
        obs, reward, done, info = env.step(action)
        tick_wall_times.append(time.perf_counter() - tick_t0)

        total_reward += reward
        rewards.append(reward)

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
                resource_effect_success += 1
                effect_success_by_type[ev["resource_type"]] += 1
                if ev["resource_type"] == "water_team":
                    fires_extinguished_events += 1
            else:
                resource_effect_wasted += 1
                effect_wasted_by_type[ev["resource_type"]] += 1

        buildings_destroyed += info["buildings_destroyed"]
        buildings_destroyed_evacuated += info["buildings_destroyed_evacuated"]

        available_now = {
            rtype: sum(1 for u in env.resources[rtype] if u["state"] == "available")
            for rtype in RESOURCE_TYPES
        }
        availability_snapshots.append(available_now)

        t += 1

    return {
        "ticks": t,
        "total_reward": total_reward,
        "rewards": rewards,
        "contained": info["contained"],
        "timeout": info["timeout"],
        "final_state_counts": info["state_counts"],
        "dispatch_attempts": dispatch_attempts,
        "dispatch_ok": dispatch_ok,
        "dispatch_wasted_no_unit": dispatch_wasted_no_unit,
        "dispatch_wasted_unreachable": dispatch_wasted_unreachable,
        "resource_effect_success": resource_effect_success,
        "resource_effect_wasted": resource_effect_wasted,
        "effect_success_by_type": effect_success_by_type,
        "effect_wasted_by_type": effect_wasted_by_type,
        "fires_extinguished_events": fires_extinguished_events,
        "buildings_destroyed": buildings_destroyed,
        "buildings_destroyed_evacuated": buildings_destroyed_evacuated,
        "availability_snapshots": availability_snapshots,
        "tick_wall_times": tick_wall_times,
    }


def check_resources_recover(availability_snapshots):
    """Did every resource type return to full availability at least once
    after having been reduced below full?"""
    ever_reduced = {rtype: False for rtype in RESOURCE_TYPES}
    ever_recovered_after_reduction = {rtype: False for rtype in RESOURCE_TYPES}
    for snap in availability_snapshots:
        for rtype in RESOURCE_TYPES:
            full = RESOURCE_COUNTS[rtype]
            if snap[rtype] < full:
                ever_reduced[rtype] = True
            elif snap[rtype] == full and ever_reduced[rtype]:
                ever_recovered_after_reduction[rtype] = True
    return ever_reduced, ever_recovered_after_reduction


def main():
    print("Building InfernoEnv (loads grid + road routing graph once, shared across all episodes)...")
    t0 = time.perf_counter()
    env = InfernoEnv(seed=123)
    env_init_s = time.perf_counter() - t0
    print(f"Env init time: {env_init_s:.2f}s")
    print(f"Zones: {env.n_zones}, travel times (s): "
          f"min={min(t for t in env.zone_travel_time_s if math.isfinite(t)):.0f}  "
          f"max={max(t for t in env.zone_travel_time_s if math.isfinite(t)):.0f}  "
          f"unreachable={sum(1 for t in env.zone_travel_time_s if not math.isfinite(t))}")
    print()

    all_results = []
    all_reward_values = []
    all_tick_times = []

    for ep in range(N_EPISODES):
        ep_t0 = time.perf_counter()
        result = run_episode(env, episode_seed=1000 + ep)
        ep_wall_s = time.perf_counter() - ep_t0

        all_results.append(result)
        all_reward_values.extend(result["rewards"])
        all_tick_times.extend(result["tick_wall_times"])

        print(f"--- Episode {ep} ---")
        print(f"  wall time: {ep_wall_s:.2f}s over {result['ticks']} ticks "
              f"({ep_wall_s / result['ticks'] * 1000:.1f} ms/tick)")
        print(f"  terminated: contained={result['contained']}  timeout={result['timeout']}")
        print(f"  total_reward: {result['total_reward']:.1f}")
        print(f"  final state counts: {result['final_state_counts']}")
        print(f"  dispatch attempts: {result['dispatch_attempts']} "
              f"(ok={result['dispatch_ok']}, no_unit={result['dispatch_wasted_no_unit']}, "
              f"unreachable_zone={result['dispatch_wasted_unreachable']})")
        per_type = ", ".join(
            f"{rtype}: {result['effect_success_by_type'][rtype]} ok / "
            f"{result['effect_wasted_by_type'][rtype]} wasted"
            for rtype in RESOURCE_TYPES
        )
        print(f"  resource arrival effects: success={result['resource_effect_success']} "
              f"wasted={result['resource_effect_wasted']}  ({per_type})")
        print(f"  buildings destroyed: {result['buildings_destroyed']} "
              f"(evacuated first: {result['buildings_destroyed_evacuated']})")
        print()

    # --- Aggregate timing ----------------------------------------------------
    total_ticks = sum(r["ticks"] for r in all_results)
    total_wall = sum(all_tick_times)
    print("=== Timing summary ===")
    print(f"Total ticks across {N_EPISODES} episodes: {total_ticks}")
    print(f"Total step() wall time: {total_wall:.2f}s")
    print(f"Mean: {total_wall / total_ticks * 1000:.2f} ms/tick "
          f"({total_ticks / total_wall:.1f} ticks/sec)")
    print(f"Env one-time init (grid + routing graph): {env_init_s:.2f}s (amortized across all episodes)")
    print()

    # --- Sanity / anomaly checks ----------------------------------------------
    print("=== Sanity checks ===")
    issues = []

    n_contained = sum(1 for r in all_results if r["contained"])
    print(f"Episodes fully contained by random luck: {n_contained}/{N_EPISODES}")
    n_over_max = sum(1 for r in all_results if r["ticks"] > 150)
    if n_over_max:
        issues.append(f"{n_over_max} episode(s) ran past MAX_TICKS -- done flag not firing correctly")

    distinct_rewards = len(set(round(r, 3) for r in all_reward_values))
    print(f"Distinct reward values seen across all steps: {distinct_rewards} "
          f"(of {len(all_reward_values)} steps)")
    if distinct_rewards <= 1:
        issues.append("Reward is constant regardless of action -- reward function is not wired up correctly")

    for rtype in RESOURCE_TYPES:
        recovered_any_ep = False
        for r in all_results:
            _, recovered = check_resources_recover(r["availability_snapshots"])
            if recovered[rtype]:
                recovered_any_ep = True
                break
        print(f"{rtype}: returns to full availability after being dispatched: {recovered_any_ep}")
        if not recovered_any_ep:
            issues.append(f"{rtype} never becomes available again after dispatch across any episode")

    total_dispatch_ok = sum(r["dispatch_ok"] for r in all_results)
    if total_dispatch_ok == 0:
        issues.append("No dispatch ever succeeded -- travel-time/routing or roster logic likely broken")

    print("Resource-arrival effects, aggregated across all episodes:")
    for rtype in RESOURCE_TYPES:
        ok = sum(r["effect_success_by_type"][rtype] for r in all_results)
        wasted = sum(r["effect_wasted_by_type"][rtype] for r in all_results)
        arrivals = ok + wasted
        rate = ok / arrivals if arrivals else float("nan")
        print(f"  {rtype}: {ok}/{arrivals} arrivals had an effect ({rate:.0%})")
        if arrivals >= 20 and ok == 0:
            issues.append(
                f"{rtype} never once had an effect across {arrivals} arrivals under a random policy -- "
                f"under EFFECT_RADIUS_CELLS={3}, this likely just means random zone-targeting almost never "
                f"lands the effect footprint on the (spatially small, localized) fire; expected to matter a "
                f"lot less once a trained policy can aim at the actual fire front instead of a random zone"
            )

    finite_travel_times = [t for t in env.zone_travel_time_s if math.isfinite(t)]
    mean_travel_min = (sum(finite_travel_times) / len(finite_travel_times)) / 60.0
    print(f"Mean zone travel time from depot: {mean_travel_min:.1f} min "
          f"(range {min(finite_travel_times) / 60:.1f}-{max(finite_travel_times) / 60:.1f} min) "
          f"-- plausible for a ~{env.width * env.meta['cell_size_m'] / 1000:.0f}km x "
          f"{env.height * env.meta['cell_size_m'] / 1000:.0f}km study area")

    print()
    if issues:
        print("=== FLAGGED ISSUES ===")
        for issue in issues:
            print(f"  ! {issue}")
    else:
        print("No anomalies flagged.")


if __name__ == "__main__":
    main()
