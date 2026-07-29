"""
Tests two changes together against a single, precise question: across every
checkpoint of the v7 multi-scenario run (old loop, no GAE/minibatch/z-score),
single_training and stone_canyon ALWAYS locked onto the SAME zone number
(19/19, 28/28, never two different zones) even though they need different
ones (single_training's known-good solution is (helicopter, 18); stone_canyon
contained at 67% under (helicopter, 19) or (helicopter, 28)). That is
consistent with one shared, scenario-blind zone preference rather than real
per-zone spatial discrimination -- and the old zone_head had no per-zone
spatial input to discriminate on: it was a plain Linear(hidden_dim, n_zones)
reading only the CNN's single 128-dim globally-pooled vector, which already
averaged the 32 zones' spatial differences away before the zone head ever
saw them.

Change 1 (models/cnn_branch.py, models/actor_critic.py, models/inferno_model.py):
the zone head now also receives, per zone, that zone's own average-pooled
slice of the full-resolution 32-channel feature map (already computed for
the classification head, zero new conv params) concatenated with the shared
trunk context -- see actor_critic.ZoneHead. resource_type_head and value_head
are UNCHANGED (still plain Linear(hidden_dim, ...) reading only the shared
context). New total param count vs the previous ~240K: verified via
models/test_model_forward.py.

Change 2 (this file): potential-based reward shaping for zone choice.
Potential function, using env.resources/env.zones read-only (no changes to
inferno_env.py itself):

    Phi(s) = -SHAPING_COEFF * sum over units with state=="traveling" of
             (active Threat+Blaze cell count in that unit's target_zone)

A unit is "traveling" only BEFORE its effect is applied (see
InfernoEnv._advance_resources(): state flips traveling->deployed the SAME
tick the effect lands, target_zone stays set through the deployed/busy
period too -- so filtering on state=="traveling" specifically captures
"currently en route, effect not yet applied", not "recently arrived and
still busy"). Shaped reward per tick: r_shaped = r_raw + GAMMA*Phi(s')-Phi(s).
This is standard potential-based shaping (Ng et al. 1999): a telescoping sum
over any complete episode, so it cannot change which policy is optimal,
regardless of SHAPING_COEFF's value. What it changes is CREDIT ASSIGNMENT
speed -- a dispatch toward a zone that still has real fire in it now gets a
reward at/around the tick it arrives, rather than only via the episode's
terminal outcome. SHAPING_COEFF default 0.02 (matches the fire-area shaping
term's coefficient already used in train_actor_critic_multi_v4.py/_v6.py, a
precedented small scale relative to terminal rewards like
FIRE_EXTINGUISHED_REWARD=+50 or the population-weighted building-destruction
penalty).

Reuses train_actor_critic.py's OWN compute_returns/RunningMeanStd/update_policy/
get_device via direct import (unmodified) -- same old-loop mechanics the
bisect validated, no GAE/minibatching/advantage-z-scoring. Only collect_rollout
is a new variant (collect_rollout_with_shaping) since it needs to read
env.resources/env.zones between steps to compute the potential.

Warm-started from models/checkpoints_bisect/episode_0260.pt with a PARTIAL
load: every key whose name+shape matches the new architecture loads
unchanged (cnn, mlp, classifier, actor_critic.trunk/resource_type_head/
value_head); actor_critic.zone_head.* does NOT match (entirely new submodule,
different key names) and is left at its fresh random init.

Direct test tracked every eval: is there ANY checkpoint where single_training
solves at (helicopter, 18) while stone_canyon SIMULTANEOUSLY contains at a
DIFFERENT zone? That has never happened across the whole v7 investigation.

    INFERNO_RUN_TAG=zonehead_v1 INFERNO_N_EPISODES=400 python -m src.train.train_zonehead_shaping
"""

import csv
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_TYPES,
    TRAINING_IGNITION_POINT,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402
from train.train_actor_critic import (  # noqa: E402
    GAMMA,
    LEARNING_RATE,
    RESOURCE_ENTROPY_MAX,
    ZONE_ENTROPY_MAX,
    RunningMeanStd,
    _is_mps_unimplemented_error,
    get_device,
    update_policy,
)
from torch.distributions import Categorical  # noqa: E402

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (distinct checkpoint dir per run).")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 400))
BASE_SEED = 2000
SHAPING_COEFF = float(os.environ.get("INFERNO_SHAPING_COEFF", 0.02))
STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
EVAL_EVERY = 20
EVAL_EPISODES = 3
KNOWN_GOOD_ZONE = {"single_training": 18}  # the pre-existing, known-good solution's zone

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_zonehead_{RUN_TAG}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_zonehead_{RUN_TAG}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_zonehead_{RUN_TAG}.csv")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_bisect", "episode_0260.pt")

TRAINING_SCENARIOS = [
    ("single_training", TRAINING_IGNITION_POINT),
    ("stone_canyon", MULTI_IGNITION_TRAINING_SCENARIO[2]),
]

HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -19082.0},
    "stone_canyon": {"avg_reward": -45379.9},
}

TRAIN_LOG_FIELDS = [
    "episode", "device", "scenario", "n_ticks", "raw_reward", "shaped_reward_total",
    "buildings_destroyed", "contained", "policy_loss", "value_loss", "classification_loss",
    "entropy", "resource_entropy", "zone_entropy", "shaping_mean_abs",
    "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario", "avg_reward", "avg_buildings_destroyed", "avg_buildings_saved",
    "containment_rate", "action_lock", "action_lock_fraction",
]


def build_model(n_grid_channels, device):
    model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    warm_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    new_sd = model.state_dict()
    loaded, skipped = [], []
    for key, value in warm_sd.items():
        if key in new_sd and new_sd[key].shape == value.shape:
            new_sd[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(new_sd)
    print(f"[zonehead:{RUN_TAG}] Warm-started from {WARM_START_CKPT}: "
          f"{len(loaded)} keys loaded unchanged, {len(skipped)} skipped (new zone_head architecture): {skipped}")
    return model


def _zone_active_fire_counts(fire_channel, zones):
    counts = []
    for z in zones:
        r0, r1 = z["row_range"]
        c0, c1 = z["col_range"]
        sub = fire_channel[r0:r1, c0:c1]
        counts.append(int(np.count_nonzero((sub == 2) | (sub == 3))))
    return counts


def compute_potential(env, fire_channel):
    zone_fire = _zone_active_fire_counts(fire_channel, env.zones)
    total = 0
    for rtype in RESOURCE_TYPES:
        for unit in env.resources[rtype]:
            if unit["state"] == "traveling":
                total += zone_fire[unit["target_zone"]]
    return -SHAPING_COEFF * total


def collect_rollout_with_shaping(env, model, ignition_point, device, seed):
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps = []
    total_raw_reward = 0.0
    total_shaped_reward = 0.0
    shaping_abs_sum = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    phi_prev = 0.0  # all resources available/idle at episode start -> Phi(s_0) = 0 exactly

    with torch.no_grad():
        while not done:
            grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
            action_logits, _value, _classification_logits = model(grid_t, scalars_t)
            resource_idx = int(Categorical(logits=action_logits["resource_type"][0]).sample())
            zone_idx = int(Categorical(logits=action_logits["zone"][0]).sample())

            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())
            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, raw_reward, done, info = env.step(action)

            phi_next = compute_potential(env, next_obs["grid"][-1])
            shaping_delta = GAMMA * phi_next - phi_prev
            shaped_reward = raw_reward + shaping_delta
            phi_prev = phi_next

            steps.append({
                "grid": obs["grid"], "scalars": flatten_scalars(obs["scalars"]),
                "resource_idx": resource_idx, "zone_idx": zone_idx,
                "fire_state_target": fire_state_target, "reward": shaped_reward,
            })
            total_raw_reward += raw_reward
            total_shaped_reward += shaped_reward
            shaping_abs_sum += abs(shaping_delta)
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    shaping_mean_abs = shaping_abs_sum / len(steps)
    return steps, total_raw_reward, total_shaped_reward, shaping_mean_abs, buildings_destroyed, info["contained"]


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    results = {}
    for name, point in TRAINING_SCENARIOS:
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                              track_actions=True)
        results[name] = result
        mc = result["most_common_action"]
        eval_writer.writerow({
            "episode": episode_num, "scenario": name, "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
            "action_lock": str(mc["action"]), "action_lock_fraction": mc["fraction_of_ticks"],
        })
        baseline = HEURISTIC_BASELINE[name]
        delta = result["avg_reward"] - baseline["avg_reward"]
        print(f"    [eval @ ep {episode_num}] {name}: avg_reward={result['avg_reward']:.1f}  "
              f"destroyed={result['avg_buildings_destroyed']:.1f}  containment={result['containment_rate']:.0%}  "
              f"action_lock={mc['action']} ({mc['fraction_of_ticks']:.1%})  "
              f"vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})", flush=True)
    eval_file.flush()

    single_zone = results["single_training"]["most_common_action"]["action"][1]
    stone_zone = results["stone_canyon"]["most_common_action"]["action"][1]
    single_solved = results["single_training"]["avg_reward"] > 0
    stone_contained = results["stone_canyon"]["containment_rate"] >= 0.5
    different_zones = single_solved and stone_contained and (single_zone != stone_zone)
    print(f"    [DIRECT TEST @ ep {episode_num}] single_training solved={single_solved} (zone {single_zone})  "
          f"stone_canyon contained={stone_contained} (zone {stone_zone})  "
          f"DIFFERENT_ZONES={different_zones}", flush=True)
    return results, different_zones


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    device = get_device()
    print(f"[zonehead:{RUN_TAG}] {N_EPISODES} episodes, device={device}, SHAPING_COEFF={SHAPING_COEFF}  "
          f"training scenarios={[n for n, _ in TRAINING_SCENARIOS]}")

    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)
    model = build_model(probe_obs["grid"].shape[0], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()
    scenario_rng = random.Random(BASE_SEED + 999)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[zonehead:{RUN_TAG}] Total parameters: {n_params:,} (previous architecture: ~240,858)")

    ever_different_zones = False
    first_different_zones_ep = None

    with open(TRAIN_LOG_PATH, "w", newline="") as train_file, \
         open(EVAL_LOG_PATH, "w", newline="") as eval_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        train_writer.writeheader()
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
        eval_writer.writeheader()

        for ep in range(1, N_EPISODES + 1):
            ep_t0 = time.perf_counter()
            seed = BASE_SEED + ep
            scenario_name, ignition_point = scenario_rng.choice(TRAINING_SCENARIOS)

            try:
                (steps, raw_reward, shaped_reward_total, shaping_mean_abs,
                 buildings_destroyed, contained) = collect_rollout_with_shaping(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy(
                    model, optimizer, steps, device, return_normalizer
                )
            except Exception as e:
                if not _is_mps_unimplemented_error(e):
                    raise
                print(f"[zonehead] MPS op not implemented -- falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
                (steps, raw_reward, shaped_reward_total, shaping_mean_abs,
                 buildings_destroyed, contained) = collect_rollout_with_shaping(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy(
                    model, optimizer, steps, device, return_normalizer
                )

            wall_s = time.perf_counter() - ep_t0
            train_writer.writerow({
                "episode": ep, "device": str(device), "scenario": scenario_name, "n_ticks": len(steps),
                "raw_reward": raw_reward, "shaped_reward_total": shaped_reward_total,
                "buildings_destroyed": buildings_destroyed, "contained": contained,
                "policy_loss": policy_loss, "value_loss": value_loss,
                "classification_loss": classification_loss, "entropy": entropy,
                "resource_entropy": resource_entropy, "zone_entropy": zone_entropy,
                "shaping_mean_abs": shaping_mean_abs, "value_grad_norm": value_grad_norm,
                "other_grad_norm": other_grad_norm, "wall_time_s": wall_s,
            })
            train_file.flush()

            if ep % STATUS_EVERY == 0:
                print(f"[status @ ep {ep:4d}/{N_EPISODES}] scenario={scenario_name:16s} "
                      f"raw_reward={raw_reward:10.1f}  shaping_mean_abs={shaping_mean_abs:.4f}  "
                      f"resource_entropy={resource_entropy:.3f}/{RESOURCE_ENTROPY_MAX:.3f}  "
                      f"zone_entropy={zone_entropy:.3f}/{ZONE_ENTROPY_MAX:.3f}  "
                      f"value_loss={value_loss:.3f}  device={device}", flush=True)

            if ep % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt"))
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))

            if ep % EVAL_EVERY == 0:
                _results, different_zones = run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                if different_zones and not ever_different_zones:
                    ever_different_zones = True
                    first_different_zones_ep = ep

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))
    print(f"\n[zonehead:{RUN_TAG}] COMPLETE: {N_EPISODES} episodes.")
    print(f"[zonehead:{RUN_TAG}] DIRECT TEST RESULT: different-zones-simultaneously "
          f"{'OCCURRED at ep ' + str(first_different_zones_ep) if ever_different_zones else 'NEVER occurred'}.")


if __name__ == "__main__":
    main()
