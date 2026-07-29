"""
Final training experiment in this line: continuously randomized ignition
points, the standard fix for RL overfitting to a fixed scenario set (in the
original v4 plan, never run -- every prior run in this whole investigation,
including zonehead_fix1_2k, trained on a small FIXED set of 1-4 named
ignition points, which the policy could memorize rather than generalize
from). A new random ignition point is sampled every single episode from
InfernoEnv's own WUI-filtered candidate set (env._ignition_candidates,
99,657 real fuel cells within WUI_PROXIMITY_RADIUS_CELLS of a building --
verified via a quick read-only check before writing this script, matching
the ~99,657 figure from the earlier coverage-density exploration) --
recomputed fresh each InfernoEnv() construction from grid_static.npy, so
there is nothing to load/regenerate separately.

Held-out exclusion: mandeville_canyon and getty_view_park (the project's two
held-out validation points) are excluded from the sampling pool along with
every candidate within MIN_HOLDOUT_DISTANCE_CELLS (30 cells = ~900m, twice
WUI_PROXIMITY_RADIUS_CELLS) of either -- verified this excludes 5,173 of
99,657 candidates (~5.2%), leaving 94,484 real, still-diverse training
points. Without this, "held-out" would be nominal only -- the policy could
train on an ignition point 2-3 cells from mandeville_canyon's own and
functionally see the same scenario.

Everything else matches zonehead_fix1_2k's config exactly, copied (not
imported) from train_zonehead_fix.py per this project's established
convention of small, self-contained per-experiment scripts:
  - Fix 1: ZoneHead reads the per-zone zone_pooled features + the RAW MLP
    vector (models/actor_critic.py, models/inferno_model.py -- unchanged
    from zonehead_fix1_2k, no further architecture edits here).
  - Fix 2: auxiliary zone cross-entropy loss toward the real per-zone
    active-fire distribution, same AUX_ZONE_LOSS_COEFF=0.1.
  - OLD training loop: full-episode MC returns + running return normalizer,
    no GAE/minibatching/advantage z-scoring (collect_rollout/
    update_policy_with_zone_aux are byte-for-byte copies of
    train_zonehead_fix.py's).
  - Warm-started from the best zonehead_fix1_2k checkpoint (chosen via
    diagnostic_dominant_action_recheck.py's whole-rollout-dominant-action
    re-evaluation, NOT the live training log's weaker tick-0 metric) --
    exact architecture match expected (Fix 1 already baked into that
    checkpoint), so this warm start should be a full strict load with zero
    missing/unexpected keys, unlike zonehead_fix1_2k's own partial load from
    the pre-zonehead checkpoints_bisect checkpoint.

Eval cadence differs from checkpoint cadence -- checkpoints every 25
episodes (crash-safety/resumability), full eval every 50 episodes (this
message's explicit spec) on single_training (the fixed anchor, still
evaluated for continuity with every prior run) + the 2 held-out scenarios
(the actual test now: does randomized-ignition training make single_training's
learned behavior transfer to unseen locations, unlike the fixed-scenario runs
that never once closed this gap). stone_canyon is deliberately NOT
evaluated here -- it was one of the 4 fixed *training* points in the old
curriculum; under randomized training it's just another point the model may
or may not have sampled near, not a meaningful category anymore.

Zone-logit-vs-real-fire correlation probe: uses a FIXED set of 3 scenarios
(single_training + both held-out points) x 5 ticks = 15 probe observations,
same construction as zonehead_fix1_2k's own probe, computed identically at
every checkpoint (not re-randomized) so the trend is comparable over time.

Resumable + auto-relaunch: same resume_state.pt pattern as
train_zonehead_fix.py (model+optimizer+return_normalizer+completed_episodes,
saved every checkpoint). Meant to be run under the same kind of bash
supervisor loop used for zonehead_fix1_2k (relaunches automatically,
resuming from the latest checkpoint, if the process is killed).

    INFERNO_RUN_TAG=zonehead_randign_v1 INFERNO_N_EPISODES=2000 python -m src.train.train_zonehead_randign
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.fire_sim import BLAZE, THREAT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402
from train.train_actor_critic import (  # noqa: E402
    CLASSIFICATION_LOSS_COEFF,
    ENTROPY_COEFF,
    GAMMA,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    RESOURCE_ENTROPY_MAX,
    VALUE_LOSS_COEFF,
    ZONE_ENTROPY_MAX,
    RunningMeanStd,
    _is_mps_unimplemented_error,
    compute_returns,
    get_device,
)

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (distinct checkpoint dir per run).")
if RUN_TAG in ("v7", "zonehead_v1", "pool8x8", "zonehead_fix1", "zonehead_fix1_2k"):
    raise SystemExit(f"Refusing to run under RUN_TAG={RUN_TAG!r} -- that collides with an "
                      f"earlier experiment's paths. Pick a distinct tag.")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 2000))
BASE_SEED = 2000
AUX_ZONE_LOSS_COEFF = float(os.environ.get("INFERNO_AUX_ZONE_COEFF", 0.1))
MIN_HOLDOUT_DISTANCE_CELLS = 30.0  # ~900m at 30m/cell -- see module docstring

STATUS_EVERY = 20
CHECKPOINT_EVERY = int(os.environ.get("INFERNO_CHECKPOINT_EVERY", 25))
EVAL_EVERY = int(os.environ.get("INFERNO_EVAL_EVERY", 50))  # explicitly every 50, decoupled from checkpoint cadence
EVAL_EPISODES = 3

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_zonehead_{RUN_TAG}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_zonehead_{RUN_TAG}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_zonehead_{RUN_TAG}.csv")
PROBE_LOG_PATH = os.path.join(LOG_DIR, f"probe_log_zonehead_{RUN_TAG}.csv")
IGNITION_LOG_PATH = os.path.join(LOG_DIR, f"ignition_log_zonehead_{RUN_TAG}.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")
# Filled in at launch time from diagnostic_dominant_action_recheck.py's ranking
# (best zonehead_fix1_2k checkpoint by combined single_training+stone_canyon
# reward, preferring one with direct_test_pass=True) -- overridable via env
# var so this script doesn't need editing once that diagnostic's result is known.
WARM_START_CKPT = os.environ.get(
    "INFERNO_WARM_START_CKPT",
    os.path.join(PROJECT_ROOT, "models", "checkpoints_zonehead_zonehead_fix1_2k", "episode_2000.pt"),
)

EVAL_SCENARIOS = [("single_training", TRAINING_IGNITION_POINT)] + list(VALIDATION_IGNITION_POINTS.items())
HEURISTIC_BASELINE = {
    "single_training": -19082.0,
    "mandeville_canyon": 30.5,
    "getty_view_park": 46.1,
}
VALIDATION_NAMES = set(VALIDATION_IGNITION_POINTS.keys())

PROBE_SCENARIOS = EVAL_SCENARIOS  # same 3 points, reused for the correlation probe
PROBE_TICKS = [0, 15, 40, 75, 120]

TRAIN_LOG_FIELDS = [
    "episode", "device", "ignition_row", "ignition_col", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "aux_zone_loss", "entropy",
    "resource_entropy", "zone_entropy", "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario", "avg_reward", "avg_buildings_destroyed", "avg_buildings_saved",
    "containment_rate", "dominant_action", "dominant_action_fraction", "vs_heuristic_delta",
]
PROBE_LOG_FIELDS = ["episode", "zone_logit_corr_flat", "zone_logit_corr_per_obs_mean"]


def _zone_active_fire_counts(fire_channel, zones):
    counts = []
    for z in zones:
        r0, r1 = z["row_range"]
        c0, c1 = z["col_range"]
        sub = fire_channel[r0:r1, c0:c1]
        counts.append(int(np.count_nonzero((sub == THREAT) | (sub == BLAZE))))
    return counts


def _build_ignition_pool(env):
    """env._ignition_candidates (99,657 WUI-adjacent fuel cells, see module
    docstring) minus every candidate within MIN_HOLDOUT_DISTANCE_CELLS of
    either held-out validation point."""
    candidates = env._ignition_candidates.astype(np.float64)
    holdout = np.array(list(VALIDATION_IGNITION_POINTS.values()), dtype=np.float64)
    dists = np.sqrt(((candidates[:, None, :] - holdout[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    pool = env._ignition_candidates[dists >= MIN_HOLDOUT_DISTANCE_CELLS]
    print(f"[zonehead_randign:{RUN_TAG}] ignition pool: {len(pool)}/{len(env._ignition_candidates)} candidates "
          f"after excluding a {MIN_HOLDOUT_DISTANCE_CELLS:.0f}-cell radius around both held-out points.")
    return pool


def build_model_and_state(n_grid_channels, device):
    model = InfernoModel(n_grid_channels=n_grid_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()

    if os.path.exists(RESUME_STATE_PATH):
        state = torch.load(RESUME_STATE_PATH, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        return_normalizer.mean = state["return_normalizer_mean"]
        return_normalizer.var = state["return_normalizer_var"]
        return_normalizer.count = state["return_normalizer_count"]
        start_ep = state["completed_episodes"] + 1
        print(f"[zonehead_randign:{RUN_TAG}] RESUMED from {RESUME_STATE_PATH} at episode {start_ep}.")
        return model, optimizer, return_normalizer, start_ep

    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    old_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(old_sd, strict=False)
    if missing or unexpected:
        print(f"[zonehead_randign:{RUN_TAG}] WARNING: warm start was not an exact match -- "
              f"missing={missing}  unexpected={unexpected}")
    else:
        print(f"[zonehead_randign:{RUN_TAG}] Warm-started from {WARM_START_CKPT} -- exact match, "
              f"every key loaded unchanged.")
    return model, optimizer, return_normalizer, 1


def save_resume_state(model, optimizer, return_normalizer, completed_episodes):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "return_normalizer_mean": return_normalizer.mean, "return_normalizer_var": return_normalizer.var,
        "return_normalizer_count": return_normalizer.count, "completed_episodes": completed_episodes,
    }, RESUME_STATE_PATH)


def save_checkpoint(model, episode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"episode_{episode:04d}.pt"))
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "latest.pt"))


def collect_rollout(env, model, ignition_point, device, seed):
    """Byte-for-byte copy of train_zonehead_fix.py's collect_rollout (see
    that file's docstring) -- ignition_point is already a parameter there,
    so nothing about randomized sampling needs to change inside this
    function itself, only in main()'s per-episode call site."""
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps = []
    total_reward = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    with torch.no_grad():
        while not done:
            grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
            action_logits, _value, _classification_logits = model(grid_t, scalars_t)
            resource_idx = int(Categorical(logits=action_logits["resource_type"][0]).sample())
            zone_idx = int(Categorical(logits=action_logits["zone"][0]).sample())

            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())
            zone_fire_counts = _zone_active_fire_counts(obs["grid"][-1], env.zones)

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            steps.append({
                "grid": obs["grid"],
                "scalars": flatten_scalars(obs["scalars"]),
                "resource_idx": resource_idx,
                "zone_idx": zone_idx,
                "fire_state_target": fire_state_target,
                "zone_fire_counts": zone_fire_counts,
                "reward": reward,
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    return steps, total_reward, buildings_destroyed, info["contained"]


def update_policy_with_zone_aux(model, optimizer, steps, device, return_normalizer):
    """Byte-for-byte copy of train_zonehead_fix.py's update_policy_with_zone_aux."""
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = return_normalizer.normalize(raw_returns)
    return_normalizer.update(raw_returns)

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = aux_zone_loss_sum = entropy_sum = 0.0
    resource_entropy_sum = zone_entropy_sum = 0.0

    for step, g_t in zip(steps, returns):
        grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        action_logits, value, classification_logits = model(grid_t, scalars_t)

        resource_dist = Categorical(logits=action_logits["resource_type"][0])
        zone_dist = Categorical(logits=action_logits["zone"][0])
        log_prob = (
            resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
            + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
        )
        resource_entropy = resource_dist.entropy()
        zone_entropy = zone_dist.entropy()
        entropy = resource_entropy + zone_entropy

        value_scalar = value.squeeze()
        g_t_tensor = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (g_t_tensor - value_scalar).detach()

        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - g_t_tensor) ** 2
        classification_loss = F.cross_entropy(
            classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
        )

        zone_counts = torch.tensor(step["zone_fire_counts"], dtype=torch.float32, device=device)
        total = zone_counts.sum()
        zone_target = zone_counts / total if total > 0 else torch.full_like(zone_counts, 1.0 / len(zone_counts))
        aux_zone_loss = -(zone_target * F.log_softmax(action_logits["zone"][0], dim=0)).sum()

        tick_loss = (
            policy_loss
            + VALUE_LOSS_COEFF * value_loss
            + CLASSIFICATION_LOSS_COEFF * classification_loss
            + AUX_ZONE_LOSS_COEFF * aux_zone_loss
            - ENTROPY_COEFF * entropy
        )
        tick_loss.backward()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        classification_loss_sum += classification_loss.item()
        aux_zone_loss_sum += aux_zone_loss.item()
        entropy_sum += entropy.item()
        resource_entropy_sum += resource_entropy.item()
        zone_entropy_sum += zone_entropy.item()

    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return (
        policy_loss_sum / n, value_loss_sum / n, classification_loss_sum / n, aux_zone_loss_sum / n, entropy_sum / n,
        resource_entropy_sum / n, zone_entropy_sum / n,
        float(value_grad_norm), float(other_grad_norm),
    )


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    for name, point in EVAL_SCENARIOS:
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                              track_actions=True)
        mc = result["most_common_action"]
        baseline = HEURISTIC_BASELINE.get(name)
        delta = result["avg_reward"] - baseline if baseline is not None else None
        eval_writer.writerow({
            "episode": episode_num, "scenario": name, "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
            "dominant_action": str(mc["action"]), "dominant_action_fraction": mc["fraction_of_ticks"],
            "vs_heuristic_delta": delta,
        })
        held_out_tag = " [HELD-OUT]" if name in VALIDATION_NAMES else ""
        delta_str = f"  vs heuristic: {delta:+.1f} (heuristic: {baseline:.1f})" if delta is not None else ""
        print(f"    [eval @ ep {episode_num}] {name}{held_out_tag}: avg_reward={result['avg_reward']:.1f}  "
              f"destroyed={result['avg_buildings_destroyed']:.1f}  containment={result['containment_rate']:.0%}  "
              f"dominant={mc['action']} ({mc['fraction_of_ticks']:.1%}){delta_str}", flush=True)
    eval_file.flush()


def run_corr_probe(model, env, episode_num, device, probe_writer):
    """Same construction/purpose as train_zonehead_fix.py's
    run_direct_test_probe, minus the tick-0-argmax (a) metric -- that part
    of the old probe was already shown to be a weak test (identical argmax
    is expected at tick 0 regardless of real differentiation, since a
    handful of ignition cells barely register in a ~188,000-cell grid), so
    this run only tracks (b), the zone-logit-vs-real-fire correlation, over
    the 3-scenario x 5-tick fixed probe set."""
    was_training = model.training
    model.eval()
    zone_logits_all, active_counts_all = [], []
    with torch.no_grad():
        for name, point in PROBE_SCENARIOS:
            obs = env.reset(ignition_point=point, scenario="single", seed=BASE_SEED, use_real_weather=True)
            tick = 0
            done = False
            for target_tick in PROBE_TICKS:
                while tick < target_tick:
                    obs, _r, done, _info = env.step(None)
                    tick += 1
                    if done:
                        break
                grid_t, scalars_t = InfernoModel.obs_to_tensors(obs, device=device)
                action_logits, _value, _cls = model(grid_t, scalars_t)
                zone_logits_all.append(action_logits["zone"][0].cpu().numpy())
                active_counts_all.append(_zone_active_fire_counts(obs["grid"][-1], env.zones))
                if done:
                    break
    if was_training:
        model.train()

    zone_logits_all = np.stack(zone_logits_all)
    active_counts_all = np.array(active_counts_all, dtype=np.float64)
    flat_corr = float(np.corrcoef(zone_logits_all.ravel(), active_counts_all.ravel())[0, 1])
    per_obs_corrs = []
    for i in range(zone_logits_all.shape[0]):
        zl, ac = zone_logits_all[i], active_counts_all[i]
        if ac.std() > 0 and zl.std() > 0:
            per_obs_corrs.append(float(np.corrcoef(zl, ac)[0, 1]))
    per_obs_mean = float(np.mean(per_obs_corrs)) if per_obs_corrs else float("nan")

    probe_writer.writerow({"episode": episode_num, "zone_logit_corr_flat": flat_corr,
                            "zone_logit_corr_per_obs_mean": per_obs_mean})
    print(f"    [CORR PROBE @ ep {episode_num}] zone_logit-vs-real-fire corr: "
          f"flat={flat_corr:.4f}  per_obs_mean={per_obs_mean:.4f}", flush=True)


def main():
    device = get_device()
    print(f"[zonehead_randign:{RUN_TAG}] target={N_EPISODES} episodes, device={device}  "
          f"AUX_ZONE_LOSS_COEFF={AUX_ZONE_LOSS_COEFF}  warm_start={WARM_START_CKPT}  "
          f"eval_every={EVAL_EVERY}  checkpoint_every={CHECKPOINT_EVERY}")

    env = InfernoEnv(seed=BASE_SEED)
    env.reset(seed=BASE_SEED)
    n_grid_channels = env.reset(seed=BASE_SEED)["grid"].shape[0]
    ignition_pool = _build_ignition_pool(env)
    model, optimizer, return_normalizer, start_ep = build_model_and_state(n_grid_channels, device)
    ignition_rng = np.random.default_rng(BASE_SEED + 999 + start_ep)

    resuming = start_ep > 1 and os.path.exists(TRAIN_LOG_PATH)
    write_mode = "a" if resuming else "w"
    train_file = open(TRAIN_LOG_PATH, write_mode, newline="")
    eval_file = open(EVAL_LOG_PATH, write_mode, newline="")
    probe_file = open(PROBE_LOG_PATH, write_mode, newline="")
    ignition_file = open(IGNITION_LOG_PATH, write_mode, newline="")
    train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
    eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
    probe_writer = csv.DictWriter(probe_file, fieldnames=PROBE_LOG_FIELDS)
    ignition_writer = csv.DictWriter(ignition_file, fieldnames=["episode", "row", "col"])
    if write_mode == "w":
        train_writer.writeheader()
        eval_writer.writeheader()
        probe_writer.writeheader()
        ignition_writer.writeheader()

    recent_rewards = []
    stopped_reason = None
    ep = start_ep - 1

    try:
        for ep in range(start_ep, N_EPISODES + 1):
            ep_t0 = time.perf_counter()
            seed = BASE_SEED + ep
            idx = ignition_rng.integers(len(ignition_pool))
            row, col = int(ignition_pool[idx][0]), int(ignition_pool[idx][1])
            ignition_point = (row, col)
            ignition_writer.writerow({"episode": ep, "row": row, "col": col})

            try:
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, aux_zone_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy_with_zone_aux(
                    model, optimizer, steps, device, return_normalizer
                )
            except Exception as e:
                if not _is_mps_unimplemented_error(e):
                    raise
                print(f"[zonehead_randign] MPS op not implemented -- falling back to CPU.")
                device = torch.device("cpu")
                model = model.to(device)
                steps, total_reward, buildings_destroyed, contained = collect_rollout(
                    env, model, ignition_point, device, seed
                )
                (policy_loss, value_loss, classification_loss, aux_zone_loss, entropy,
                 resource_entropy, zone_entropy, value_grad_norm, other_grad_norm) = update_policy_with_zone_aux(
                    model, optimizer, steps, device, return_normalizer
                )

            wall_s = time.perf_counter() - ep_t0
            recent_rewards.append(total_reward)
            if len(recent_rewards) > STATUS_EVERY:
                recent_rewards.pop(0)

            train_writer.writerow({
                "episode": ep, "device": str(device), "ignition_row": row, "ignition_col": col,
                "n_ticks": len(steps), "reward": total_reward, "buildings_destroyed": buildings_destroyed,
                "contained": contained, "policy_loss": policy_loss, "value_loss": value_loss,
                "classification_loss": classification_loss, "aux_zone_loss": aux_zone_loss, "entropy": entropy,
                "resource_entropy": resource_entropy, "zone_entropy": zone_entropy,
                "value_grad_norm": value_grad_norm, "other_grad_norm": other_grad_norm, "wall_time_s": wall_s,
            })
            train_file.flush()
            ignition_file.flush()

            if ep % STATUS_EVERY == 0:
                print(f"[status @ ep {ep:5d}/{N_EPISODES}] ignition=({row},{col})  "
                      f"avg_reward(last{len(recent_rewards)})={sum(recent_rewards)/len(recent_rewards):10.1f}  "
                      f"aux_zone_loss={aux_zone_loss:.4f}  "
                      f"resource_entropy={resource_entropy:.3f}/{RESOURCE_ENTROPY_MAX:.3f}  "
                      f"zone_entropy={zone_entropy:.3f}/{ZONE_ENTROPY_MAX:.3f}  device={device}", flush=True)

            if ep % CHECKPOINT_EVERY == 0:
                save_checkpoint(model, ep)
                save_resume_state(model, optimizer, return_normalizer, ep)

            if ep % EVAL_EVERY == 0:
                print(f"\n[zonehead_randign:{RUN_TAG}] ===== eval @ ep {ep} =====", flush=True)
                run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                run_corr_probe(model, env, ep, device, probe_writer)
                probe_file.flush()
                print(f"[zonehead_randign:{RUN_TAG}] ===== end eval =====\n", flush=True)

        stopped_reason = f"Reached target N_EPISODES={N_EPISODES}."

    finally:
        save_checkpoint(model, ep)
        save_resume_state(model, optimizer, return_normalizer, ep)
        train_file.close()
        eval_file.close()
        probe_file.close()
        ignition_file.close()

    print(f"\n[zonehead_randign:{RUN_TAG}] STOPPED at episode {ep}. Reason: {stopped_reason}")


if __name__ == "__main__":
    main()
