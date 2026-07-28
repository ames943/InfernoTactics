"""
Feature-by-feature toggle diff: isolates WHICH of v4's additions over the old
(pre-v4) training loop causes the entropy collapse the bisect run (see
logs/console_run_bisect_old_loop_v5env.log, models/checkpoints_bisect/)
proved does NOT happen in the old loop itself on this same frozen v5
environment. v4 bundled seven changes into one rewrite:
  E. per-episode advantage normalization (z-score over that episode's own
     advantages, vs. the old loop's cross-episode running RunningMeanStd
     applied to RETURNS, no separate advantage z-score at all)
  D. minibatching (MINIBATCH_SIZE=16-tick chunks, each its own
     zero_grad/backward/step, vs. the old loop's single accumulate-then-
     one-optimizer.step() per full episode) plus the PPO ratio/clip
     surrogate that minibatching necessitates (old_log_prob captured at
     rollout time, ratio=exp(new-old), clipped)
  A. GAE (bootstrapped advantage) replacing the full-episode discounted
     Monte-Carlo return
  C. per-minibatch KL-based early stopping
  B. multi-epoch updates (PPO_EPOCHS>1, replaying the same rollout's
     minibatches multiple times)
  G. entropy-floor + gradient-norm-spike combined early stop
  F. binary anchor-identity scalar appended to the MLP input
Each is gated by its own INFERNO_TOGGLE_<letter> env var (default "0" = old
loop's exact behavior for that piece). The experiment enables them
CUMULATIVELY in the order E, D, A, C, B, G, F -- C and B are only meaningful
once D's minibatch structure exists, so testing them in isolation from D
would not exercise the actual mechanism v4 used.

Every run MUST warm-start from models/checkpoints_bisect/episode_0260.pt --
the bisect's own reference checkpoint: healthy entropy, single_training
already solved (+42.4, 100% containment), ON THIS SAME v5 environment. This
matches how the actual v4/v5/v6 collapse experiments were run (warm-started
from an already-solved policy, then watching whether the new update
mechanism destroys it) -- starting from scratch would conflate "hasn't
learned yet" (bisect took to ep260) with "collapsed", since 100 episodes
isn't enough to re-solve from random init.

REQUIRED: INFERNO_RUN_TAG must be set and unique per run -- two earlier
checkpoint sets (v4_baseline, plain EPOCHS=1) were permanently lost to
unsuffixed-directory overwrites during the v4/v5 investigation; this script
refuses to run without a tag to prevent repeating that.

    INFERNO_TOGGLE_E=1 INFERNO_RUN_TAG=toggle_E python -m src.train.train_toggle_diff
"""

import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES,
    SCALAR_KEYS,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402

# --- Toggles -------------------------------------------------------------
TOGGLE_E = os.environ.get("INFERNO_TOGGLE_E", "0") == "1"
TOGGLE_D = os.environ.get("INFERNO_TOGGLE_D", "0") == "1"
TOGGLE_A = os.environ.get("INFERNO_TOGGLE_A", "0") == "1"
TOGGLE_C = os.environ.get("INFERNO_TOGGLE_C", "0") == "1"
TOGGLE_B = os.environ.get("INFERNO_TOGGLE_B", "0") == "1"
TOGGLE_G = os.environ.get("INFERNO_TOGGLE_G", "0") == "1"
TOGGLE_F = os.environ.get("INFERNO_TOGGLE_F", "0") == "1"

if TOGGLE_C and not TOGGLE_D:
    print("[toggle-diff] WARNING: TOGGLE_C without TOGGLE_D is a no-op (no minibatch loop to stop mid-way).")
if TOGGLE_B and not TOGGLE_D:
    print("[toggle-diff] WARNING: TOGGLE_B without TOGGLE_D is a no-op (multi-epoch requires the minibatch loop).")

RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "")
if not RUN_TAG:
    raise SystemExit("INFERNO_RUN_TAG is required (must be unique per run -- see module docstring).")

# --- Hyperparameters (values match v4 where a toggle borrows its mechanism) -
N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 100))
BASE_SEED = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
VALUE_LOSS_COEFF = 0.5
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.01
GRAD_CLIP_NORM = 0.5
PPO_CLIP_EPS = 0.2
MINIBATCH_SIZE = 16
PPO_EPOCHS = 4 if TOGGLE_B else 1
TARGET_KL = 0.02
KL_EARLY_STOP_THRESHOLD = 1.5 * TARGET_KL
ENTROPY_FLOOR = 5e-3
GRAD_NORM_SPIKE_WINDOW = 20
GRAD_NORM_SPIKE_MULTIPLIER = 5.0
SCALED_REWARD_CLIP = 10.0

STATUS_EVERY = 5
CHECKPOINT_EVERY = 10
EVAL_EVERY = 10
EVAL_EPISODES = 3
PROBE_TICKS = (0, 5, 10, 20, 40)

ANCHOR_FLAG_DIM = 1 if TOGGLE_F else 0
N_SCALARS = len(SCALAR_KEYS) + ANCHOR_FLAG_DIM

_TAG_SUFFIX = f"_{RUN_TAG}"
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_toggle{_TAG_SUFFIX}")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, f"train_log_toggle{_TAG_SUFFIX}.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, f"eval_log_toggle{_TAG_SUFFIX}.csv")
PROBE_LOG_PATH = os.path.join(LOG_DIR, f"probe_log_toggle{_TAG_SUFFIX}.jsonl")
WARM_START_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_bisect", "episode_0260.pt")

RESOURCE_ENTROPY_MAX = math.log(4)
ZONE_ENTROPY_MAX = math.log(32)

TRAIN_LOG_FIELDS = [
    "episode", "device", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "resource_entropy", "zone_entropy", "value_grad_norm", "other_grad_norm",
    "epochs_run", "minibatches_run", "early_stopped", "stop_reason", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "ignition_point_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate", "action_lock", "action_lock_fraction",
]
HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -19082.0, "avg_buildings_destroyed": 65.2, "containment_rate": 0.80},
}


def build_scalars(obs_scalars):
    if TOGGLE_F:
        return np.concatenate([flatten_scalars(obs_scalars), [1.0]]).astype(np.float32)
    return flatten_scalars(obs_scalars)


def build_model(n_grid_channels, device):
    model = InfernoModel(n_grid_channels=n_grid_channels, n_scalars=N_SCALARS).to(device)
    if not os.path.exists(WARM_START_CKPT):
        raise SystemExit(f"Warm-start checkpoint not found: {WARM_START_CKPT}")
    warm_sd = torch.load(WARM_START_CKPT, map_location=device, weights_only=False)
    if not TOGGLE_F:
        model.load_state_dict(warm_sd)
        print(f"[toggle-diff] Warm-started from {WARM_START_CKPT} (plain load, N_SCALARS={N_SCALARS}).")
        return model
    new_sd = model.state_dict()
    for key, value in warm_sd.items():
        if key == "mlp.net.0.weight":
            new_sd[key][:, :len(SCALAR_KEYS)] = value
            new_sd[key][:, len(SCALAR_KEYS):] = 0.0
        else:
            new_sd[key] = value
    model.load_state_dict(new_sd)
    print(f"[toggle-diff] Warm-started from {WARM_START_CKPT}, mlp.net.0.weight gained one "
          f"zero-initialized column for the anchor-identity flag (N_SCALARS={N_SCALARS}).")
    return model


class RunningMeanStd:
    """Welford's online mean/variance. normalize() (mean+std, for the A-off
    return-normalization path) and std() (scale only, for the A-on GAE
    reward-scaling path) are both provided; each toggle path uses only one."""

    def __init__(self, epsilon=1e-4, prior_std=1.0):
        self.mean = 0.0
        self.var = prior_std ** 2
        self.count = epsilon

    def update(self, values):
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total_count
        self.mean, self.var, self.count = new_mean, m2 / total_count, total_count

    def normalize(self, values):
        return [(v - self.mean) / (math.sqrt(self.var) + 1e-8) for v in values]

    def std(self):
        return math.sqrt(self.var) + 1e-8


def _is_mps_unimplemented_error(exc):
    msg = str(exc).lower()
    return "mps" in msg and "not implemented" in msg


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def collect_rollout(env, model, device, seed):
    obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, use_real_weather=True, seed=seed)
    steps = []
    total_reward = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    with torch.no_grad():
        while not done:
            grid_t, _ = InfernoModel.obs_to_tensors(obs, device=device)
            scalars_np = build_scalars(obs["scalars"])
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            action_logits, value, _classification_logits = model(grid_t, scalars_t)
            resource_dist = Categorical(logits=action_logits["resource_type"][0])
            zone_dist = Categorical(logits=action_logits["zone"][0])
            resource_idx = int(resource_dist.sample())
            zone_idx = int(zone_dist.sample())
            old_log_prob = float(
                resource_dist.log_prob(torch.tensor(resource_idx, device=device))
                + zone_dist.log_prob(torch.tensor(zone_idx, device=device))
            )
            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            steps.append({
                "grid": obs["grid"], "scalars": scalars_np,
                "resource_idx": resource_idx, "zone_idx": zone_idx,
                "fire_state_target": fire_state_target,
                "reward": reward, "old_log_prob": old_log_prob,
                "raw_value": float(value.squeeze().item()),
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    bootstrap_value_raw = 0.0
    if TOGGLE_A and info["timeout"]:
        grid_t, _ = InfernoModel.obs_to_tensors(obs, device=device)
        scalars_t = torch.from_numpy(build_scalars(obs["scalars"])).unsqueeze(0).to(device)
        with torch.no_grad():
            _, bv, _ = model(grid_t, scalars_t)
        bootstrap_value_raw = float(bv.squeeze().item())

    return steps, total_reward, buildings_destroyed, info["contained"], bootstrap_value_raw


def compute_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def compute_gae(rewards, values, bootstrap_value, gamma, lam):
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    next_value = bootstrap_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def compute_batch_kl_and_entropy(model, steps, device):
    kl_total = 0.0
    entropy_total = 0.0
    with torch.no_grad():
        for step in steps:
            grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
            action_logits, _value, _cls = model(grid_t, scalars_t)
            resource_dist = Categorical(logits=action_logits["resource_type"][0])
            zone_dist = Categorical(logits=action_logits["zone"][0])
            new_log_prob = float(
                resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
                + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
            )
            kl_total += step["old_log_prob"] - new_log_prob
            entropy_total += float(resource_dist.entropy() + zone_dist.entropy())
    n = len(steps)
    return kl_total / n, entropy_total / n


def single_batch_update(model, optimizer, steps, advantages, returns, device):
    """Old-loop style: one forward+backward per tick, gradients accumulate,
    ONE optimizer.step() after the whole episode. Used when TOGGLE_D is off."""
    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0
    resource_entropy_sum = zone_entropy_sum = 0.0
    n = len(steps)

    for step, adv, ret in zip(steps, advantages, returns):
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
        ret_t = torch.tensor(ret, dtype=value_scalar.dtype, device=device)
        adv_t = torch.tensor(adv, dtype=value_scalar.dtype, device=device)

        policy_loss = -log_prob * adv_t
        value_loss = (value_scalar - ret_t) ** 2
        classification_loss = F.cross_entropy(
            classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
        )
        tick_loss = (
            policy_loss + VALUE_LOSS_COEFF * value_loss
            + CLASSIFICATION_LOSS_COEFF * classification_loss - ENTROPY_COEFF * entropy
        )
        tick_loss.backward()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        classification_loss_sum += classification_loss.item()
        entropy_sum += entropy.item()
        resource_entropy_sum += resource_entropy.item()
        zone_entropy_sum += zone_entropy.item()

    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    return {
        "policy_loss": policy_loss_sum / n, "value_loss": value_loss_sum / n,
        "classification_loss": classification_loss_sum / n, "entropy": entropy_sum / n,
        "resource_entropy": resource_entropy_sum / n, "zone_entropy": zone_entropy_sum / n,
        "value_grad_norm": float(value_grad_norm), "other_grad_norm": float(other_grad_norm),
        "epochs_run": 1, "minibatches_run": 1, "early_stopped": False, "stop_reason": None,
    }


def minibatch_ppo_update(model, optimizer, steps, advantages, returns, device, seed, grad_norm_baseline):
    """v4-style: PPO_EPOCHS epochs of shuffled MINIBATCH_SIZE-tick minibatches,
    each its own zero_grad/backward/step. advantages/returns are fixed for the
    whole episode (computed once, outside this loop); only new_log_prob/value
    are recomputed fresh at every minibatch pass as the model changes.
    TOGGLE_C gates the per-minibatch KL check; TOGGLE_G gates the combined
    entropy-floor+grad-norm-spike check. Used when TOGGLE_D is on."""
    n = len(steps)
    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    resource_entropy_sum = zone_entropy_sum = 0.0
    with torch.no_grad():
        for step in steps:
            grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
            scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
            action_logits, _v, _c = model(grid_t, scalars_t)
            resource_entropy_sum += float(Categorical(logits=action_logits["resource_type"][0]).entropy())
            zone_entropy_sum += float(Categorical(logits=action_logits["zone"][0]).entropy())

    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0
    n_updates = 0
    value_grad_norm_last = other_grad_norm_last = 0.0
    epochs_run = 0
    minibatches_run = 0
    early_stopped = False
    stop_reason = None
    saw_grad_norm_spike = False

    shuffle_rng = random.Random(seed)
    indices = list(range(n))
    stop = False
    for epoch in range(PPO_EPOCHS):
        if stop:
            break
        shuffle_rng.shuffle(indices)
        for mb_start in range(0, n, MINIBATCH_SIZE):
            mb_idx = indices[mb_start:mb_start + MINIBATCH_SIZE]
            optimizer.zero_grad()
            for idx in mb_idx:
                step = steps[idx]
                grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
                scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
                action_logits, value, classification_logits = model(grid_t, scalars_t)

                resource_dist = Categorical(logits=action_logits["resource_type"][0])
                zone_dist = Categorical(logits=action_logits["zone"][0])
                new_log_prob = (
                    resource_dist.log_prob(torch.tensor(step["resource_idx"], device=device))
                    + zone_dist.log_prob(torch.tensor(step["zone_idx"], device=device))
                )
                entropy = resource_dist.entropy() + zone_dist.entropy()

                old_log_prob_t = torch.tensor(step["old_log_prob"], dtype=new_log_prob.dtype, device=device)
                ratio = torch.exp(new_log_prob - old_log_prob_t)
                adv = advantages[idx]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS, 1.0 + PPO_CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2)

                value_scalar = value.squeeze()
                return_target = torch.tensor(returns[idx], dtype=value_scalar.dtype, device=device)
                value_loss = (value_scalar - return_target) ** 2

                classification_loss = F.cross_entropy(
                    classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
                )
                tick_loss = (
                    policy_loss + VALUE_LOSS_COEFF * value_loss
                    + CLASSIFICATION_LOSS_COEFF * classification_loss - ENTROPY_COEFF * entropy
                )
                tick_loss.backward()

                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                classification_loss_sum += classification_loss.item()
                entropy_sum += entropy.item()
                n_updates += 1

            value_grad_norm_last = float(torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM))
            other_grad_norm_last = float(torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM))
            optimizer.step()
            minibatches_run += 1

            if TOGGLE_G and grad_norm_baseline is not None and other_grad_norm_last > GRAD_NORM_SPIKE_MULTIPLIER * grad_norm_baseline:
                saw_grad_norm_spike = True

            if TOGGLE_C or TOGGLE_G:
                kl_last, entropy_last = compute_batch_kl_and_entropy(model, steps, device)
                if TOGGLE_C and abs(kl_last) > KL_EARLY_STOP_THRESHOLD:
                    early_stopped = True
                    stop_reason = "kl"
                    stop = True
                    break
                if TOGGLE_G and entropy_last < ENTROPY_FLOOR and saw_grad_norm_spike:
                    early_stopped = True
                    stop_reason = "entropy_floor+grad_spike"
                    stop = True
                    break

        if not stop:
            epochs_run += 1

    return {
        "policy_loss": policy_loss_sum / n_updates, "value_loss": value_loss_sum / n_updates,
        "classification_loss": classification_loss_sum / n_updates, "entropy": entropy_sum / n_updates,
        "resource_entropy": resource_entropy_sum / n, "zone_entropy": zone_entropy_sum / n,
        "value_grad_norm": value_grad_norm_last, "other_grad_norm": other_grad_norm_last,
        "epochs_run": epochs_run, "minibatches_run": minibatches_run,
        "early_stopped": early_stopped, "stop_reason": stop_reason,
    }


def run_training_episode(env, model, optimizer, device, seed, return_normalizer, reward_scale_normalizer,
                          grad_norm_baseline):
    steps, total_reward, buildings_destroyed, contained, bootstrap_value_raw = collect_rollout(
        env, model, device, seed
    )
    raw_rewards = [s["reward"] for s in steps]
    raw_values = [s["raw_value"] for s in steps]

    if TOGGLE_A:
        scale = reward_scale_normalizer.std()
        scaled_rewards = [max(-SCALED_REWARD_CLIP, min(SCALED_REWARD_CLIP, r / scale)) for r in raw_rewards]
        advantages_raw, returns = compute_gae(scaled_rewards, raw_values, bootstrap_value_raw, GAMMA, GAE_LAMBDA)
        reward_scale_normalizer.update(raw_rewards)
    else:
        raw_returns = compute_returns(raw_rewards, GAMMA)
        returns = return_normalizer.normalize(raw_returns)
        advantages_raw = [ret - val for ret, val in zip(returns, raw_values)]
        return_normalizer.update(raw_returns)

    if TOGGLE_E:
        n = len(advantages_raw)
        adv_mean = sum(advantages_raw) / n
        adv_var = sum((a - adv_mean) ** 2 for a in advantages_raw) / n
        adv_std = math.sqrt(adv_var) + 1e-8
        advantages_final = [(a - adv_mean) / adv_std for a in advantages_raw]
    else:
        advantages_final = advantages_raw

    if TOGGLE_D:
        result = minibatch_ppo_update(model, optimizer, steps, advantages_final, returns, device, seed,
                                       grad_norm_baseline)
    else:
        result = single_batch_update(model, optimizer, steps, advantages_final, returns, device)

    result.update({
        "n_ticks": len(steps), "reward": total_reward,
        "buildings_destroyed": buildings_destroyed, "contained": contained,
    })
    return result


def probe_current_model(model, probe_states, device):
    per_state = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for tick, obs in probe_states:
            grid, _ = InfernoModel.obs_to_tensors(obs, device=device)
            scalars = torch.from_numpy(build_scalars(obs["scalars"])).unsqueeze(0).to(device)
            action_logits, _value, _cls = model(grid, scalars)
            resource_probs = torch.softmax(action_logits["resource_type"][0], dim=0)
            zone_probs = torch.softmax(action_logits["zone"][0], dim=0)
            r_idx = int(torch.argmax(resource_probs))
            z_idx = int(torch.argmax(zone_probs))
            per_state.append({
                "tick": tick, "resource_type": RESOURCE_TYPES[r_idx],
                "resource_max_prob": float(resource_probs[r_idx]),
                "zone": z_idx, "zone_max_prob": float(zone_probs[z_idx]),
            })
    if was_training:
        model.train()
    return per_state


def collect_probe_states(env):
    obs = env.reset(seed=0, ignition_point=TRAINING_IGNITION_POINT, use_real_weather=True)
    states = []
    if 0 in PROBE_TICKS:
        states.append((0, obs))
    t = 0
    done = False
    while not done and t < max(PROBE_TICKS):
        obs, _reward, done, _info = env.step(None)
        t += 1
        if t in PROBE_TICKS:
            states.append((t, obs))
    return states


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    result = eval_policy(model, env, ignition_point=TRAINING_IGNITION_POINT, n_episodes=EVAL_EPISODES,
                          use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                          scalars_fn=build_scalars, track_actions=True)
    mc = result["most_common_action"]
    eval_writer.writerow({
        "episode": episode_num, "ignition_point_name": "single_training",
        "avg_reward": result["avg_reward"], "avg_buildings_destroyed": result["avg_buildings_destroyed"],
        "avg_buildings_saved": result["avg_buildings_saved"], "containment_rate": result["containment_rate"],
        "action_lock": str(mc["action"]), "action_lock_fraction": mc["fraction_of_ticks"],
    })
    eval_file.flush()
    baseline = HEURISTIC_BASELINE["single_training"]
    delta = result["avg_reward"] - baseline["avg_reward"]
    print(f"    [eval @ ep {episode_num}] single_training: avg_reward={result['avg_reward']:.1f}  "
          f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
          f"containment_rate={result['containment_rate']:.0%}  "
          f"vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})", flush=True)
    print(f"        action_lock: {mc['action']} on {mc['fraction_of_ticks']:.1%} of ticks", flush=True)
    return result


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    device = get_device()
    print(f"[toggle-diff:{RUN_TAG}] {N_EPISODES} episodes, device={device}  "
          f"toggles: E={TOGGLE_E} D={TOGGLE_D} A={TOGGLE_A} C={TOGGLE_C} B={TOGGLE_B} G={TOGGLE_G} F={TOGGLE_F}")

    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)
    probe_states = collect_probe_states(env)

    model = build_model(probe_obs["grid"].shape[0], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()
    reward_scale_normalizer = RunningMeanStd()
    grad_norm_history = deque(maxlen=GRAD_NORM_SPIKE_WINDOW)

    episode_wall_times, episode_rewards = [], []
    completed_episodes = 0
    interrupted = False

    with open(TRAIN_LOG_PATH, "w", newline="") as train_file, \
         open(EVAL_LOG_PATH, "w", newline="") as eval_file, \
         open(PROBE_LOG_PATH, "w") as probe_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        train_writer.writeheader()
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
        eval_writer.writeheader()

        try:
            for ep in range(1, N_EPISODES + 1):
                ep_t0 = time.perf_counter()
                seed = BASE_SEED + ep
                grad_norm_baseline = statistics.median(grad_norm_history) if len(grad_norm_history) >= GRAD_NORM_SPIKE_WINDOW else None

                try:
                    result = run_training_episode(env, model, optimizer, device, seed, return_normalizer,
                                                   reward_scale_normalizer, grad_norm_baseline)
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    print(f"[toggle-diff] MPS op not implemented -- falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    result = run_training_episode(env, model, optimizer, device, seed, return_normalizer,
                                                   reward_scale_normalizer, grad_norm_baseline)

                grad_norm_history.append(result["other_grad_norm"])
                wall_s = time.perf_counter() - ep_t0
                episode_wall_times.append(wall_s)
                episode_rewards.append(result["reward"])
                completed_episodes = ep

                train_writer.writerow({
                    "episode": ep, "device": str(device), "n_ticks": result["n_ticks"],
                    "reward": result["reward"], "buildings_destroyed": result["buildings_destroyed"],
                    "contained": result["contained"], "policy_loss": result["policy_loss"],
                    "value_loss": result["value_loss"], "classification_loss": result["classification_loss"],
                    "entropy": result["entropy"], "resource_entropy": result["resource_entropy"],
                    "zone_entropy": result["zone_entropy"], "value_grad_norm": result["value_grad_norm"],
                    "other_grad_norm": result["other_grad_norm"], "epochs_run": result["epochs_run"],
                    "minibatches_run": result["minibatches_run"], "early_stopped": result["early_stopped"],
                    "stop_reason": result["stop_reason"], "wall_time_s": wall_s,
                })
                train_file.flush()

                if ep % STATUS_EVERY == 0:
                    recent = episode_rewards[-STATUS_EVERY:]
                    print(f"[status @ ep {ep:4d}/{N_EPISODES}] avg_reward(last{STATUS_EVERY})={sum(recent)/len(recent):10.1f}  "
                          f"entropy={result['entropy']:.3f}  "
                          f"resource_entropy={result['resource_entropy']:.3f}/{RESOURCE_ENTROPY_MAX:.3f}  "
                          f"zone_entropy={result['zone_entropy']:.3f}/{ZONE_ENTROPY_MAX:.3f}  "
                          f"early_stopped={result['early_stopped']} ({result['stop_reason']})  device={device}",
                          flush=True)

                if ep % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    torch.save(model.state_dict(), latest_ckpt_path)

                if ep % EVAL_EVERY == 0:
                    run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                    probe = probe_current_model(model, probe_states, device)
                    probe_file.write(json.dumps({"episode": ep, "probe": probe}) + "\n")
                    probe_file.flush()
                    locked_resources = {s["resource_type"] for s in probe}
                    locked_zones = {s["zone"] for s in probe}
                    print(f"        probe: resource={'/'.join(sorted(locked_resources))} "
                          f"(avg max_prob={sum(s['resource_max_prob'] for s in probe)/len(probe):.3f})  "
                          f"zone={'/'.join(str(z) for z in sorted(locked_zones))} "
                          f"(avg max_prob={sum(s['zone_max_prob'] for s in probe)/len(probe):.3f})", flush=True)

        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[toggle-diff] Caught Ctrl+C after episode {completed_episodes}/{N_EPISODES}.")
            torch.save(model.state_dict(), latest_ckpt_path)

        if completed_episodes > 0:
            torch.save(model.state_dict(), latest_ckpt_path)

    total_wall = sum(episode_wall_times)
    eps_per_min = completed_episodes / (total_wall / 60.0) if total_wall > 0 else float("nan")
    label = "INTERRUPTED" if interrupted else "COMPLETE"
    print(f"\n[toggle-diff:{RUN_TAG}] {label}: {completed_episodes}/{N_EPISODES} episodes, "
          f"{eps_per_min:.2f} episodes/min")


if __name__ == "__main__":
    main()
