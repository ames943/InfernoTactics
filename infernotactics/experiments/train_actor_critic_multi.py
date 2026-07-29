"""
Multi-ignition curriculum training. Same algorithm, model, reward function,
and stability fixes as src/train/train_actor_critic.py (plain single-episode
actor-critic with return normalization, entropy bonus, split gradient
clipping, MPS-with-CPU-fallback) -- see that file's module docstring for the
full rationale behind those choices, none of which changed here. This file
exists as a SEPARATE script (rather than editing the working single-scenario
one) specifically to fix the generalization gap that single-scenario run
exposed: trained for 2000 episodes on one fixed ignition point (Skull Rock),
it beat the heuristic baseline decisively there (+29,443.0 reward) but
performed far worse than the heuristic on both held-out validation points
(mandeville_canyon, getty_view_park) -- overfitting/memorization to one
location's fixed sequence, not a broken training mechanism (losses/entropy/
gradients stayed healthy the whole prior run).

What's different from train_actor_critic.py:
  1. Each episode ignites ONE point sampled uniformly at random (not a fixed
     rotation) from env.inferno_env.MULTI_IGNITION_TRAINING_SCENARIO (Topanga
     ridge/Trippet Ranch, Sullivan Canyon/Brentwood, Stone Canyon/Bel-Air --
     three real WUI hillsides across the same Santa Ana corridor). This is
     three single-fire episodes of varying start location, NOT one
     simultaneous three-front episode (InfernoEnv also supports the latter
     via ignition_points=[...]/scenario='multi', which is a different, harder
     scenario -- deliberately not used here, since the goal is a policy that
     generalizes to WHERE a fire starts, not a policy for fighting three fires
     at once). VALIDATION_IGNITION_POINTS (mandeville_canyon, getty_view_park)
     are never sampled for training -- imported only for eval.
  2. Eval suite (run_eval_suite) reports all 5 scenarios individually every
     time -- the 3 training ignition points AND both validation points, each
     with its own heuristic-baseline delta -- instead of the prior script's
     3 (single_training + 2 validation). The point is to watch validation
     performance specifically, not just aggregate/training performance,
     since that's the number that was broken last time.
  3. best.pt now means something: saved whenever a new eval's reward,
     AVERAGED ACROSS ALL 5 SCENARIOS, beats the previous best -- not just the
     training-point reward. This directly targets the "checkpoint that's
     great on Skull Rock but terrible everywhere else" failure mode a manual,
     after-the-fact pick (best.pt = ep 260 latest.pt) worked around last time;
     here it's the automatic checkpointing criterion.
  4. Separate log/checkpoint paths (models/checkpoints_multi/,
     logs/train_log_multi.csv, logs/eval_log_multi.csv) so this run does not
     overwrite the completed single-scenario run's artifacts.
  5. Checkpoint/resume support (added after two real runs were both killed by
     something external -- no crash, no NaN, no graceful Ctrl+C message logged
     either time -- after ~30-40 wall-clock minutes each, losing all progress
     since train_actor_critic.py's original single-scenario script had no
     resume path). resume_state.pt (saved alongside every latest.pt) holds
     model + optimizer state, return_normalizer's mean/var/count, the
     ignition-sampling RNG's state, and all the running counters/lists needed
     to make the final summary accurate across a resume -- NOT just model
     weights, so a resumed run continues learning (correct Adam momentum,
     correct return-normalization stats, correct RNG stream) rather than
     restarting the optimizer cold on top of a trained model. A SIGTERM
     handler re-raises as KeyboardInterrupt so an external kill -- not just
     Ctrl+C -- hits the same graceful save-and-exit path (Python's default
     SIGTERM handling does not do this, which is presumably why the first two
     runs left no interrupt message despite dying cleanly enough to leave
     valid checkpoints).

Heuristic baselines for the 3 multi-ignition points below were measured the
same way the single-scenario script's HEURISTIC_BASELINE values were (5
episodes, seed=0, real weather, deterministic HeuristicPolicy) but per
INDIVIDUAL point (ignition_point=<one point>), since MULTI_IGNITION_TRAINING_
SCENARIO's own baseline table entry evaluates all 3 simultaneously (a
different, harder scenario -- see point 1 above) and isn't the right
comparison for a policy trained on one-fire-at-a-time episodes.

    python -m src.train.train_actor_critic_multi                 # real 2000-episode run
    INFERNO_N_EPISODES=25 python -m src.train.train_actor_critic_multi   # smoke test
"""

import csv
import math
import os
import random
import re
import signal
import sys
import time
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from env.inferno_env import (  # noqa: E402
    MULTI_IGNITION_TRAINING_SCENARIO,
    RESOURCE_TYPES,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402

# Order must match MULTI_IGNITION_TRAINING_SCENARIO's tuple order (see that
# list's docstring in inferno_env.py: topanga_ridge, sullivan_canyon, stone_canyon).
MULTI_IGNITION_NAMES = ["topanga_ridge", "sullivan_canyon", "stone_canyon"]

# --- Hyperparameters -- IDENTICAL to train_actor_critic.py, not retuned here -----
# 3500 (not 2000): with 3 training ignition points sampled uniformly at random
# instead of one fixed point, 2000 episodes would give each point only ~667
# repetitions vs. the single-scenario run's 2000 -- bumped up (user's call,
# given a flagged 2000-vs-more tradeoff) to keep per-point repetition closer
# to that prior run's scale without ~3x'ing total wall time.
N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 3500))
BASE_SEED = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.99
VALUE_LOSS_COEFF = 0.5
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.01
GRAD_CLIP_NORM = 0.5

STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
EVAL_EVERY = 20
EVAL_EPISODES = 3

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints_multi")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log_multi.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_log_multi.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")

TRAIN_LOG_FIELDS = [
    "episode", "ignition_name", "device", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "ignition_point_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate",
]

# Per-individual-point heuristic baselines (5 episodes, seed=0, real weather,
# deterministic HeuristicPolicy) -- see module docstring for why these are
# measured per-point rather than reusing the combined "multi_ignition"
# baseline table entry.
HEURISTIC_BASELINE = {
    "topanga_ridge": {"avg_reward": -5030.5, "avg_buildings_destroyed": 18.6, "containment_rate": 0.80},
    "sullivan_canyon": {"avg_reward": -5519.5, "avg_buildings_destroyed": 22.0, "containment_rate": 0.80},
    "stone_canyon": {"avg_reward": -300.3, "avg_buildings_destroyed": 1.0, "containment_rate": 1.00},
    "mandeville_canyon": {"avg_reward": 30.5, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 46.1, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
}


class RunningMeanStd:
    """Welford's online mean/variance for return normalization -- unchanged
    from train_actor_critic.py, see that file's docstring."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, values):
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, values):
        return [(v - self.mean) / (math.sqrt(self.var) + 1e-8) for v in values]


def _is_mps_unimplemented_error(exc):
    msg = str(exc).lower()
    return "mps" in msg and "not implemented" in msg


def _extract_op_name(exc):
    msg = str(exc)
    match = re.search(r"operator '([^']+)'", msg)
    if match:
        return match.group(1)
    match = re.search(r"^([A-Za-z0-9_ ]+?MPS)[:,.]", msg)
    if match:
        return match.group(1).strip()
    return msg.splitlines()[0]


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def save_resume_state(path, model, optimizer, return_normalizer, ignition_rng, completed_episodes,
                       best_avg_eval_reward, best_episode, ignition_counts, episode_rewards,
                       episode_losses, episode_wall_times):
    """Everything needed to continue training as if it had never stopped --
    not just model weights (see module docstring point 5). Written to a
    TEMP path then renamed over the real one so a kill mid-save (the exact
    failure mode this exists to survive) can't leave a half-written,
    unloadable resume file."""
    tmp_path = path + ".tmp"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "return_normalizer": {
            "mean": return_normalizer.mean, "var": return_normalizer.var, "count": return_normalizer.count,
        },
        "ignition_rng_state": ignition_rng.getstate(),
        "completed_episodes": completed_episodes,
        "best_avg_eval_reward": best_avg_eval_reward,
        "best_episode": best_episode,
        "ignition_counts": dict(ignition_counts),
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "episode_wall_times": episode_wall_times,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_resume_state(path, model, optimizer, return_normalizer, ignition_rng):
    """Inverse of save_resume_state -- mutates model/optimizer/return_normalizer/
    ignition_rng in place (matching torch's load_state_dict convention) and
    returns the plain-value fields the caller needs to seed its own loop
    state with."""
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return_normalizer.mean = state["return_normalizer"]["mean"]
    return_normalizer.var = state["return_normalizer"]["var"]
    return_normalizer.count = state["return_normalizer"]["count"]
    ignition_rng.setstate(state["ignition_rng_state"])
    return {
        "completed_episodes": state["completed_episodes"],
        "best_avg_eval_reward": state["best_avg_eval_reward"],
        "best_episode": state["best_episode"],
        "ignition_counts": Counter(state["ignition_counts"]),
        "episode_rewards": state["episode_rewards"],
        "episode_losses": state["episode_losses"],
        "episode_wall_times": state["episode_wall_times"],
    }


def collect_rollout(env, model, ignition_point, device, seed):
    """Identical to train_actor_critic.py's collect_rollout -- one full
    episode, current policy, no gradients -- just parameterized on whichever
    single ignition_point the caller sampled for this episode."""
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

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            steps.append({
                "grid": obs["grid"],
                "scalars": flatten_scalars(obs["scalars"]),
                "resource_idx": resource_idx,
                "zone_idx": zone_idx,
                "fire_state_target": fire_state_target,
                "reward": reward,
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs

    return steps, total_reward, buildings_destroyed, info["contained"]


def compute_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def update_policy(model, optimizer, steps, device, return_normalizer):
    """Identical to train_actor_critic.py's update_policy -- see that file's
    docstring for the full rationale (return normalization, per-tick
    backward accumulation, split gradient clipping)."""
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = return_normalizer.normalize(raw_returns)
    return_normalizer.update(raw_returns)

    value_head_params = list(model.actor_critic.value_head.parameters())
    value_head_param_ids = {id(p) for p in value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0

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
        entropy = resource_dist.entropy() + zone_dist.entropy()

        value_scalar = value.squeeze()
        g_t_tensor = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (g_t_tensor - value_scalar).detach()

        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - g_t_tensor) ** 2
        classification_loss = F.cross_entropy(
            classification_logits, step["fire_state_target"].unsqueeze(0).to(device)
        )

        tick_loss = (
            policy_loss
            + VALUE_LOSS_COEFF * value_loss
            + CLASSIFICATION_LOSS_COEFF * classification_loss
            - ENTROPY_COEFF * entropy
        )
        tick_loss.backward()

        policy_loss_sum += policy_loss.item()
        value_loss_sum += value_loss.item()
        classification_loss_sum += classification_loss.item()
        entropy_sum += entropy.item()

    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return (
        policy_loss_sum / n, value_loss_sum / n, classification_loss_sum / n, entropy_sum / n,
        float(value_grad_norm), float(other_grad_norm),
    )


def run_training_episode(env, model, optimizer, device, seed, return_normalizer, ignition_point):
    """Same rollout + update as train_actor_critic.py, but the ignition point
    is passed in by the caller (main()'s per-episode random sample) rather
    than a fixed module constant."""
    steps, total_reward, buildings_destroyed, contained = collect_rollout(
        env, model, ignition_point, device, seed
    )
    policy_loss, value_loss, classification_loss, entropy, value_grad_norm, other_grad_norm = update_policy(
        model, optimizer, steps, device, return_normalizer
    )
    return {
        "n_ticks": len(steps),
        "reward": total_reward,
        "buildings_destroyed": buildings_destroyed,
        "contained": contained,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "classification_loss": classification_loss,
        "entropy": entropy,
        "value_grad_norm": value_grad_norm,
        "other_grad_norm": other_grad_norm,
    }


def run_eval_suite(model, env, episode_num, eval_writer, eval_file, device):
    """Deterministic eval (argmax, no exploration noise) on all 5 scenarios
    individually -- the 3 multi-ignition training points AND both held-out
    validation points -- each with its own heuristic-baseline delta printed
    inline. Unlike train_actor_critic.py's version, there is no single
    'the' training point here, so every scenario gets equal billing; this is
    the number that's supposed to reveal whether generalization is actually
    improving, not just training-point performance. Returns {name: result}."""
    scenarios = list(zip(MULTI_IGNITION_NAMES, MULTI_IGNITION_TRAINING_SCENARIO)) + \
        list(VALIDATION_IGNITION_POINTS.items())
    results = {}
    for name, point in scenarios:
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device)
        results[name] = result
        eval_writer.writerow({
            "episode": episode_num,
            "ignition_point_name": name,
            "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
        })
        eval_file.flush()
        kind = "train" if name in MULTI_IGNITION_NAMES else "VALIDATION"
        line = (f"    [eval @ ep {episode_num}] ({kind:10s}) {name}: avg_reward={result['avg_reward']:.1f}  "
                f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
                f"containment_rate={result['containment_rate']:.0%}")
        baseline = HEURISTIC_BASELINE.get(name)
        if baseline is not None:
            delta = result["avg_reward"] - baseline["avg_reward"]
            line += f"  |  vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})"
        print(line, flush=True)
    return results


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best.pt")

    # Re-raise SIGTERM as KeyboardInterrupt so an external kill (observed
    # twice now: no crash, no NaN, no Ctrl+C message, dead ~30-40 wall-clock
    # minutes in both times) hits the same graceful save-and-exit path
    # Ctrl+C already does -- Python's default SIGTERM handling does not do
    # this. See module docstring point 5.
    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = get_device()
    run_kind = "REAL TRAINING RUN" if N_EPISODES > 100 else "dry run"
    print(f"[train-multi] {run_kind}: {N_EPISODES} episodes, initial device={device}")
    print(f"[train-multi] Sampling uniformly at random per episode from: {MULTI_IGNITION_NAMES}")

    print("[train-multi] Building InfernoEnv...")
    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)

    model = InfernoModel(n_grid_channels=probe_obs["grid"].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()
    ignition_rng = random.Random(BASE_SEED)

    mps_errors = []
    ignition_counts = Counter()
    recent_wall_times = deque(maxlen=STATUS_EVERY)
    recent_rewards = deque(maxlen=STATUS_EVERY)
    last_eval_results = None
    best_avg_eval_reward = float("-inf")
    best_episode = None
    episode_rewards = []
    episode_losses = []  # (policy, value, classification, entropy)
    episode_wall_times = []
    completed_episodes = 0
    interrupted = False

    resuming = os.path.exists(RESUME_STATE_PATH)
    if resuming:
        print(f"[train-multi] Found {RESUME_STATE_PATH} -- resuming (not starting from episode 1).")
        resumed = load_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizer, ignition_rng)
        completed_episodes = resumed["completed_episodes"]
        best_avg_eval_reward = resumed["best_avg_eval_reward"]
        best_episode = resumed["best_episode"]
        ignition_counts = resumed["ignition_counts"]
        episode_rewards = resumed["episode_rewards"]
        episode_losses = resumed["episode_losses"]
        episode_wall_times = resumed["episode_wall_times"]
        print(f"[train-multi] Resuming from episode {completed_episodes + 1}/{N_EPISODES} "
              f"(best combined-avg eval reward so far: {best_avg_eval_reward:.1f} at ep {best_episode})")
    start_episode = completed_episodes + 1
    if start_episode > N_EPISODES:
        print(f"[train-multi] Resume state already shows {completed_episodes}/{N_EPISODES} episodes "
              f"complete -- nothing left to do. Delete {RESUME_STATE_PATH} to force a fresh run.")
        return

    log_mode = "a" if resuming else "w"
    with open(TRAIN_LOG_PATH, log_mode, newline="") as train_file, \
         open(EVAL_LOG_PATH, log_mode, newline="") as eval_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
        if not resuming:
            train_writer.writeheader()
            eval_writer.writeheader()

        try:
            for ep in range(start_episode, N_EPISODES + 1):
                ep_t0 = time.perf_counter()
                seed = BASE_SEED + ep

                ignition_idx = ignition_rng.randrange(len(MULTI_IGNITION_TRAINING_SCENARIO))
                ignition_point = MULTI_IGNITION_TRAINING_SCENARIO[ignition_idx]
                ignition_name = MULTI_IGNITION_NAMES[ignition_idx]

                try:
                    result = run_training_episode(
                        env, model, optimizer, device, seed, return_normalizer, ignition_point
                    )
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    op_name = _extract_op_name(e)
                    mps_errors.append(op_name)
                    print(f"[train-multi] MPS op not implemented: '{op_name}' -- "
                          f"falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    result = run_training_episode(
                        env, model, optimizer, device, seed, return_normalizer, ignition_point
                    )

                wall_s = time.perf_counter() - ep_t0
                episode_wall_times.append(wall_s)
                episode_rewards.append(result["reward"])
                episode_losses.append((result["policy_loss"], result["value_loss"],
                                        result["classification_loss"], result["entropy"]))
                recent_wall_times.append(wall_s)
                recent_rewards.append(result["reward"])
                completed_episodes = ep
                ignition_counts[ignition_name] += 1

                train_writer.writerow({
                    "episode": ep, "ignition_name": ignition_name, "device": str(device),
                    "n_ticks": result["n_ticks"],
                    "reward": result["reward"], "buildings_destroyed": result["buildings_destroyed"],
                    "contained": result["contained"], "policy_loss": result["policy_loss"],
                    "value_loss": result["value_loss"], "classification_loss": result["classification_loss"],
                    "entropy": result["entropy"], "value_grad_norm": result["value_grad_norm"],
                    "other_grad_norm": result["other_grad_norm"], "wall_time_s": wall_s,
                })
                train_file.flush()

                if ep % STATUS_EVERY == 0:
                    window_eps_per_min = len(recent_wall_times) / (sum(recent_wall_times) / 60.0)
                    window_avg_reward = sum(recent_rewards) / len(recent_rewards)
                    print(f"[status @ ep {ep:5d}/{N_EPISODES}] "
                          f"episodes/min(last{STATUS_EVERY})={window_eps_per_min:6.2f}  "
                          f"avg_reward(last{STATUS_EVERY})={window_avg_reward:10.1f}  "
                          f"entropy={result['entropy']:.3f}  "
                          f"policy_loss={result['policy_loss']:8.3f}  "
                          f"value_loss={result['value_loss']:8.3f}  "
                          f"class_loss={result['classification_loss']:.3f}  device={device}",
                          flush=True)

                if ep % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    torch.save(model.state_dict(), latest_ckpt_path)
                    save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizer, ignition_rng,
                                       completed_episodes, best_avg_eval_reward, best_episode, ignition_counts,
                                       episode_rewards, episode_losses, episode_wall_times)
                    print(f"  [checkpoint] saved {ckpt_path} (and latest.pt, resume_state.pt)", flush=True)

                if ep % EVAL_EVERY == 0:
                    last_eval_results = run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                    avg_eval_reward = sum(r["avg_reward"] for r in last_eval_results.values()) / len(last_eval_results)
                    print(f"    [eval @ ep {ep}] combined avg reward across all 5 scenarios: {avg_eval_reward:.1f}",
                          flush=True)
                    if avg_eval_reward > best_avg_eval_reward:
                        best_avg_eval_reward = avg_eval_reward
                        best_episode = ep
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"  [checkpoint] new best combined-avg eval reward ({avg_eval_reward:.1f}) "
                              f"at ep {ep} -- saved {best_ckpt_path}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[train-multi] Caught interrupt (Ctrl+C or SIGTERM) after episode "
                  f"{completed_episodes}/{N_EPISODES} -- saving checkpoint + resume state before exit.")
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizer, ignition_rng,
                               completed_episodes, best_avg_eval_reward, best_episode, ignition_counts,
                               episode_rewards, episode_losses, episode_wall_times)
            print(f"  [checkpoint] saved {latest_ckpt_path} and {RESUME_STATE_PATH}")

        if completed_episodes > 0:
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizer, ignition_rng,
                               completed_episodes, best_avg_eval_reward, best_episode, ignition_counts,
                               episode_rewards, episode_losses, episode_wall_times)

    # --- Summary (natural completion or Ctrl+C -- same block either way) ------
    total_wall = sum(episode_wall_times)
    episodes_per_min = completed_episodes / (total_wall / 60.0) if total_wall > 0 else float("nan")
    policy_losses = [p for p, v, c, e in episode_losses]
    value_losses = [v for p, v, c, e in episode_losses]
    class_losses = [c for p, v, c, e in episode_losses]
    entropies = [e for p, v, c, e in episode_losses]

    def _nan_or_frozen(values):
        if not values:
            return "no data"
        if any(math.isnan(v) for v in values):
            return "NaN encountered"
        if len(set(round(v, 6) for v in values)) <= 1:
            return "frozen (never changed)"
        return f"moved (range {min(values):.3g} to {max(values):.3g})"

    label = f"INTERRUPTED at episode {completed_episodes}/{N_EPISODES}" if interrupted \
        else f"completed all {completed_episodes} episodes"
    print(f"\n=== Multi-ignition training run summary ({label}) ===")
    print(f"Total wall time: {total_wall:.1f}s  Overall episodes/min: {episodes_per_min:.2f}")
    print(f"Ignition sampling distribution: {dict(ignition_counts)}")
    print(f"Policy loss: {_nan_or_frozen(policy_losses)}")
    print(f"Value loss: {_nan_or_frozen(value_losses)}")
    print(f"Classification loss: {_nan_or_frozen(class_losses)}")
    print(f"Entropy: {_nan_or_frozen(entropies)}")
    if episode_rewards:
        n_tail = min(20, len(episode_rewards))
        print(f"Stochastic rollout reward (all sampled ignition points combined), "
              f"first->last episode: {episode_rewards[0]:.1f} -> {episode_rewards[-1]:.1f}")
        print(f"Stochastic rollout reward, mean first {n_tail} vs last {n_tail}: "
              f"{sum(episode_rewards[:n_tail]) / n_tail:.1f} vs "
              f"{sum(episode_rewards[-n_tail:]) / n_tail:.1f}")
    if last_eval_results:
        print("Final deterministic eval (all 5 scenarios) vs heuristic baseline:")
        for name, result in last_eval_results.items():
            kind = "train" if name in MULTI_IGNITION_NAMES else "VALIDATION"
            print(f"  ({kind:10s}) {name}: avg_reward={result['avg_reward']:.1f}  "
                  f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
                  f"containment_rate={result['containment_rate']:.0%}")
            baseline = HEURISTIC_BASELINE.get(name)
            if baseline is not None:
                delta = result["avg_reward"] - baseline["avg_reward"]
                print(f"    heuristic:   avg_reward={baseline['avg_reward']:.1f}  "
                      f"avg_buildings_destroyed={baseline['avg_buildings_destroyed']:.1f}  "
                      f"containment_rate={baseline['containment_rate']:.0%}  "
                      f"(delta: {delta:+.1f})")
    else:
        print("No eval completed yet (fewer than EVAL_EVERY episodes ran).")
    if best_episode is not None:
        print(f"Best checkpoint: ep {best_episode}, combined avg eval reward = {best_avg_eval_reward:.1f} "
              f"-> {best_ckpt_path}")
    else:
        print("No best.pt saved yet (fewer than EVAL_EVERY episodes ran).")
    print(f"MPS errors encountered: {mps_errors if mps_errors else 'none'}")
    print(f"Final device: {device}")
    print(f"Latest checkpoint: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
