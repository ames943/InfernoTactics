"""
FIRST training loop in this project -- gradients flow and weights update here
for the first time. Everything upstream (env, fire physics, weather, depots,
model forward pass, eval_policy) was built and verified untrained; this file
is where that stops being true.

Algorithm: plain single-episode actor-critic (REINFORCE with a learned value
baseline), NOT PPO. Chosen over PPO for this first version because it needs
no importance-sampling ratio/clipping, no multiple-epoch replay of a stored
rollout, and no "old policy vs new policy" bookkeeping -- fewer places for a
first-ever gradient-flow bug to hide. The code is still structured as
rollout-then-update (see below), which is the natural skeleton to grow into
PPO later (add a clip term + multiple update epochs over the same rollout)
once this simpler version is confirmed working.

Rollout / update split (and why):
  1. ROLLOUT (torch.no_grad()): step through one full episode with the
     current policy, storing each tick's raw observation, sampled action,
     and reward. No computation graph is kept.
  2. UPDATE: after the episode ends (so discounted returns G_t are known for
     every t), loop back over the stored ticks ONE AT A TIME, re-running
     that tick's observation through the model WITH gradients enabled,
     computing that tick's policy/value/classification loss, and calling
     .backward() immediately (accumulating into .grad, no optimizer.step()
     yet). One optimizer.step() happens after the whole episode's ticks
     have been accumulated.
  This costs a second forward pass per tick (rollout's was no_grad, update's
  isn't) but never holds more than one tick's activations in memory at once.
  The grid is 316x595 -- holding all ~150 ticks' worth of per-cell CNN
  activations and (4, 316, 595) classification logits simultaneously (as a
  single batched update would) risks several GB of peak memory for what is
  still just a smoke test; this trades some speed for headroom while the
  mechanism itself is first being validated.

Loss = policy_loss + VALUE_LOSS_COEFF * value_loss + CLASSIFICATION_LOSS_COEFF * classification_loss - ENTROPY_COEFF * entropy
  - policy_loss: -log_prob(action) * advantage, advantage = (G_t - V(s_t)).detach()
  - value_loss: (V(s_t) - G_t)^2
  - classification_loss: cross-entropy of the classification head's per-cell
    logits against that SAME tick's true fire state (fire_sim ordinal,
    Burned Out folded into Safe -- see classification_head.fire_state_to_class).
    This looks tautological (the fire state is also one of the CNN's own
    input channels) but it isn't circular for the RL heads: the actor/critic
    only see the pooled fused feature vector, never the classification
    labels directly, so this loss's real job is regularizing the shared CNN
    features to stay fire-state-aware, per classification_head.py's own
    docstring/design (written before this file, followed as-is here).
  - entropy: resource_dist.entropy() + zone_dist.entropy(), SUBTRACTED (i.e.
    rewarded) to discourage the policy from collapsing to a near-deterministic
    distribution before it has actually learned anything.
  CLASSIFICATION_LOSS_COEFF = 0.3: middle of the user-specified 0.1-0.5
  range -- enough to meaningfully shape the CNN features, not enough to
  dominate the RL objective.
  ENTROPY_COEFF = 0.01: standard PPO-family default (e.g. the original PPO
  paper's Atari/MuJoCo configs use 0.0-0.01) -- small enough not to prevent
  the policy from ever committing to a good action, large enough to resist
  the collapse-to-zero-entropy failure mode a first smoke test surfaced (see
  below).

Return normalization (added after the first smoke test): G_t values were
running to the thousands (buildings destroyed alone are -100 each), which
made value_loss ((V - G_t)^2) start out ~1e8, four to six orders of
magnitude larger than policy_loss/classification_loss. A single shared
optimizer sees one combined gradient per parameter, so returns are
normalized with a RunningMeanStd (Welford's online algorithm, mean/variance
accumulated across every episode so far, EPSILON-initialized so episode 1
doesn't divide by zero) before use in both the value target and the
advantage -- chosen over a fixed scale constant because episode length (and
therefore return magnitude) varies enormously tick to tick (a 5-tick
early-contained episode vs. a full 150-tick one), so no single constant
would fit every episode, whereas a running estimate adapts as training
progresses. Normalizer stats update AFTER being used to normalize the
current episode's returns (no look-ahead).

Gradient clipping is applied to two disjoint parameter groups separately --
actor_critic.value_head's own parameters, and everything else (shared
CNN/MLP trunk + actor heads + classification head) -- rather than one
combined clip_grad_norm_ call across all parameters. This does NOT fully
decompose "value_loss's contribution vs. policy_loss's contribution" on the
SHARED trunk (both still land in the same accumulated .grad there, since
they share parameters); what it does guarantee is that the value head's own
dedicated weights -- structurally the closest thing to the squared-error
target, and empirically where the smoke test's exploding gradient was
concentrated -- can't force a single tiny global clip-scale factor onto the
actor/classification heads' comparatively small gradients the way one
shared clip_grad_norm_ call did before. Return normalization (above) already
fixes the root scale mismatch; this is a defense-in-depth measure against
future scale drift, not something actively relied on to fix today's
imbalance. Both groups' PRE-clip norms are logged every episode
(value_grad_norm, other_grad_norm columns) for visibility either way.

Device: prefers MPS, falls back to CPU (whole training run, not per-op) the
first time any op raises PyTorch's "not implemented for MPS" error -- see
_is_mps_unimplemented_error() / the try/except around each episode in main().
The first smoke test hit a real MPS gap in cnn_branch.py's AdaptiveAvgPool2d
(needs evenly-divisible input/output sizes; CPU doesn't) -- fixed at the
source in cnn_branch.py (dynamic zero-padding before the adaptive pool)
rather than papered over here, so MPS should now actually run.

Console output is a periodic status line every STATUS_EVERY episodes, NOT
one line per episode -- at N_EPISODES=2000 that would be 2000 lines of
noise. Every episode's full detail is still written to TRAIN_LOG_PATH
(logs/train_log.csv) regardless; the console is a live human-readable digest
on top of that, not a replacement for it.

Ctrl+C is caught around the training loop: saves models/checkpoints/latest.pt
immediately, then falls through to the same summary block a natural
completion would print (labeled INTERRUPTED, using whatever episodes
actually completed) -- an early stop still leaves a usable checkpoint and a
readable summary instead of losing the run.

N_EPISODES defaults to 2000 (a real run), overridable via the
INFERNO_N_EPISODES env var for quick dry runs, e.g.:
    INFERNO_N_EPISODES=3 python -m src.train.train_actor_critic

    python -m src.train.train_actor_critic
"""

import csv
import math
import os
import re
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
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402

# --- Hyperparameters (documented above / inline) ------------------------------
# 2000 = a real run (~3.2h at the ~5.8s/episode seen in the diagnostic run).
# Override with INFERNO_N_EPISODES for a quick dry run without editing this file.
N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 2000))
BASE_SEED = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.99  # discounted-return horizon; episodes run up to MAX_TICKS=150 ticks
VALUE_LOSS_COEFF = 0.5  # standard A2C/PPO default; meaningful now that returns are normalized (see below)
CLASSIFICATION_LOSS_COEFF = 0.3  # see module docstring
ENTROPY_COEFF = 0.01  # see module docstring
GRAD_CLIP_NORM = 0.5  # standard PPO-family default; applied per parameter-group, see module docstring

STATUS_EVERY = 20  # console status line cadence, see module docstring
CHECKPOINT_EVERY = 20
EVAL_EVERY = 20
EVAL_EPISODES = 3

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_log.csv")

TRAIN_LOG_FIELDS = [
    "episode", "device", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "ignition_point_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate",
]

# Heuristic baseline (src/train/heuristic_policy.py's _run_baseline_table(), 5
# episodes, seed=0, real weather, deterministic) -- hardcoded here as the fixed
# comparison target printed alongside every training-time eval, so "is RL closing
# the gap toward/past the heuristic" is visible live rather than only inferable
# from reward trending up in the abstract. "single_training" is the number that
# matters most: it's the same ignition point/scenario the RL agent is training on.
HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -29410.9, "avg_buildings_destroyed": 98.4, "containment_rate": 0.80},
    "mandeville_canyon": {"avg_reward": 30.5, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 46.1, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
}


class RunningMeanStd:
    """Welford's online mean/variance, used to normalize returns before they
    hit the loss (see module docstring's "Return normalization" section).
    epsilon-initialized count so the very first episode doesn't divide by
    zero; normalize() is called BEFORE update() each episode so an episode's
    own returns don't bias the statistics used to normalize it."""

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
    """Broad match: PyTorch's MPS-gap error wording varies by op (e.g.
    aten-dispatch gaps say "not currently implemented for the MPS device",
    but e.g. adaptive_avg_pool2d's says "not implemented on MPS device yet")
    -- checking for "mps" + "not implemented" together catches both rather
    than hardcoding one exact phrase."""
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


def collect_rollout(env, model, ignition_point, device, seed):
    """One full episode, current policy, NO gradients. Returns a list of
    per-tick dicts (raw numpy obs + sampled action + reward + classification
    target) plus episode-level summary stats."""
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
    """Re-run each stored tick WITH gradients, one at a time, accumulating
    .backward() before a single optimizer.step() -- see module docstring for
    why. Returns mean policy/value/classification loss, mean entropy, and
    the two clip groups' pre-clip gradient norms, all over the episode."""
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

    # Two disjoint parameter groups clipped separately -- see module docstring.
    value_grad_norm = torch.nn.utils.clip_grad_norm_(value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return (
        policy_loss_sum / n, value_loss_sum / n, classification_loss_sum / n, entropy_sum / n,
        float(value_grad_norm), float(other_grad_norm),
    )


def run_training_episode(env, model, optimizer, device, seed, return_normalizer):
    """One rollout + one update. Raises through any exception (including MPS
    unimplemented-op errors) for main()'s fallback handler to catch."""
    steps, total_reward, buildings_destroyed, contained = collect_rollout(
        env, model, TRAINING_IGNITION_POINT, device, seed
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
    """Deterministic eval (argmax, no exploration noise) on the training
    ignition point itself ("single_training") PLUS both held-out validation
    points. The training-point number here is a different signal than the
    stochastic rollout reward already logged every episode in
    TRAIN_LOG_PATH: it isolates "how good is the CURRENT greedy policy" from
    the exploration noise baked into the rollout used to compute gradients.
    Every scenario with a HEURISTIC_BASELINE entry also gets its delta vs
    that baseline printed inline. Returns {name: result} for the caller to
    keep as "last known" eval numbers, e.g. for the final/interrupted-run
    summary."""
    scenarios = [("single_training", TRAINING_IGNITION_POINT)] + list(VALIDATION_IGNITION_POINTS.items())
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
        line = (f"    [eval @ ep {episode_num}] {name}: avg_reward={result['avg_reward']:.1f}  "
                f"avg_buildings_destroyed={result['avg_buildings_destroyed']:.1f}  "
                f"containment_rate={result['containment_rate']:.0%}")
        baseline = HEURISTIC_BASELINE.get(name)
        if baseline is not None:
            delta = result["avg_reward"] - baseline["avg_reward"]
            line += f"  |  reward vs heuristic: {delta:+.1f} (heuristic: {baseline['avg_reward']:.1f})"
        print(line, flush=True)
    return results


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    latest_ckpt_path = os.path.join(CHECKPOINT_DIR, "latest.pt")

    device = get_device()
    run_kind = "REAL TRAINING RUN" if N_EPISODES > 100 else "dry run"
    print(f"[train] {run_kind}: {N_EPISODES} episodes, initial device={device}")

    print("[train] Building InfernoEnv...")
    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)

    model = InfernoModel(n_grid_channels=probe_obs["grid"].shape[0]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizer = RunningMeanStd()

    mps_errors = []
    episode_wall_times = []
    episode_rewards = []
    episode_losses = []  # (policy, value, classification, entropy)
    recent_wall_times = deque(maxlen=STATUS_EVERY)
    recent_rewards = deque(maxlen=STATUS_EVERY)
    last_eval_results = None
    completed_episodes = 0
    interrupted = False

    with open(TRAIN_LOG_PATH, "w", newline="") as train_file, \
         open(EVAL_LOG_PATH, "w", newline="") as eval_file:
        train_writer = csv.DictWriter(train_file, fieldnames=TRAIN_LOG_FIELDS)
        train_writer.writeheader()
        eval_writer = csv.DictWriter(eval_file, fieldnames=EVAL_LOG_FIELDS)
        eval_writer.writeheader()

        try:
            for ep in range(1, N_EPISODES + 1):
                ep_t0 = time.perf_counter()
                seed = BASE_SEED + ep

                try:
                    result = run_training_episode(env, model, optimizer, device, seed, return_normalizer)
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    op_name = _extract_op_name(e)
                    mps_errors.append(op_name)
                    print(f"[train] MPS op not implemented: '{op_name}' -- "
                          f"falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    result = run_training_episode(env, model, optimizer, device, seed, return_normalizer)

                wall_s = time.perf_counter() - ep_t0
                episode_wall_times.append(wall_s)
                episode_rewards.append(result["reward"])
                episode_losses.append((result["policy_loss"], result["value_loss"],
                                        result["classification_loss"], result["entropy"]))
                recent_wall_times.append(wall_s)
                recent_rewards.append(result["reward"])
                completed_episodes = ep

                train_writer.writerow({
                    "episode": ep, "device": str(device), "n_ticks": result["n_ticks"],
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
                    print(f"  [checkpoint] saved {ckpt_path} (and latest.pt)", flush=True)

                if ep % EVAL_EVERY == 0:
                    last_eval_results = run_eval_suite(model, env, ep, eval_writer, eval_file, device)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[train] Caught Ctrl+C after episode {completed_episodes}/{N_EPISODES} -- "
                  f"saving final checkpoint before exit.")
            torch.save(model.state_dict(), latest_ckpt_path)
            print(f"  [checkpoint] saved {latest_ckpt_path}")

        if completed_episodes > 0:
            # Unconditional final save so latest.pt is always current at exit,
            # even if completion didn't land on a CHECKPOINT_EVERY boundary
            # (the interrupt branch above already covers the Ctrl+C case;
            # this covers natural completion, redundantly-but-harmlessly).
            torch.save(model.state_dict(), latest_ckpt_path)

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
    print(f"\n=== Training run summary ({label}) ===")
    print(f"Total wall time: {total_wall:.1f}s  Overall episodes/min: {episodes_per_min:.2f}")
    print(f"Policy loss: {_nan_or_frozen(policy_losses)}")
    print(f"Value loss: {_nan_or_frozen(value_losses)}")
    print(f"Classification loss: {_nan_or_frozen(class_losses)}")
    print(f"Entropy: {_nan_or_frozen(entropies)}")
    if episode_rewards:
        n_tail = min(20, len(episode_rewards))
        print(f"Training-point reward, first->last episode: {episode_rewards[0]:.1f} -> {episode_rewards[-1]:.1f}")
        print(f"Training-point reward, mean first {n_tail} vs last {n_tail}: "
              f"{sum(episode_rewards[:n_tail]) / n_tail:.1f} vs "
              f"{sum(episode_rewards[-n_tail:]) / n_tail:.1f}")
    if last_eval_results:
        print("Final deterministic eval (all 3 scenarios) vs heuristic baseline:")
        for name, result in last_eval_results.items():
            print(f"  {name}: avg_reward={result['avg_reward']:.1f}  "
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
    print(f"MPS errors encountered: {mps_errors if mps_errors else 'none'}")
    print(f"Final device: {device}")
    print(f"Latest checkpoint: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
