"""
Multi-ignition training, v3 -- ONE additional fix on top of v2 (which is left
untouched, alongside its 1000-episode logs/checkpoints, as the second
comparison baseline: v1 = no fixes, v2 = per-scenario return normalization +
scenario one-hot + curriculum warm-start, still flat/oscillating on
topanga_ridge, sullivan_canyon, stone_canyon, mandeville_canyon,
getty_view_park after 1000 episodes -- only single_training, the
warm-started already-solved scenario, held steady at +32.1 the whole run).

v2's final numbers landed almost exactly back on v1's bands for every
non-anchor scenario (e.g. stone_canyon -121,138 in v2 vs. v1's ~-121k
throughout; mandeville_canyon -17,389 in v2, never once hitting the "good
mode" v1 occasionally found). Per-scenario return normalization alone wasn't
enough: it fixes the SCALE mismatch feeding into the value loss, but the
value head's own final Linear(128,1) layer is still ONE set of weights that
every scenario's backward pass writes into every single episode -- so even
with correctly-scaled per-scenario advantages, the critic itself has no way
to specialize per scenario. This is Fix 4 (originally "fix #3" on the user's
list, not yet tried): separate small value heads, one per training scenario,
sharing everything upstream (CNN, MLP, the actor's trunk/resource_type_head/
zone_head) -- see models/actor_critic.py's n_value_heads parameter (added
there, backward-compatible: n_value_heads=1 is byte-identical to the
original single-shared-head architecture, so v1/v2/heuristic_policy.py/
eval.py's own untrained-model check are all completely unaffected).

Honest scope of this fix: it only isolates the LAST layer's gradient
conflict. Because the value heads still branch off the SAME shared trunk
(ActorCritic.trunk, shared with the actor), that trunk's weights still
receive backward-pass signal from every scenario's value loss regardless of
which head it exits through. This is deliberate, matching exactly what the
user asked for ("shared CNN/MLP trunk + actor, but a distinct small value
head per training scenario") -- not a claim that this fully decouples the
scenarios, just the specific, bounded next experiment requested.

Everything else is unchanged from v2: per-scenario return normalization
(Fix 1), scenario one-hot input (Fix 2), curriculum sampling schedule (Fix 3)
-- same curriculum_weights(episode) function, unmodified, which for a
500-episode run only ever reaches partway through its ep300-1000 "shift
toward 50/50" phase (at ep500 it's at ~71/29, not 50/50 -- see v2's module
docstring for the full schedule; not rescaled to this run's shorter length,
since the instruction was to keep it as-is).

Warm-start (extending v2's Fix 3, not replacing it): loads
models/checkpoints_multi_v2/best.pt -- v2's OWN best checkpoint, not the
original single-scenario one -- since v2's checkpoint already has the
correctly-shaped one-hot-aware MLP input layer (N_SCALARS=13, matching this
run) and 1000 episodes of curriculum training baked in; going back to the
pre-one-hot single-scenario checkpoint would throw that away for no reason.
The one new wrinkle: v2's checkpoint has a single shared value_head
(Linear(128,1)); this run's model has 4 separate value_heads. All 4 are
initialized as identical COPIES of v2's one value_head's weights -- i.e.
every scenario's critic starts from exactly where v2 left off, and only
diverges from there as training proceeds. (Every other parameter -- cnn.*,
mlp.*, actor_critic.trunk/resource_type_head/zone_head, classifier.* -- is
shape-identical between v2 and v3 and loads directly, unchanged.)

Verification this fix is actually engaged (not just present in code): a
FIXED probe input (same random seed every run) is pushed through
actor_critic.trunk once per STATUS_EVERY, and each of the 4 value_heads'
output on that SAME frozen input is printed -- if the 4 numbers are
identical, the heads haven't diverged yet (they start as copies, per above);
if they've spread apart, that's direct proof gradients are actually flowing
into them differently per scenario, not just architecturally separate.

500 episodes only (not 1000, not 3500) -- a bounded, time-limited follow-up
experiment, not a fresh full run.

    python -m src.train.train_actor_critic_multi_v3                 # real 500-episode run
    INFERNO_N_EPISODES=50 python -m src.train.train_actor_critic_multi_v3   # smoke test
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
    SCALAR_KEYS,
    TRAINING_IGNITION_POINT,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    flatten_scalars,
)
from models.classification_head import fire_state_to_class  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402
from train.eval import eval_policy  # noqa: E402

TRAINING_SCENARIO_NAMES = ["single_training", "topanga_ridge", "sullivan_canyon", "stone_canyon"]
TRAINING_SCENARIO_POINTS = {
    "single_training": TRAINING_IGNITION_POINT,
    "topanga_ridge": MULTI_IGNITION_TRAINING_SCENARIO[0],
    "sullivan_canyon": MULTI_IGNITION_TRAINING_SCENARIO[1],
    "stone_canyon": MULTI_IGNITION_TRAINING_SCENARIO[2],
}
SCENARIO_ONEHOT_INDEX = {name: i for i, name in enumerate(TRAINING_SCENARIO_NAMES)}
UNSEEN_ONEHOT_INDEX = len(TRAINING_SCENARIO_NAMES)
ONEHOT_DIM = len(TRAINING_SCENARIO_NAMES) + 1
N_SCALARS = len(SCALAR_KEYS) + ONEHOT_DIM  # 13, unchanged from v2
N_VALUE_HEADS = len(TRAINING_SCENARIO_NAMES)  # 4 -- Fix 4

V2_BEST_CKPT = os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v2", "best.pt")

N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 500))  # bounded follow-up, not a fresh full run
BASE_SEED = 2000
LEARNING_RATE = 3e-4
GAMMA = 0.99
VALUE_LOSS_COEFF = 0.5
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.01
GRAD_CLIP_NORM = 0.5

STATUS_EVERY = 20
CHECKPOINT_EVERY = 20
EVAL_EVERY = 100
EVAL_EPISODES = 3

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", "checkpoints_multi_v3")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
TRAIN_LOG_PATH = os.path.join(LOG_DIR, "train_log_multi_v3.csv")
EVAL_LOG_PATH = os.path.join(LOG_DIR, "eval_log_multi_v3.csv")
RESUME_STATE_PATH = os.path.join(CHECKPOINT_DIR, "resume_state.pt")

TRAIN_LOG_FIELDS = [
    "episode", "scenario_name", "device", "n_ticks", "reward", "buildings_destroyed", "contained",
    "policy_loss", "value_loss", "classification_loss", "entropy",
    "value_grad_norm", "other_grad_norm", "wall_time_s",
]
EVAL_LOG_FIELDS = [
    "episode", "scenario_name", "avg_reward", "avg_buildings_destroyed",
    "avg_buildings_saved", "containment_rate",
]

HEURISTIC_BASELINE = {
    "single_training": {"avg_reward": -29410.9, "avg_buildings_destroyed": 98.4, "containment_rate": 0.80},
    "topanga_ridge": {"avg_reward": -5030.5, "avg_buildings_destroyed": 18.6, "containment_rate": 0.80},
    "sullivan_canyon": {"avg_reward": -5519.5, "avg_buildings_destroyed": 22.0, "containment_rate": 0.80},
    "stone_canyon": {"avg_reward": -300.3, "avg_buildings_destroyed": 1.0, "containment_rate": 1.00},
    "mandeville_canyon": {"avg_reward": 30.5, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
    "getty_view_park": {"avg_reward": 46.1, "avg_buildings_destroyed": 0.0, "containment_rate": 1.00},
}


def curriculum_weights(episode):
    """Identical to v2 -- see that file's docstring. Not rescaled for this
    run's shorter length (deliberate, per the user's "keep everything else
    from v2 unchanged")."""
    if episode <= 300:
        p_single = 0.80
    elif episode <= 1000:
        frac = (episode - 300) / 700.0
        p_single = 0.80 - 0.30 * frac
    else:
        p_single = 0.25
    p_other = (1.0 - p_single) / 3.0
    return [p_single, p_other, p_other, p_other]


def scenario_one_hot(scenario_name):
    vec = np.zeros(ONEHOT_DIM, dtype=np.float32)
    vec[SCENARIO_ONEHOT_INDEX.get(scenario_name, UNSEEN_ONEHOT_INDEX)] = 1.0
    return vec


def build_scalars(obs_scalars, scenario_name):
    return np.concatenate([flatten_scalars(obs_scalars), scenario_one_hot(scenario_name)]).astype(np.float32)


def build_model(n_grid_channels, device):
    """Fix 4's warm-start: v2's best.pt (not the original single-scenario
    one -- see module docstring) for every parameter except the value head,
    which is split into N_VALUE_HEADS identical copies of v2's one shared
    value_head."""
    model = InfernoModel(n_grid_channels=n_grid_channels, n_scalars=N_SCALARS, n_value_heads=N_VALUE_HEADS).to(device)
    if not os.path.exists(V2_BEST_CKPT):
        print(f"[train-multi-v3] WARNING: {V2_BEST_CKPT} not found -- "
              f"falling back to random init instead of warm-start.")
        return model

    v2_sd = torch.load(V2_BEST_CKPT, map_location=device, weights_only=False)
    new_sd = model.state_dict()
    for key, value in v2_sd.items():
        if key in ("actor_critic.value_head.weight", "actor_critic.value_head.bias"):
            continue  # handled below -- v2 has one value_head, this model has N_VALUE_HEADS
        new_sd[key] = value
    for i in range(N_VALUE_HEADS):
        new_sd[f"actor_critic.value_heads.{i}.weight"] = v2_sd["actor_critic.value_head.weight"].clone()
        new_sd[f"actor_critic.value_heads.{i}.bias"] = v2_sd["actor_critic.value_head.bias"].clone()
    model.load_state_dict(new_sd)
    print(f"[train-multi-v3] Warm-started from {V2_BEST_CKPT}: all layers loaded unchanged except the value head, "
          f"which was split into {N_VALUE_HEADS} identical copies of v2's single shared value_head "
          f"(one per training scenario: {TRAINING_SCENARIO_NAMES}) -- they start tied and diverge from here.")
    return model


class RunningMeanStd:
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


def make_probe_fused(device):
    """A FIXED (seeded) random fused-feature vector, reused every status
    print -- see module docstring's "Verification" section. Independent of
    any real observation; its only job is to be an unchanging input we can
    push through the 4 value heads to see whether their outputs have
    diverged from each other."""
    g = torch.Generator().manual_seed(13)
    fused_dim = 128 + 128  # CNNBranch.POOLED_DIM + MLPBranch.OUTPUT_DIM, both fixed at 128
    return torch.randn(1, fused_dim, generator=g).to(device)


def log_value_head_divergence(model, probe_fused):
    with torch.no_grad():
        h = model.actor_critic.trunk(probe_fused)
        values = [float(model.actor_critic.value_heads[i](h).item()) for i in range(N_VALUE_HEADS)]
    parts = "  ".join(f"{name}={v:.4f}" for name, v in zip(TRAINING_SCENARIO_NAMES, values))
    spread = max(values) - min(values)
    print(f"    [value-head probe] {parts}  (spread={spread:.4f})", flush=True)


def save_resume_state(path, model, optimizer, return_normalizers, ignition_rng, completed_episodes,
                       best_avg_eval_reward, best_episode, scenario_counts, episode_rewards,
                       episode_losses, episode_wall_times):
    tmp_path = path + ".tmp"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "return_normalizers": {
            name: {"mean": rn.mean, "var": rn.var, "count": rn.count}
            for name, rn in return_normalizers.items()
        },
        "ignition_rng_state": ignition_rng.getstate(),
        "completed_episodes": completed_episodes,
        "best_avg_eval_reward": best_avg_eval_reward,
        "best_episode": best_episode,
        "scenario_counts": dict(scenario_counts),
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "episode_wall_times": episode_wall_times,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_resume_state(path, model, optimizer, return_normalizers, ignition_rng):
    state = torch.load(path, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    for name, rn in return_normalizers.items():
        saved = state["return_normalizers"][name]
        rn.mean, rn.var, rn.count = saved["mean"], saved["var"], saved["count"]
    ignition_rng.setstate(state["ignition_rng_state"])
    return {
        "completed_episodes": state["completed_episodes"],
        "best_avg_eval_reward": state["best_avg_eval_reward"],
        "best_episode": state["best_episode"],
        "scenario_counts": Counter(state["scenario_counts"]),
        "episode_rewards": state["episode_rewards"],
        "episode_losses": state["episode_losses"],
        "episode_wall_times": state["episode_wall_times"],
    }


def collect_rollout(env, model, ignition_point, scenario_name, value_head_idx, device, seed):
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps = []
    total_reward = 0.0
    buildings_destroyed = 0
    done = False
    info = None

    with torch.no_grad():
        while not done:
            grid_t, _ = InfernoModel.obs_to_tensors(obs, device=device)
            scalars_np = build_scalars(obs["scalars"], scenario_name)
            scalars_t = torch.from_numpy(scalars_np).unsqueeze(0).to(device)
            action_logits, _value, _classification_logits = model(grid_t, scalars_t, value_head_idx=value_head_idx)
            resource_idx = int(Categorical(logits=action_logits["resource_type"][0]).sample())
            zone_idx = int(Categorical(logits=action_logits["zone"][0]).sample())

            fire_state_target = fire_state_to_class(torch.from_numpy(obs["grid"][-1]).long())

            action = (RESOURCE_TYPES[resource_idx], zone_idx)
            next_obs, reward, done, info = env.step(action)

            steps.append({
                "grid": obs["grid"],
                "scalars": scalars_np,
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


def update_policy(model, optimizer, steps, value_head_idx, device, return_normalizer, all_value_head_params):
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = return_normalizer.normalize(raw_returns)
    return_normalizer.update(raw_returns)

    value_head_param_ids = {id(p) for p in all_value_head_params}
    other_params = [p for p in model.parameters() if id(p) not in value_head_param_ids]

    optimizer.zero_grad()
    policy_loss_sum = value_loss_sum = classification_loss_sum = entropy_sum = 0.0

    for step, g_t in zip(steps, returns):
        grid_t = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars_t = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        action_logits, value, classification_logits = model(grid_t, scalars_t, value_head_idx=value_head_idx)

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

    value_grad_norm = torch.nn.utils.clip_grad_norm_(all_value_head_params, GRAD_CLIP_NORM)
    other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, GRAD_CLIP_NORM)
    optimizer.step()

    n = len(steps)
    return (
        policy_loss_sum / n, value_loss_sum / n, classification_loss_sum / n, entropy_sum / n,
        float(value_grad_norm), float(other_grad_norm),
    )


def run_training_episode(env, model, optimizer, device, seed, return_normalizer, ignition_point, scenario_name,
                          value_head_idx, all_value_head_params):
    steps, total_reward, buildings_destroyed, contained = collect_rollout(
        env, model, ignition_point, scenario_name, value_head_idx, device, seed
    )
    policy_loss, value_loss, classification_loss, entropy, value_grad_norm, other_grad_norm = update_policy(
        model, optimizer, steps, value_head_idx, device, return_normalizer, all_value_head_params
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
    """value_head_idx is irrelevant here (0 always) -- eval_policy() discards
    the value output entirely, it only ever uses action_logits for
    (deterministic) action selection. Only the scenario one-hot (via
    scalars_fn) affects eval's actual behavior."""
    scenarios = [(name, TRAINING_SCENARIO_POINTS[name]) for name in TRAINING_SCENARIO_NAMES] + \
        list(VALIDATION_IGNITION_POINTS.items())
    results = {}
    for name, point in scenarios:
        result = eval_policy(model, env, ignition_point=point, n_episodes=EVAL_EPISODES,
                              use_real_weather=True, deterministic=True, seed=BASE_SEED, device=device,
                              scalars_fn=lambda s, n=name: build_scalars(s, n))
        results[name] = result
        eval_writer.writerow({
            "episode": episode_num,
            "scenario_name": name,
            "avg_reward": result["avg_reward"],
            "avg_buildings_destroyed": result["avg_buildings_destroyed"],
            "avg_buildings_saved": result["avg_buildings_saved"],
            "containment_rate": result["containment_rate"],
        })
        eval_file.flush()
        kind = "train" if name in TRAINING_SCENARIO_NAMES else "VALIDATION"
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

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, _handle_sigterm)

    device = get_device()
    run_kind = "REAL TRAINING RUN" if N_EPISODES > 100 else "dry run"
    print(f"[train-multi-v3] {run_kind}: {N_EPISODES} episodes, initial device={device}")
    print(f"[train-multi-v3] Training scenarios: {TRAINING_SCENARIO_NAMES}, N_VALUE_HEADS={N_VALUE_HEADS} (Fix 4)")

    print("[train-multi-v3] Building InfernoEnv...")
    env = InfernoEnv(seed=BASE_SEED)
    probe_obs = env.reset(seed=BASE_SEED)

    model = build_model(n_grid_channels=probe_obs["grid"].shape[0], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return_normalizers = {name: RunningMeanStd() for name in TRAINING_SCENARIO_NAMES}
    ignition_rng = random.Random(BASE_SEED)
    probe_fused = make_probe_fused(device)
    all_value_head_params = [p for vh in model.actor_critic.value_heads for p in vh.parameters()]

    mps_errors = []
    scenario_counts = Counter()
    recent_wall_times = deque(maxlen=STATUS_EVERY)
    recent_rewards = deque(maxlen=STATUS_EVERY)
    last_eval_results = None
    best_avg_eval_reward = float("-inf")
    best_episode = None
    episode_rewards = []
    episode_losses = []
    episode_wall_times = []
    completed_episodes = 0
    interrupted = False

    resuming = os.path.exists(RESUME_STATE_PATH)
    if resuming:
        print(f"[train-multi-v3] Found {RESUME_STATE_PATH} -- resuming (not starting from episode 1).")
        resumed = load_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizers, ignition_rng)
        completed_episodes = resumed["completed_episodes"]
        best_avg_eval_reward = resumed["best_avg_eval_reward"]
        best_episode = resumed["best_episode"]
        scenario_counts = resumed["scenario_counts"]
        episode_rewards = resumed["episode_rewards"]
        episode_losses = resumed["episode_losses"]
        episode_wall_times = resumed["episode_wall_times"]
        print(f"[train-multi-v3] Resuming from episode {completed_episodes + 1}/{N_EPISODES} "
              f"(best combined-avg eval reward so far: {best_avg_eval_reward:.1f} at ep {best_episode})")
        all_value_head_params = [p for vh in model.actor_critic.value_heads for p in vh.parameters()]
    start_episode = completed_episodes + 1
    if start_episode > N_EPISODES:
        print(f"[train-multi-v3] Resume state already shows {completed_episodes}/{N_EPISODES} episodes "
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

                weights = curriculum_weights(ep)
                scenario_name = ignition_rng.choices(TRAINING_SCENARIO_NAMES, weights=weights, k=1)[0]
                ignition_point = TRAINING_SCENARIO_POINTS[scenario_name]
                value_head_idx = SCENARIO_ONEHOT_INDEX[scenario_name]

                try:
                    result = run_training_episode(
                        env, model, optimizer, device, seed, return_normalizers[scenario_name],
                        ignition_point, scenario_name, value_head_idx, all_value_head_params
                    )
                except Exception as e:
                    if not _is_mps_unimplemented_error(e):
                        raise
                    op_name = _extract_op_name(e)
                    mps_errors.append(op_name)
                    print(f"[train-multi-v3] MPS op not implemented: '{op_name}' -- "
                          f"falling back to CPU for the rest of this run.")
                    device = torch.device("cpu")
                    model = model.to(device)
                    probe_fused = probe_fused.to(device)
                    all_value_head_params = [p for vh in model.actor_critic.value_heads for p in vh.parameters()]
                    result = run_training_episode(
                        env, model, optimizer, device, seed, return_normalizers[scenario_name],
                        ignition_point, scenario_name, value_head_idx, all_value_head_params
                    )

                wall_s = time.perf_counter() - ep_t0
                episode_wall_times.append(wall_s)
                episode_rewards.append(result["reward"])
                episode_losses.append((result["policy_loss"], result["value_loss"],
                                        result["classification_loss"], result["entropy"]))
                recent_wall_times.append(wall_s)
                recent_rewards.append(result["reward"])
                completed_episodes = ep
                scenario_counts[scenario_name] += 1

                train_writer.writerow({
                    "episode": ep, "scenario_name": scenario_name, "device": str(device),
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
                    norm_parts = []
                    for name in TRAINING_SCENARIO_NAMES:
                        rn = return_normalizers[name]
                        norm_parts.append(f"{name}: mean={rn.mean:11.1f} std={math.sqrt(rn.var):10.1f} n={int(rn.count)}")
                    print("    [return-norm] " + "  |  ".join(norm_parts), flush=True)
                    log_value_head_divergence(model, probe_fused)
                    print(f"    [curriculum] weights@ep{ep}={dict(zip(TRAINING_SCENARIO_NAMES, weights))}  "
                          f"cumulative sample counts={dict(scenario_counts)}", flush=True)

                if ep % CHECKPOINT_EVERY == 0:
                    ckpt_path = os.path.join(CHECKPOINT_DIR, f"episode_{ep:04d}.pt")
                    torch.save(model.state_dict(), ckpt_path)
                    torch.save(model.state_dict(), latest_ckpt_path)
                    save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizers, ignition_rng,
                                       completed_episodes, best_avg_eval_reward, best_episode, scenario_counts,
                                       episode_rewards, episode_losses, episode_wall_times)
                    print(f"  [checkpoint] saved {ckpt_path} (and latest.pt, resume_state.pt)", flush=True)

                if ep % EVAL_EVERY == 0:
                    last_eval_results = run_eval_suite(model, env, ep, eval_writer, eval_file, device)
                    avg_eval_reward = sum(r["avg_reward"] for r in last_eval_results.values()) / len(last_eval_results)
                    print(f"    [eval @ ep {ep}] combined avg reward across all 6 scenarios: {avg_eval_reward:.1f}",
                          flush=True)
                    if avg_eval_reward > best_avg_eval_reward:
                        best_avg_eval_reward = avg_eval_reward
                        best_episode = ep
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f"  [checkpoint] new best combined-avg eval reward ({avg_eval_reward:.1f}) "
                              f"at ep {ep} -- saved {best_ckpt_path}", flush=True)
        except KeyboardInterrupt:
            interrupted = True
            print(f"\n[train-multi-v3] Caught interrupt (Ctrl+C or SIGTERM) after episode "
                  f"{completed_episodes}/{N_EPISODES} -- saving checkpoint + resume state before exit.")
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizers, ignition_rng,
                               completed_episodes, best_avg_eval_reward, best_episode, scenario_counts,
                               episode_rewards, episode_losses, episode_wall_times)
            print(f"  [checkpoint] saved {latest_ckpt_path} and {RESUME_STATE_PATH}")

        if completed_episodes > 0:
            torch.save(model.state_dict(), latest_ckpt_path)
            save_resume_state(RESUME_STATE_PATH, model, optimizer, return_normalizers, ignition_rng,
                               completed_episodes, best_avg_eval_reward, best_episode, scenario_counts,
                               episode_rewards, episode_losses, episode_wall_times)

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
    print(f"\n=== Multi-ignition v3 (+ separate value heads per scenario) summary ({label}) ===")
    print(f"Total wall time: {total_wall:.1f}s  Overall episodes/min: {episodes_per_min:.2f}")
    print(f"Scenario sampling distribution: {dict(scenario_counts)}")
    print(f"Policy loss: {_nan_or_frozen(policy_losses)}")
    print(f"Value loss: {_nan_or_frozen(value_losses)}")
    print(f"Classification loss: {_nan_or_frozen(class_losses)}")
    print(f"Entropy: {_nan_or_frozen(entropies)}")
    print("Final per-scenario return normalizer stats:")
    for name in TRAINING_SCENARIO_NAMES:
        rn = return_normalizers[name]
        print(f"  {name}: mean={rn.mean:.1f}  std={math.sqrt(rn.var):.1f}  n={int(rn.count)}")
    print("Final value-head divergence probe (proof Fix 4 diverged, or didn't):")
    log_value_head_divergence(model, probe_fused)
    if episode_rewards:
        n_tail = min(20, len(episode_rewards))
        print(f"Stochastic rollout reward (all sampled scenarios combined), "
              f"first->last episode: {episode_rewards[0]:.1f} -> {episode_rewards[-1]:.1f}")
        print(f"Stochastic rollout reward, mean first {n_tail} vs last {n_tail}: "
              f"{sum(episode_rewards[:n_tail]) / n_tail:.1f} vs "
              f"{sum(episode_rewards[-n_tail:]) / n_tail:.1f}")
    if last_eval_results:
        print("Final deterministic eval (all 6 scenarios) vs heuristic baseline:")
        for name, result in last_eval_results.items():
            kind = "train" if name in TRAINING_SCENARIO_NAMES else "VALIDATION"
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
