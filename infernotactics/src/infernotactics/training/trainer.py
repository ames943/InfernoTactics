"""V8/v10 training loop: randomized ignitions with fire-relative actions
and policy-decided list-only multi-dispatch per simulation tick.

Uses the proven pre-v4 Monte Carlo loop.  The action representation is
fire-relative (semantic targets), the dispatch count per tick is chosen
by the policy, and the environment runs in synthetic-traffic mode with
configurable per-resource delays.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from infernotactics.operations.environment import (
    GRID_META_PATH,
    GRID_STATIC_PATH,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    TRAINING_IGNITION_POINT,
)
from infernotactics.data.config import PROJECT_ROOT
from infernotactics.policy.models.classification_head import fire_state_to_class
from infernotactics.policy.models import RelativeInfernoModel
from infernotactics.policy.targets import (  # noqa: E402
    N_TARGET_TYPES,
    TARGET_TYPES,
    decode_action,
    resolve_relative_targets,
)
from infernotactics.domain.zones import build_zone_boundaries  # noqa: E402
from infernotactics.domain.observations import SCALAR_KEYS, flatten_scalars  # noqa: E402
from infernotactics.domain.resources import RESOURCE_TYPES  # noqa: E402
from infernotactics.policy import forward_policy  # noqa: E402
from infernotactics.training.checkpoints import (  # noqa: E402
    build_training_checkpoint,
    checkpoint_metadata,
    load_checkpoint,
    restore_training_state,
)
from infernotactics.training.normalization import RunningMeanStd  # noqa: E402
from infernotactics.world import world_fingerprint  # noqa: E402
from infernotactics.training.run_logger import RunLogger, summarize_episode
from infernotactics.training.progress import EpisodeProgress


# Hyperparameters (previously in train_actor_critic.py; inlined here after that
# file was removed in the v10 reorganization to keep only the active pipeline).
CLASSIFICATION_LOSS_COEFF = 0.3
ENTROPY_COEFF = 0.02  # bumped from 0.01 to maintain exploration under v11's larger reward magnitudes
GAMMA = 0.99
GRAD_CLIP_NORM = 0.5


def compute_returns(rewards, gamma):
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def get_device(force_cpu: bool = False):
    # Try DirectML (AMD/Intel GPUs on Windows) for inference
    try:
        import torch_directml
        if torch_directml.is_available() and not force_cpu:
            return torch_directml.device()
    except Exception:
        pass
    # Fallback: CUDA
    if torch.cuda.is_available() and not force_cpu:
        return torch.device("cuda")
    # Fallback: CPU
    return torch.device("cpu")


def _categorical_log_prob(logits, value):
    """log_prob for a Categorical without torch.gather. DirectML can't
    backprop through gather when the sampled dim sizes differ, so select
    the sampled entry with a one-hot dot product instead."""
    logp = F.log_softmax(logits, dim=-1)
    onehot = torch.zeros(logp.shape[-1], dtype=logp.dtype, device=logp.device)
    onehot[value.to(device=logp.device).long()] = 1.0
    return torch.dot(logp, onehot)


def _categorical_entropy(logits):
    """Shannon entropy of a Categorical logits vector, gather-free."""
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return -(p * logp).sum(-1)



def _setting(name, legacy_name, default):
    """Read the stable setting name, retaining old experiment compatibility."""
    return os.environ.get(name, os.environ.get(legacy_name, default))


N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 2000))
BASE_SEED = 8200
LEARNING_RATE = float(_setting("INFERNO_LEARNING_RATE", "INFERNO_V8_LR", 1e-4))
AUX_TARGET_LOSS_COEFF = float(_setting("INFERNO_AUX_COEFF", "INFERNO_V8_AUX_COEFF", 0.05))
STATUS_EVERY = 20
EVAL_EVERY = int(_setting("INFERNO_EVAL_EVERY", "INFERNO_V8_EVAL_EVERY", 50))
RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "relative_v12_landfire")
CHECKPOINT_EVERY = int(_setting("INFERNO_CHECKPOINT_EVERY", "INFERNO_V8_CHECKPOINT_EVERY", 2))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_{RUN_TAG}")
MAX_DISPATCH_SLOTS = int(os.environ.get("INFERNO_MAX_DISPATCH_SLOTS", 10))
TRACE_EVERY = int(os.environ.get("INFERNO_TRACE_EVERY", 0))
PROGRESS_ENABLED = os.environ.get("INFERNO_PROGRESS_ENABLED", "1") != "0"
_PROGRESS_START_RAW = os.environ.get("INFERNO_PROGRESS_START")
PROGRESS_START = int(_PROGRESS_START_RAW) if _PROGRESS_START_RAW else None
PROGRESS_WINDOW = int(os.environ.get("INFERNO_ROLLING_WINDOW", 50))
FORCE_CPU = os.environ.get("INFERNO_FORCE_CPU", "0") != "0"

def _resource_mask(obs):
    return torch.tensor([
        obs["scalars"][f"{rtype}_available"] > 0 for rtype in RESOURCE_TYPES
    ], dtype=torch.bool)


_forward = forward_policy  # compatibility for historical notebooks/imports


def collect_rollout(env, model, ignition_point, device, seed):
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps, total_reward, buildings_destroyed = [], 0.0, 0
    done = False
    info = None
    with torch.no_grad():
        while not done:
            tick_obs = obs
            tick_actions = []
            logits, _value, _classification, raw_zones, raw_features = _forward(
                model, tick_obs, env, device
            )
            local_available = {
                rtype: int(tick_obs["scalars"][f"{rtype}_available"])
                for rtype in RESOURCE_TYPES
            }
            for _ in range(MAX_DISPATCH_SLOTS):
                resource_logits = logits["resource_type"][0].clone()
                available = torch.tensor(
                    [local_available[rtype] > 0 for rtype in RESOURCE_TYPES],
                    dtype=torch.bool, device=device,
                )
                resource_logits[~available] = -1e9
                if not bool(available.any()):
                    break
                resource_idx = int(Categorical(logits=resource_logits).sample())
                target_idx = int(Categorical(logits=logits["target"][0, resource_idx]).sample())
                action = decode_action(resource_idx, target_idx, raw_zones)
                tick_actions.append({
                    "resource_idx": resource_idx,
                    "target_idx": target_idx,
                    "resource_mask": available.detach().cpu().numpy(),
                    "is_stop": action is None,
                })
                if action is None:
                    break
                local_available[RESOURCE_TYPES[resource_idx]] -= 1

            actions = [
                decode_action(a["resource_idx"], a["target_idx"], raw_zones)
                for a in tick_actions
                if not a["is_stop"]
            ]
            next_obs, reward, done, info = env.step([a for a in actions if a is not None])
            steps.append({
                "grid": tick_obs["grid"],
                "scalars": flatten_scalars(tick_obs["scalars"]),
                "target_zones": raw_zones,
                "target_features": raw_features,
                "actions": tick_actions,
                # One-step prediction target. The old auxiliary task copied
                # the current fire-state channel already present in its own
                # input and could be solved as an identity function.
                "fire_state_target": fire_state_to_class(
                    torch.from_numpy(next_obs["grid"][-1]).long()
                ),
                "info": info,
                "reward": reward,
            })
            total_reward += reward
            buildings_destroyed += info["buildings_destroyed"]
            obs = next_obs
    return steps, total_reward, buildings_destroyed, info["contained"]


def _aux_target(resource_idx, target_zones):
    # Suppression and nearest-fire candidates are interchangeable operationally;
    # prefer the semantic candidate that exposes the transferable rule.
    preferred = {0: 0, 1: 2, 2: 3, 3: 0}[resource_idx]
    if target_zones[resource_idx, preferred] >= 0:
        return preferred
    valid = np.flatnonzero(target_zones[resource_idx] >= 0)
    return int(valid[0]) if len(valid) else N_TARGET_TYPES - 1


def update_policy(model, optimizer, steps, device, normalizer):
    raw_returns = compute_returns([s["reward"] for s in steps], GAMMA)
    returns = normalizer.normalize(raw_returns)
    normalizer.update(raw_returns)
    optimizer.zero_grad()
    sums = [0.0] * 7
    for step, g_t in zip(steps, returns):
        grid = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        target_zones_np = step["target_zones"]
        zones = torch.from_numpy(target_zones_np).unsqueeze(0).to(device)
        features_np = step["target_features"]
        features = torch.from_numpy(features_np).unsqueeze(0).to(device)
        logits, value, classification = model(grid, scalars, zones, features)
        log_prob = torch.tensor(0.0, device=device)
        entropy = torch.tensor(0.0, device=device)
        target_entropy = torch.tensor(0.0, device=device)
        resource_entropy = torch.tensor(0.0, device=device)
        aux_loss = torch.tensor(0.0, device=device)
        for action_data in step["actions"]:
            resource_idx = action_data["resource_idx"]
            target_idx = action_data["target_idx"]
            resource_logits = logits["resource_type"][0].clone()
            resource_mask = torch.from_numpy(action_data["resource_mask"]).to(device)
            resource_logits[~resource_mask] = -1e9
            log_prob = log_prob + _categorical_log_prob(resource_logits, torch.tensor(resource_idx, device=device)) \
                + _categorical_log_prob(logits["target"][0, resource_idx], torch.tensor(target_idx, device=device))
            entropy = entropy + _categorical_entropy(logits["target"][0, resource_idx])
            target_entropy = target_entropy + _categorical_entropy(logits["target"][0, resource_idx])
            resource_entropy = resource_entropy + _categorical_entropy(resource_logits)
            aux_idx = _aux_target(resource_idx, target_zones_np)
            aux_loss = aux_loss + F.cross_entropy(
                logits["target"][0, resource_idx].unsqueeze(0),
                torch.tensor([aux_idx], device=device),
            )
        value_scalar = value.squeeze()
        target_return = torch.tensor(g_t, dtype=value_scalar.dtype, device=device)
        advantage = (target_return - value_scalar).detach()
        classification_loss = F.cross_entropy(classification, step["fire_state_target"].unsqueeze(0).to(device))
        policy_loss = -log_prob * advantage
        value_loss = (value_scalar - target_return) ** 2
        loss = policy_loss + 0.5 * value_loss + CLASSIFICATION_LOSS_COEFF * classification_loss + AUX_TARGET_LOSS_COEFF * aux_loss - ENTROPY_COEFF * entropy
        loss.backward()
        sums[0] += policy_loss.item(); sums[1] += value_loss.item(); sums[2] += classification_loss.item()
        sums[3] += aux_loss.item(); sums[4] += entropy.item(); sums[5] += resource_entropy.item(); sums[6] += target_entropy.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    optimizer.step()
    n = len(steps)
    return tuple(x / n for x in sums)


def resolve_relative_targets_from_state(grid, scalars, target_zones):
    """Rebuild candidate features during update without retaining env objects."""
    fire = grid[-1]
    active = np.isin(fire, (2, 3))
    fuel = fire == 1
    adjacent = _dilate(active) & fuel
    building = grid[2] > 0.10
    population = grid[7]
    rows, cols = fire.shape
    boundaries = build_zone_boundaries(rows, cols)
    out = np.zeros((len(RESOURCE_TYPES), N_TARGET_TYPES, 10), dtype=np.float32)
    for r in range(len(RESOURCE_TYPES)):
        for t, zone in enumerate(target_zones[r]):
            if zone < 0:
                continue
            # Legacy reconstruction helper retained for old diagnostics. New
            # rollouts persist the exact candidate features used for action
            # selection and do not call this approximation during updates.
            bounds = boundaries[int(zone)]
            r0, r1 = bounds.row_range
            c0, c1 = bounds.col_range
            region = fire[r0:r1, c0:c1]
            threat = building[r0:r1, c0:c1] & active[r0:r1, c0:c1]
            out[r, t] = [1.0, min(float(active[r0:r1, c0:c1].sum()) / 256, 1), min(float(adjacent[r0:r1, c0:c1].sum()) / 256, 1), float(population[r0:r1, c0:c1][threat].max()) if threat.any() else 0.0, float((region == 1).mean()), float(building[r0:r1, c0:c1].mean()), 0.0, r0 / rows, c0 / cols, float(active[r0:r1, c0:c1].any())]
    return out


def _dilate(mask):
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            out[max(0, dr):min(mask.shape[0], mask.shape[0] + dr), max(0, dc):min(mask.shape[1], mask.shape[1] + dc)] |= mask[max(0, -dr):min(mask.shape[0], mask.shape[0] - dr), max(0, -dc):min(mask.shape[1], mask.shape[1] - dc)]
    return out


def evaluate(model, env, point, device, episodes=2):
    rewards, destroyed, contained = [], [], []
    model.eval()
    with torch.no_grad():
        for ep in range(episodes):
            obs = env.reset(ignition_point=point, seed=BASE_SEED + ep, use_real_weather=True)
            total, lost, done = 0.0, 0, False
            while not done:
                logits, _v, _c, zones, _features = _forward(model, obs, env, device)
                resource_logits = logits["resource_type"][0].clone()
                available = _resource_mask(obs).to(device)
                resource_logits[~available] = -1e9
                ri = int(torch.argmax(resource_logits)) if bool(available.any()) else 0
                ti = int(torch.argmax(logits["target"][0, ri]))
                action = decode_action(ri, ti, zones)
                obs, reward, done, info = env.step([action] if action is not None else [])
                total += reward; lost += info["buildings_destroyed"]
            rewards.append(total); destroyed.append(lost); contained.append(info["contained"])
    model.train()
    return float(np.mean(rewards)), float(np.mean(destroyed)), float(np.mean(contained))


def save_checkpoint(model, optimizer, normalizer, episode, sampling_rng, metadata=None):
    """Save a complete, versioned training snapshot for exact continuation."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = os.path.join(CHECKPOINT_DIR, f"episode_{episode:04d}.pt")
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    payload = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        episode=episode,
        sampling_rng=sampling_rng,
        metadata=metadata,
    )
    torch.save(payload, checkpoint)
    torch.save(payload, latest)
    return checkpoint


def main():
    print(f"[training] N_EPISODES={N_EPISODES} LEARNING_RATE={LEARNING_RATE} "
          f"AUX_TARGET_LOSS_COEFF={AUX_TARGET_LOSS_COEFF} STATUS_EVERY={STATUS_EVERY} EVAL_EVERY={EVAL_EVERY} "
          f"RUN_TAG={RUN_TAG} CHECKPOINT_EVERY={CHECKPOINT_EVERY} MAX_DISPATCH_SLOTS={MAX_DISPATCH_SLOTS} TRACE_EVERY={TRACE_EVERY} "
          f"PROGRESS_ENABLED={PROGRESS_ENABLED} PROGRESS_START={PROGRESS_START} PROGRESS_WINDOW={PROGRESS_WINDOW}")
    device = get_device(force_cpu=FORCE_CPU)
    env = InfernoEnv(seed=BASE_SEED)
    obs = env.reset(seed=BASE_SEED)
    model = RelativeInfernoModel(len(obs["grid"]), len(SCALAR_KEYS), len(RESOURCE_TYPES), env.n_zones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    normalizer = RunningMeanStd()
    rng = np.random.default_rng(BASE_SEED + 1)
    active_world_fingerprint = world_fingerprint(GRID_STATIC_PATH, GRID_META_PATH)

    # Resume from latest checkpoint if available
    start_episode = 1
    latest_checkpoint = os.path.join(CHECKPOINT_DIR, "latest.pt")
    print(f"[training] Looking for latest checkpoint at {latest_checkpoint}")
    if os.path.exists(latest_checkpoint):
        print(f"[training] Resuming from {latest_checkpoint}")
        checkpoint = load_checkpoint(latest_checkpoint, map_location=device)
        saved_world = checkpoint_metadata(checkpoint).get("world_fingerprint")
        if saved_world is not None and saved_world != active_world_fingerprint:
            raise RuntimeError(
                "Checkpoint/world mismatch: the checkpoint was trained against "
                f"{saved_world}, but the configured world is {active_world_fingerprint}."
            )
        completed_episode = restore_training_state(
            checkpoint,
            model=model,
            optimizer=optimizer,
            normalizer=normalizer,
            sampling_rng=rng,
        )
        if completed_episode is not None:
            start_episode = completed_episode + 1
        elif PROGRESS_START is None:
            print("[training] Legacy weights loaded; set INFERNO_PROGRESS_START to continue numbering")
    if PROGRESS_START is not None:
        start_episode = PROGRESS_START

    pool = env._ignition_candidates
    holdout = np.array(list(VALIDATION_IGNITION_POINTS.values()), dtype=np.float32)
    distances = np.sqrt(((pool[:, None, :].astype(np.float32) - holdout[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    pool = pool[distances >= 30.0]
    logger = RunLogger(PROJECT_ROOT, RUN_TAG, {
        "episodes": N_EPISODES, "device": str(device), "traffic_mode": env.traffic_mode,
        "delay_config": env.delay_config, "max_dispatch_slots": MAX_DISPATCH_SLOTS,
        "learning_rate": LEARNING_RATE, "gamma": GAMMA, "scalar_keys": SCALAR_KEYS,
        "resource_counts": {rtype: len(env.resources[rtype]) for rtype in RESOURCE_TYPES},
        "world_fingerprint": active_world_fingerprint,
    }, trace_every=TRACE_EVERY, resume=start_episode > 1)
    print(f"[training] device={device} episodes={N_EPISODES} ignition_pool={len(pool)} run={RUN_TAG} start_episode={start_episode}")
    eval_scenarios = ["anchor", *VALIDATION_IGNITION_POINTS.keys()]
    progress = EpisodeProgress(
        n_episodes_start=start_episode,
        n_episodes=N_EPISODES,
        run_dir=logger.run_dir,
        run_tag=RUN_TAG,
        env_cfg={
            "device": str(device),
            "traffic_mode": env.traffic_mode,
            "max_dispatch_slots": MAX_DISPATCH_SLOTS,
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
        },
        rolling_window=PROGRESS_WINDOW,
        eval_scenarios=eval_scenarios,
        enable=PROGRESS_ENABLED,
    )
    print(f"[training] progress={progress.status_file} live={progress.is_live} window={PROGRESS_WINDOW}", flush=True)
    try:
        with progress:
            for episode in range(start_episode, N_EPISODES + 1):
                point = tuple(int(x) for x in pool[rng.integers(len(pool))])
                steps, reward, destroyed, contained = collect_rollout(env, model, point, device, BASE_SEED + episode)
                losses = update_policy(model, optimizer, steps, device, normalizer)
                for tick, step in enumerate(steps):
                    logger.log_tick(episode, tick, point, step, device)
                summary = summarize_episode(
                    steps, episode, point, device, losses, seed=BASE_SEED + episode
                )
                logger.log_episode(summary)
                progress.tick(episode, summary)
                if episode % STATUS_EVERY == 0 or not progress.is_live:
                    print(f"[training @ {episode}] reward={reward:.1f} destroyed={destroyed} contained={contained} "
                          f"policy={losses[0]:.3f} aux={losses[3]:.3f} entropy={losses[4]:.3f}", flush=True)
                if episode % EVAL_EVERY == 0:
                    eval_rows = []
                    for name, eval_point in [("anchor", TRAINING_IGNITION_POINT), *VALIDATION_IGNITION_POINTS.items()]:
                        result = evaluate(model, env, eval_point, device)
                        row = {
                            "checkpoint_episode": episode, "scenario": name,
                            "ignition_row": eval_point[0], "ignition_col": eval_point[1],
                            "evaluation_seed": BASE_SEED, "avg_reward": result[0],
                            "avg_buildings_destroyed": result[1], "containment_rate": result[2],
                        }
                        logger.log_eval(row)
                        eval_rows.append(row)
                        print(f"  eval {name}: reward={result[0]:.1f} destroyed={result[1]:.1f} containment={result[2]:.0%}", flush=True)
                    progress.refresh_evals(eval_rows)
                if episode % CHECKPOINT_EVERY == 0:
                    checkpoint = save_checkpoint(
                        model,
                        optimizer,
                        normalizer,
                        episode,
                        rng,
                        metadata={
                            "world_fingerprint": active_world_fingerprint,
                            "run_tag": RUN_TAG,
                            "scalar_keys": list(SCALAR_KEYS),
                        },
                    )
                    logger.log_checkpoint(episode, checkpoint)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
