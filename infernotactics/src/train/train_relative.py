"""V8 training loop: randomized ignitions with fire-relative actions.

This deliberately uses the proven pre-v4 Monte-Carlo loop.  The experiment
changes the action representation and training distribution, not the unstable
GAE/PPO rewrite.
"""

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import (  # noqa: E402
    RESOURCE_TYPES,
    SCALAR_KEYS,
    VALIDATION_IGNITION_POINTS,
    InfernoEnv,
    TRAINING_IGNITION_POINT,
    flatten_scalars,
)
from data_pipeline.config import PROJECT_ROOT  # noqa: E402
from models.classification_head import fire_state_to_class  # noqa: E402
from models.relative_model import RelativeInfernoModel  # noqa: E402
from train.relative_actions import (  # noqa: E402
    N_TARGET_TYPES,
    TARGET_TYPES,
    decode_action,
    resolve_relative_targets,
)
from train.train_actor_critic import (  # noqa: E402
    CLASSIFICATION_LOSS_COEFF,
    ENTROPY_COEFF,
    GAMMA,
    GRAD_CLIP_NORM,
    RunningMeanStd,
    compute_returns,
    get_device,
)


N_EPISODES = int(os.environ.get("INFERNO_N_EPISODES", 2000))
BASE_SEED = 8200
LEARNING_RATE = float(os.environ.get("INFERNO_V8_LR", 3e-4))
AUX_TARGET_LOSS_COEFF = float(os.environ.get("INFERNO_V8_AUX_COEFF", 0.05))
STATUS_EVERY = 20
EVAL_EVERY = int(os.environ.get("INFERNO_V8_EVAL_EVERY", 50))
RUN_TAG = os.environ.get("INFERNO_RUN_TAG", "relative_v8")
CHECKPOINT_EVERY = int(os.environ.get("INFERNO_V8_CHECKPOINT_EVERY", 2))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "models", f"checkpoints_{RUN_TAG}")
MAX_DISPATCH_SLOTS = int(os.environ.get("INFERNO_MAX_DISPATCH_SLOTS", 10))


def _tensor_targets(env, obs, device):
    zones, features = resolve_relative_targets(env, obs)
    return (
        torch.from_numpy(zones).unsqueeze(0).to(device),
        torch.from_numpy(features).unsqueeze(0).to(device),
        zones,
    )


def _resource_mask(obs):
    return torch.tensor([
        obs["scalars"][f"{rtype}_available"] > 0 for rtype in RESOURCE_TYPES
    ], dtype=torch.bool)


def _forward(model, obs, env, device):
    grid, scalars = RelativeInfernoModel.obs_to_tensors(obs, device=device) if hasattr(RelativeInfernoModel, "obs_to_tensors") else (
        torch.from_numpy(np.ascontiguousarray(obs["grid"])).unsqueeze(0).to(device),
        torch.from_numpy(flatten_scalars(obs["scalars"])).unsqueeze(0).to(device),
    )
    target_zones, target_features, raw_zones = _tensor_targets(env, obs, device)
    logits, value, classification = model(grid, scalars, target_zones, target_features)
    return logits, value, classification, raw_zones


def collect_rollout(env, model, ignition_point, device, seed):
    obs = env.reset(ignition_point=ignition_point, use_real_weather=True, seed=seed)
    steps, total_reward, buildings_destroyed = [], 0.0, 0
    done = False
    info = None
    with torch.no_grad():
        while not done:
            tick_obs = obs
            tick_actions = []
            local_available = {
                rtype: int(tick_obs["scalars"][f"{rtype}_available"])
                for rtype in RESOURCE_TYPES
            }
            for _ in range(MAX_DISPATCH_SLOTS):
                logits, _value, _classification, raw_zones = _forward(model, tick_obs, env, device)
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
                if action is None:
                    break
                local_available[RESOURCE_TYPES[resource_idx]] -= 1
                tick_actions.append({
                    "resource_idx": resource_idx,
                    "target_idx": target_idx,
                    "target_zones": raw_zones,
                })

            actions = [
                decode_action(a["resource_idx"], a["target_idx"], a["target_zones"])
                for a in tick_actions
            ]
            next_obs, reward, done, info = env.step([a for a in actions if a is not None])
            steps.append({
                "grid": tick_obs["grid"],
                "scalars": flatten_scalars(tick_obs["scalars"]),
                "actions": tick_actions,
                "fire_state_target": fire_state_to_class(torch.from_numpy(tick_obs["grid"][-1]).long()),
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
    sums = [0.0] * 6
    for step, g_t in zip(steps, returns):
        grid = torch.from_numpy(step["grid"]).unsqueeze(0).to(device)
        scalars = torch.from_numpy(step["scalars"]).unsqueeze(0).to(device)
        target_zones_np = step["actions"][0]["target_zones"] if step["actions"] else np.full(
            (len(RESOURCE_TYPES), N_TARGET_TYPES), -1, dtype=np.int64
        )
        zones = torch.from_numpy(target_zones_np).unsqueeze(0).to(device)
        features_np = resolve_relative_targets_from_state(step["grid"], step["scalars"], target_zones_np)
        features = torch.from_numpy(features_np).unsqueeze(0).to(device)
        logits, value, classification = model(grid, scalars, zones, features)
        resource_dist = Categorical(logits=logits["resource_type"][0])
        log_prob = torch.tensor(0.0, device=device)
        entropy = resource_dist.entropy()
        aux_loss = torch.tensor(0.0, device=device)
        for action_data in step["actions"]:
            resource_idx = action_data["resource_idx"]
            target_idx = action_data["target_idx"]
            target_dist = Categorical(logits=logits["target"][0, resource_idx])
            log_prob = log_prob + resource_dist.log_prob(torch.tensor(resource_idx, device=device)) \
                + target_dist.log_prob(torch.tensor(target_idx, device=device))
            entropy = entropy + target_dist.entropy()
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
        sums[3] += aux_loss.item(); sums[4] += entropy.item(); sums[5] += target_dist.entropy().item()
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
    out = np.zeros((len(RESOURCE_TYPES), N_TARGET_TYPES, 10), dtype=np.float32)
    for r in range(len(RESOURCE_TYPES)):
        for t, zone in enumerate(target_zones[r]):
            if zone < 0:
                continue
            # Zone indices are stable only for feature reconstruction; the
            # action itself remains relative because the resolver selected it.
            zr, zc = divmod(int(zone), 8)
            r0, r1 = zr * 80, min((zr + 1) * 80, rows)
            c0, c1 = zc * 80, min((zc + 1) * 80, cols)
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
                logits, _v, _c, zones = _forward(model, obs, env, device)
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


def save_checkpoint(model, episode):
    """Save frequent lightweight model snapshots for crash recovery/inspection."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint = os.path.join(CHECKPOINT_DIR, f"episode_{episode:04d}.pt")
    latest = os.path.join(CHECKPOINT_DIR, "latest.pt")
    torch.save(model.state_dict(), checkpoint)
    torch.save(model.state_dict(), latest)


def main():
    device = get_device()
    env = InfernoEnv(seed=BASE_SEED)
    obs = env.reset(seed=BASE_SEED)
    model = RelativeInfernoModel(len(obs["grid"]), len(SCALAR_KEYS), len(RESOURCE_TYPES), env.n_zones).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    normalizer = RunningMeanStd()
    pool = env._ignition_candidates
    holdout = np.array(list(VALIDATION_IGNITION_POINTS.values()), dtype=np.float32)
    distances = np.sqrt(((pool[:, None, :].astype(np.float32) - holdout[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    pool = pool[distances >= 30.0]
    print(f"[relative_v8] device={device} episodes={N_EPISODES} ignition_pool={len(pool)}")
    rng = np.random.default_rng(BASE_SEED + 1)
    for episode in range(1, N_EPISODES + 1):
        point = tuple(int(x) for x in pool[rng.integers(len(pool))])
        steps, reward, destroyed, contained = collect_rollout(env, model, point, device, BASE_SEED + episode)
        losses = update_policy(model, optimizer, steps, device, normalizer)
        if episode % STATUS_EVERY == 0:
            print(f"[relative_v8 @ {episode}] reward={reward:.1f} destroyed={destroyed} contained={contained} "
                  f"policy={losses[0]:.3f} aux={losses[3]:.3f} entropy={losses[4]:.3f}", flush=True)
        if episode % EVAL_EVERY == 0:
            for name, point in [("anchor", TRAINING_IGNITION_POINT), *VALIDATION_IGNITION_POINTS.items()]:
                result = evaluate(model, env, point, device)
                print(f"  eval {name}: reward={result[0]:.1f} destroyed={result[1]:.1f} containment={result[2]:.0%}", flush=True)
        if episode % CHECKPOINT_EVERY == 0:
            save_checkpoint(model, episode)


if __name__ == "__main__":
    main()
