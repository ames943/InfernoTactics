"""Small, inspectable PPO on present-only candidate observations.

Candidate features use fixed scenario bounds/capacities, never evaluation-set
statistics. Horizon and domain-exit censoring bootstrap the final observation;
GAE never propagates into the reset episode. Checkpoints are not compatible with
the historical course models and are loaded with tensor-only deserialization.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .contracts import Action, SCHEMA_VERSION, Scenario
from .fire import ENGINE_VERSION
from .world import World

CHECKPOINT_SCHEMA_VERSION = 1
OPERATIONS = ("wait", "water", "line", "return")
STATUSES = ("available", "mobilizing", "traveling", "working", "returning", "refilling")
FEATURE_NAMES = (
    *(f"operation_{name}" for name in OPERATIONS),
    "unit_x", "unit_y", "unit_supply_fraction",
    *(f"unit_status_{name}" for name in STATUSES),
    "target_x", "target_y", "target_fireline_intensity_fraction",
    "target_asset_fraction", "feasible_eta_fraction", "elapsed_time_fraction",
)
# The intensity transform I/(I+scale) is bounded without fitting observations.
INTENSITY_SCALE_KW_M = 10000.0
HYPERPARAMETERS = {
    "learning_rate": 3e-4, "gamma": .99, "gae_lambda": .95,
    "clip": .2, "value_coefficient": .5, "entropy_coefficient": .01,
    "max_gradient_norm": .5, "transitions_per_update": 256,
    "epochs": 4, "minibatch_size": 64, "hidden_sizes": [64, 64],
    "activation": "tanh", "critic_pooling": "masked_mean",
    "normalize_advantages": True, "adam_eps": 1e-8,
    "intensity_scale_kw_m": INTENSITY_SCALE_KW_M,
    "reward_burned_weight": 1.0, "reward_assets_weight": 1.0,
    "reward_resource_time_weight": .01,
    "bootstrap_truncations": True, "torch_num_threads": 1,
}
TRAINING_SEED_MIN = 0
TRAINING_SEED_MAX = 999


def _json_value(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {"dtype": array.dtype.str, "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _digest(value) -> str:
    data = json.dumps(_json_value(value), sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def scenario_hashes(scenario: Scenario) -> dict:
    """Hash actual content, including in-memory scenarios without source files."""
    content = {field.name: getattr(scenario, field.name) for field in fields(scenario)}
    return {"scenario": _digest(content), "landscape": _digest(scenario.landscape),
            "weather": _digest(scenario.weather), "roads": _digest(scenario.roads),
            "resources": _digest(scenario.resources),
            "source_hashes": _json_value(scenario.provenance.get("source_hashes", {}))}


def _seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed <= 999:
        raise ValueError("training seeds must be integers in 0..999; evaluation seeds are reserved")
    return int(seed)


def checkpoint_metadata(scenario: Scenario, seed: int, training_seeds: list[int]) -> dict:
    """Public checkpoint fixture/provenance builder; no executable objects."""
    seed = _seed(seed)
    seeds = [_seed(item) for item in training_seeds]
    if not seeds or len(set(seeds)) != len(seeds) or seeds[0] != seed:
        raise ValueError("training_seeds must be unique and start with the optimizer seed")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION, "scenario_schema_version": SCHEMA_VERSION,
        "backend": "pyretechnics", "engine_version": ENGINE_VERSION,
        "torch_version": str(torch.__version__), "policy": "candidate_ppo",
        "feature_ordering": list(FEATURE_NAMES), "hyperparameters": dict(HYPERPARAMETERS),
        "input_hashes": scenario_hashes(scenario), "seed": seed,
        "training_seeds": seeds, "training_seed_range": [0, 999],
        "truncation_semantics": "bootstrap_final_observation_cut_gae_at_reset",
    }


def candidate_features(observation: dict, scenario: Scenario) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode exactly the public candidates and mask; never regenerate actions."""
    candidates = observation["candidates"]
    mask = np.asarray(observation["candidate_mask"], dtype=np.bool_)
    metadata = observation["candidate_metadata"]
    if not candidates or mask.shape != (len(candidates),) or len(metadata) != len(candidates):
        raise ValueError("candidate observation lengths are inconsistent")
    if candidates[0] != Action() or not mask[0]:
        raise ValueError("candidate observations must include feasible wait first")
    height, width = scenario.landscape["valid"].shape
    a, _, c, _, e, f = scenario.transform
    units = {unit["id"]: unit for unit in observation["units"]}
    total_assets = max(float(scenario.landscape["asset_value"][scenario.landscape["valid"]].sum()), 1.)
    result = np.zeros((len(candidates), len(FEATURE_NAMES)), dtype=np.float32)

    def xy(x, y):
        return (float(np.clip((x - c) / (a * width), 0., 1.)),
                float(np.clip((y - f) / (e * height), 0., 1.)))

    for index, (action, data) in enumerate(zip(candidates, metadata)):
        row = result[index]
        row[OPERATIONS.index(action.operation)] = 1.
        if action.unit_id is not None:
            unit = units[action.unit_id]
            row[4:6] = xy(unit["x"], unit["y"])
            capacity = float(unit.get("capacity_l", 0.))
            row[6] = np.clip(float(unit.get("supply_l", capacity)) / capacity, 0., 1.) if capacity else 0.
            row[7 + STATUSES.index(unit["status"])] = 1.
        if action.x is not None and action.y is not None:
            row[13:15] = xy(action.x, action.y)
            r, col = int(np.floor((action.y - f) / e)), int(np.floor((action.x - c) / a))
            if 0 <= r < height and 0 <= col < width:
                intensity = max(float(observation["fire"]["fireline_intensity"][r, col]), 0.)
                row[15] = intensity / (intensity + INTENSITY_SCALE_KW_M)
                row[16] = np.clip(float(observation["grid"]["asset_value"][r, col]) / total_assets, 0., 1.)
        eta = data.get("eta_s")
        row[17] = 0. if eta is None else np.clip(float(eta) / scenario.duration_s, 0., 1.)
        row[18] = np.clip(observation["time_s"] / scenario.duration_s, 0., 1.)
    if not np.isfinite(result).all():
        raise ValueError("candidate features must be finite")
    return torch.from_numpy(result), torch.from_numpy(mask.copy())


class ActorCritic(nn.Module):
    """Independent two-layer tanh actor and masked-mean pooled critic."""
    def __init__(self):
        super().__init__()
        size = len(FEATURE_NAMES)
        self.actor = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64),
                                   nn.Tanh(), nn.Linear(64, 1))
        self.critic = nn.Sequential(nn.Linear(size, 64), nn.Tanh(), nn.Linear(64, 64),
                                    nn.Tanh(), nn.Linear(64, 1))

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        if features.ndim not in (2, 3) or mask.shape != features.shape[:-1]:
            raise ValueError("feature and mask shapes do not match")
        if not mask.any(dim=-1).all():
            raise ValueError("each observation needs at least one feasible candidate")
        logits = self.actor(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        # Masked candidates/padding cannot change either action probabilities or value.
        pooled = features.masked_fill(~mask.unsqueeze(-1), 0.).sum(dim=-2)
        pooled = pooled / mask.sum(dim=-1, keepdim=True)
        return Categorical(logits=logits), self.critic(pooled).squeeze(-1)


def compute_gae(rewards, values, next_values, terminated, episode_ends,
                gamma: float = .99, gae_lambda: float = .95) -> tuple[np.ndarray, np.ndarray]:
    """Terminal transitions zero bootstrap; truncations bootstrap but cut traces.

    next_values contains the value of each actual next/final observation, not
    the value of a reset observation. A rollout boundary needs no special flag:
    its final residual bootstraps, while the uncollected future trace is zero.
    """
    rewards, values, next_values = [np.asarray(item, dtype=np.float64)
                                   for item in (rewards, values, next_values)]
    terminated, episode_ends = [np.asarray(item, dtype=np.bool_)
                               for item in (terminated, episode_ends)]
    if rewards.ndim != 1 or any(item.shape != rewards.shape for item in
                               (values, next_values, terminated, episode_ends)):
        raise ValueError("GAE inputs must be equally sized vectors")
    if np.any(terminated & ~episode_ends):
        raise ValueError("a terminal transition must also end its episode")
    advantages = np.empty_like(rewards)
    trace = 0.
    for index in range(len(rewards) - 1, -1, -1):
        delta = rewards[index] + gamma * next_values[index] * (not terminated[index]) - values[index]
        trace = delta + gamma * gae_lambda * (not episode_ends[index]) * trace
        advantages[index] = trace
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


class LearnedPolicy:
    def __init__(self, model: ActorCritic, metadata: dict):
        self.model = model.eval()
        self.metadata = metadata
        self.training_seeds = tuple(metadata["training_seeds"])

    def act(self, observation: dict, scenario: Scenario) -> Action:
        features, mask = candidate_features(observation, scenario)
        with torch.no_grad():
            distribution, _ = self.model(features, mask)
            index = int(distribution.logits.argmax().item())
        return observation["candidates"][index]


def load_policy(path, scenario: Scenario) -> LearnedPolicy:
    """Load only this exact backend/schema/features/scenario; never warm-start."""
    scenario.validate()
    try:
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("invalid or incompatible tensor-only policy checkpoint") from exc
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"metadata", "state_dict"}:
        raise ValueError("incompatible checkpoint: expected candidate PPO metadata and state_dict; legacy models unsupported")
    metadata = checkpoint["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("incompatible checkpoint metadata")
    try:
        expected = checkpoint_metadata(scenario, metadata["seed"], metadata["training_seeds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incompatible checkpoint training-seed metadata") from exc
    if metadata != expected:
        mismatches = sorted(key for key in set(metadata) | set(expected)
                            if metadata.get(key) != expected.get(key))
        raise ValueError(f"incompatible checkpoint metadata: {', '.join(mismatches)}")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("checkpoint state_dict must contain only tensors")
    model = ActorCritic()
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError) as exc:
        raise ValueError("incompatible checkpoint tensor shapes or keys") from exc
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("checkpoint contains nonfinite weights")
    torch.set_num_threads(HYPERPARAMETERS["torch_num_threads"])
    return LearnedPolicy(model, metadata)


def _batch(transitions, indices):
    width = max(len(transitions[index][0]) for index in indices)
    features = torch.zeros((len(indices), width, len(FEATURE_NAMES)))
    mask = torch.zeros((len(indices), width), dtype=torch.bool)
    for row, index in enumerate(indices):
        encoded, feasible = transitions[index][:2]
        features[row, :len(encoded)] = encoded
        mask[row, :len(encoded)] = feasible
    return features, mask


def _update(model, optimizer, transitions, advantages, returns):
    advantages = torch.as_tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    returns = torch.as_tensor(returns)
    actions = torch.tensor([item[2] for item in transitions], dtype=torch.long)
    old_log_probs = torch.tensor([item[3] for item in transitions])
    for _ in range(HYPERPARAMETERS["epochs"]):
        order = torch.randperm(len(transitions))
        for start in range(0, len(transitions), HYPERPARAMETERS["minibatch_size"]):
            indices = order[start:start + HYPERPARAMETERS["minibatch_size"]]
            features, mask = _batch(transitions, indices.tolist())
            distribution, values = model(features, mask)
            ratio = (distribution.log_prob(actions[indices]) - old_log_probs[indices]).exp()
            unclipped = ratio * advantages[indices]
            clipped = ratio.clamp(1. - HYPERPARAMETERS["clip"], 1. + HYPERPARAMETERS["clip"]) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (values - returns[indices]).square().mean()
            loss = (policy_loss + HYPERPARAMETERS["value_coefficient"] * value_loss
                    - HYPERPARAMETERS["entropy_coefficient"] * distribution.entropy().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMETERS["max_gradient_norm"])
            optimizer.step()


def train(scenario: Scenario, steps: int, seed: int, out) -> dict:
    """Collect exactly steps minute decisions, update PPO, and save a checkpoint.

    Episodes cycle reproducibly through [seed, ..., 999, 0, ...]. Seeds 1000+
    are never used for training. Each episode uses the scenario's declared
    ignition, and the World owns all incremental physical utility accounting.
    """
    scenario.validate()
    seed = _seed(seed)
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    steps = int(steps)
    ignition = scenario.provenance.get("ignition", {})
    if ignition.get("longitude") is None or ignition.get("latitude") is None:
        raise ValueError("training scenario must declare ignition longitude and latitude")
    out = Path(out)
    if out.exists():
        raise ValueError(f"checkpoint output already exists: {out}")
    torch.set_num_threads(HYPERPARAMETERS["torch_num_threads"])
    training_seeds = []
    episode = 0

    def reset():
        nonlocal episode
        world_seed = (seed + episode) % 1000
        episode += 1
        if world_seed not in training_seeds:
            training_seeds.append(world_seed)
        world = World(scenario, world_seed)
        world.ignite(ignition["longitude"], ignition["latitude"])
        return world, world.observe()

    updates = 0
    reward_total = 0.
    # Seed CPU sampling without replacing the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETERS["learning_rate"],
                                     eps=HYPERPARAMETERS["adam_eps"])
        world, observation = reset()
        features, mask = candidate_features(observation, scenario)
        for offset in range(0, steps, HYPERPARAMETERS["transitions_per_update"]):
            transitions, rewards, values, next_values, terminated, ends = [], [], [], [], [], []
            count = min(HYPERPARAMETERS["transitions_per_update"], steps - offset)
            for local_step in range(count):
                with torch.no_grad():
                    distribution, value = model(features, mask)
                    action_index = distribution.sample()
                    log_prob = distribution.log_prob(action_index)
                following, reward, done, info = world.step([observation["candidates"][int(action_index)]])
                next_features, next_mask = candidate_features(following, scenario)
                terminal = bool(info.get("terminated", False))
                with torch.no_grad():
                    _, next_value = model(next_features, next_mask)
                transitions.append((features, mask, int(action_index), float(log_prob)))
                rewards.append(reward)
                values.append(float(value))
                next_values.append(float(next_value))
                terminated.append(terminal)
                ends.append(bool(done))
                reward_total += reward
                observation, features, mask = following, next_features, next_mask
                if done and offset + local_step + 1 < steps:
                    world, observation = reset()
                    features, mask = candidate_features(observation, scenario)
            advantages, returns = compute_gae(rewards, values, next_values, terminated, ends,
                                              HYPERPARAMETERS["gamma"], HYPERPARAMETERS["gae_lambda"])
            _update(model, optimizer, transitions, advantages, returns)
            updates += 1
        metadata = checkpoint_metadata(scenario, seed, training_seeds)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "state_dict": model.state_dict()}, out)
    return {"checkpoint": str(out), "steps": steps, "updates": updates,
            "training_seeds": training_seeds, "seed": seed, "reward": float(reward_total),
            "metadata": metadata, "experimental": True}
