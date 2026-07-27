"""
eval_policy(): run a (possibly untrained) InfernoModel against InfernoEnv for
a fixed ignition point, with NO gradient updates and NO optimizer step
anywhere in this module. Reused throughout the (not-yet-written) training
loop to check whether a policy generalizes to held-out ignition points (see
env.inferno_env.VALIDATION_IGNITION_POINTS) rather than just memorizing the
one it trained on.

    python -m src.train.eval    # sanity-checks the plumbing on an UNTRAINED model
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import RESOURCE_TYPES  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402


def _select_action(action_logits, deterministic):
    """Decode one (resource_type, zone) index pair from a single (batch
    size 1) forward pass's logits -- argmax if deterministic, else sampled
    from the categorical the actor head defines. Matches ActorCritic's
    factorization (see actor_critic.py): the two heads are sampled
    independently, never combined into one flat distribution."""
    resource_logits = action_logits["resource_type"][0]
    zone_logits = action_logits["zone"][0]
    if deterministic:
        resource_idx = int(torch.argmax(resource_logits))
        zone_idx = int(torch.argmax(zone_logits))
    else:
        resource_idx = int(torch.distributions.Categorical(logits=resource_logits).sample())
        zone_idx = int(torch.distributions.Categorical(logits=zone_logits).sample())
    return resource_idx, zone_idx


def eval_policy(model, env, ignition_point=None, ignition_points=None, n_episodes=5, use_real_weather=True,
                 deterministic=False, seed=0, device=None, scalars_fn=None):
    """Run `model` against `env` for n_episodes, every episode starting from
    the same fixed ignition point(s) (e.g. env.inferno_env.TRAINING_IGNITION_POINT
    or one of VALIDATION_IGNITION_POINTS, via `ignition_point`; or
    env.inferno_env.MULTI_IGNITION_TRAINING_SCENARIO via `ignition_points` --
    mutually exclusive, see InfernoEnv.reset()) -- no gradient updates,
    model.eval() for the duration (restored to its prior mode on return).
    Only the env's per-episode RNG (weather-independent; wind/humidity come
    from the real series when use_real_weather=True) varies episode to
    episode.

    On an UNTRAINED (randomly initialized) model this returns real but
    meaningless numbers -- the point at this stage is just confirming the
    env<->model<->action-decoding plumbing runs end to end, before any
    training exists to actually evaluate.

    Returns a dict: avg_reward, avg_buildings_destroyed, avg_buildings_saved
    (total building cells in the grid minus avg destroyed -- a coarse
    metric, since only a small fraction of the grid's buildings are ever
    within reach of any single localized fire episode, so this number is
    dominated by grid geometry rather than policy quality; avg_buildings_destroyed
    is the more informative of the two for comparing runs), containment_rate
    (fraction of episodes that ended by the fire actually being contained,
    i.e. Threat/Blaze both hit zero, rather than hitting MAX_TICKS), and the
    raw per-episode reward list.

    scalars_fn: optional callable(obs["scalars"] dict) -> np.float32 vector,
    overriding the default env.inferno_env.flatten_scalars() encoding. Added
    for train_actor_critic_multi_v2.py's scenario-one-hot MLP input (see that
    file) without changing this function's behavior for any existing caller
    that omits it (heuristic_policy.py, train_actor_critic.py, this module's
    own main()).
    """
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)

    episode_rewards = []
    episode_buildings_destroyed = []
    episode_contained = []

    with torch.no_grad():
        for ep in range(n_episodes):
            obs = env.reset(ignition_point=ignition_point, ignition_points=ignition_points,
                             seed=seed + ep, use_real_weather=use_real_weather)
            total_reward = 0.0
            total_destroyed = 0
            done = False
            contained = False
            while not done:
                grid, scalars = InfernoModel.obs_to_tensors(obs, device=device)
                if scalars_fn is not None:
                    scalars = torch.from_numpy(scalars_fn(obs["scalars"])).unsqueeze(0)
                    if device is not None:
                        scalars = scalars.to(device)
                action_logits, value, _classification_logits = model(grid, scalars)
                resource_idx, zone_idx = _select_action(action_logits, deterministic)
                action = (RESOURCE_TYPES[resource_idx], zone_idx)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                total_destroyed += info["buildings_destroyed"]
                contained = info["contained"]

            episode_rewards.append(total_reward)
            episode_buildings_destroyed.append(total_destroyed)
            episode_contained.append(contained)

    if was_training:
        model.train()

    n_building_cells = int(env._building_mask.sum())
    avg_destroyed = sum(episode_buildings_destroyed) / n_episodes
    return {
        "n_episodes": n_episodes,
        "ignition_point": ignition_point,
        "avg_reward": sum(episode_rewards) / n_episodes,
        "avg_buildings_destroyed": avg_destroyed,
        "avg_buildings_saved": n_building_cells - avg_destroyed,
        "containment_rate": sum(episode_contained) / n_episodes,
        "episode_rewards": episode_rewards,
    }


def main():
    from env.inferno_env import TRAINING_IGNITION_POINT, VALIDATION_IGNITION_POINTS, InfernoEnv

    print("Building InfernoEnv + untrained InfernoModel (sanity check ONLY -- no training)...")
    env = InfernoEnv(seed=0)
    obs = env.reset(seed=0)
    model = InfernoModel(n_grid_channels=obs["grid"].shape[0])

    val_name, val_point = next(iter(VALIDATION_IGNITION_POINTS.items()))

    for label, point in [("TRAINING_IGNITION_POINT (Palisades Highlands)", TRAINING_IGNITION_POINT),
                         (f"VALIDATION_IGNITION_POINTS['{val_name}']", val_point)]:
        print(f"\n=== eval_policy on {label} = {point} ===")
        result = eval_policy(model, env, ignition_point=point, n_episodes=3)
        for k, v in result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
