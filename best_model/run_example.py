"""
Standalone smoke-test / example for inferno_best_model.pt (v8, fire-relative
action space).

Run from inside this folder:
    pip install -r requirements.txt
    python run_example.py

Loads the trained model, resets the environment on the real Palisades Fire
ignition point (Skull Rock trailhead), and runs one deterministic rollout,
printing the resource/semantic-target dispatched each tick and the final
outcome.

Unlike the previous (pre-v8) checkpoint, the policy here never emits an
absolute zone id directly. Every tick, `resolve_relative_targets()` looks at
the CURRENT fire state and resolves a small set of semantic candidates
(active_fire, downwind_fire_front, adjacent_fuel, threatened_population,
nearest_reachable_fire, noop) to concrete zones; the model only chooses
among those candidates. See README.md for why this exists and what it does
and does not fix.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch

from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES
from models.relative_model import RelativeInfernoModel
from train.relative_actions import TARGET_TYPES, decode_action, resolve_relative_targets

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(THIS_DIR, "inferno_best_model.pt")


def main():
    env = InfernoEnv()
    obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, scenario="single",
                     seed=0, use_real_weather=True)

    from env.inferno_env import SCALAR_KEYS
    model = RelativeInfernoModel(
        n_grid_channels=obs["grid"].shape[0],
        n_scalars=len(SCALAR_KEYS),
        n_resources=len(RESOURCE_TYPES),
        n_zones=env.n_zones,
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True))
    model.eval()

    total_reward = 0.0
    done = False
    tick = 0
    with torch.no_grad():
        while not done:
            grid_t, scalars_t = RelativeInfernoModel.obs_to_tensors(obs)
            target_zones, target_features = resolve_relative_targets(env, obs)
            zones_t = torch.from_numpy(target_zones).unsqueeze(0)
            features_t = torch.from_numpy(target_features).unsqueeze(0)
            logits, value, _ = model(grid_t, scalars_t, zones_t, features_t)

            resource_logits = logits["resource_type"][0].clone()
            available = torch.tensor(
                [obs["scalars"][f"{r}_available"] > 0 for r in RESOURCE_TYPES], dtype=torch.bool
            )
            resource_logits[~available] = -1e9
            resource_idx = int(torch.argmax(resource_logits)) if bool(available.any()) else 0
            target_idx = int(torch.argmax(logits["target"][0, resource_idx]))
            action = decode_action(resource_idx, target_idx, target_zones)

            obs, reward, done, info = env.step(action)
            total_reward += reward
            tick += 1
            label = f"({RESOURCE_TYPES[resource_idx]}, {TARGET_TYPES[target_idx]})" if action else "noop"
            if tick % 10 == 0 or done:
                print(f"tick {tick:4d}  action={label:38s}  reward={reward:8.2f}  total={total_reward:10.2f}")

    print("\n--- Episode finished ---")
    print(f"ticks: {tick}")
    print(f"total_reward: {total_reward:.2f}")
    print(f"buildings_destroyed: {info.get('buildings_destroyed')}")
    print(f"contained: {info.get('contained')}")


if __name__ == "__main__":
    main()
