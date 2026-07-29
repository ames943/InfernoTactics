"""
Standalone smoke-test / example for inferno_best_model.pt.

Run from inside this folder:
    pip install -r requirements.txt
    python run_example.py

Loads the trained model, resets the environment on the real Palisades Fire
ignition point (Skull Rock trailhead), and runs one deterministic rollout,
printing the resource/zone dispatched each tick and the final outcome.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch

from env.inferno_env import InfernoEnv, TRAINING_IGNITION_POINT, RESOURCE_TYPES
from models.inferno_model import InfernoModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(THIS_DIR, "inferno_best_model.pt")


def main():
    env = InfernoEnv()
    obs = env.reset(ignition_point=TRAINING_IGNITION_POINT, scenario="single",
                     seed=0, use_real_weather=True)

    n_grid_channels = obs["grid"].shape[0]
    model = InfernoModel(n_grid_channels=n_grid_channels)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
    model.eval()

    total_reward = 0.0
    done = False
    tick = 0
    with torch.no_grad():
        while not done:
            grid_t, scalars_t = InfernoModel.obs_to_tensors(obs)
            action_logits, value, _ = model(grid_t, scalars_t)
            resource_idx = action_logits["resource_type"].argmax(dim=-1).item()
            zone_idx = action_logits["zone"].argmax(dim=-1).item()
            resource_type = RESOURCE_TYPES[resource_idx] if resource_idx < len(RESOURCE_TYPES) else None

            obs, reward, done, info = env.step((resource_type, zone_idx))
            total_reward += reward
            tick += 1
            if tick % 10 == 0 or done:
                print(f"tick {tick:4d}  action=({resource_type}, {zone_idx})  "
                      f"reward={reward:8.2f}  total={total_reward:10.2f}")

    print("\n--- Episode finished ---")
    print(f"ticks: {tick}")
    print(f"total_reward: {total_reward:.2f}")
    print(f"buildings_destroyed: {info.get('buildings_destroyed')}")
    print(f"contained: {info.get('contained')}")


if __name__ == "__main__":
    main()
