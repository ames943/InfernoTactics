"""
Forward-pass shape/perf check for InfernoModel against a REAL InfernoEnv
observation -- NOT training. Confirms the CNN/MLP/classification/actor-
critic branches all line up (32 zones, 3 resource types, 4 classes, and the
grid's actual (H, W)) before any training-loop code gets written.

    python -m src.models.test_model_forward
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import InfernoEnv  # noqa: E402
from models.inferno_model import InfernoModel  # noqa: E402

N_TIMING_REPS = 20


def main():
    print("Building InfernoEnv (loads grid + road routing graph)...")
    env = InfernoEnv(seed=0)
    obs = env.reset(seed=0)

    n_grid_channels = obs["grid"].shape[0]
    print(f"\nObservation grid shape: {obs['grid'].shape}  dtype={obs['grid'].dtype}")
    print(f"Observation scalars: {dict(obs['scalars'])}")
    print(f"n_zones={env.n_zones}")

    model = InfernoModel(n_grid_channels=n_grid_channels)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {n_params:,} (trainable: {n_trainable:,})")
    for name, module in [("cnn", model.cnn), ("mlp", model.mlp),
                          ("classifier", model.classifier), ("actor_critic", model.actor_critic)]:
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:14s} {n:>12,} params")

    grid, scalars = InfernoModel.obs_to_tensors(obs)
    print(f"\nInput tensors: grid={tuple(grid.shape)}  scalars={tuple(scalars.shape)}")

    with torch.no_grad():
        action_logits, value, classification_logits = model(grid, scalars)  # warmup

        t0 = time.perf_counter()
        for _ in range(N_TIMING_REPS):
            action_logits, value, classification_logits = model(grid, scalars)
        elapsed = time.perf_counter() - t0

    print("\n=== Output shapes ===")
    print(f"action_logits['resource_type']: {tuple(action_logits['resource_type'].shape)}  "
          f"(expected (1, 3))")
    print(f"action_logits['zone']:          {tuple(action_logits['zone'].shape)}  "
          f"(expected (1, {env.n_zones}))")
    print(f"value:                          {tuple(value.shape)}  (expected (1, 1))")
    print(f"classification_logits:          {tuple(classification_logits.shape)}  "
          f"(expected (1, 4, {env.height}, {env.width}))")

    predicted_classes = classification_logits.argmax(dim=1)
    print(f"\nclassification argmax shape: {tuple(predicted_classes.shape)}, "
          f"unique predicted classes: {torch.unique(predicted_classes).tolist()}")

    print(f"\n=== Timing (CPU, batch size 1) ===")
    print(f"Forward pass: {elapsed / N_TIMING_REPS * 1000:.2f} ms/pass avg over {N_TIMING_REPS} reps "
          f"({N_TIMING_REPS / elapsed:.1f} passes/sec)")


if __name__ == "__main__":
    main()
