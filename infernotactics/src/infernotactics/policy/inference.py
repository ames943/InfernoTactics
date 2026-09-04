"""Shared model inference for trainers, evaluators, and API services."""

import torch

from .targets import resolve_relative_targets


DEFAULT_MAX_DISPATCH_SLOTS = 10


def forward_policy(model, obs, env, device):
    """Run one policy forward pass and return its exact candidate context."""
    grid, scalars = model.obs_to_tensors(obs, device=device)

    zones, features = resolve_relative_targets(env, obs)
    target_zones = torch.from_numpy(zones).unsqueeze(0).to(device)
    target_features = torch.from_numpy(features).unsqueeze(0).to(device)
    logits, value, classification = model(grid, scalars, target_zones, target_features)
    return logits, value, classification, zones, features
