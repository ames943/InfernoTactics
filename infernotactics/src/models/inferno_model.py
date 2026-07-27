"""
Ties the CNN branch, MLP branch, classification head, and actor-critic head
into one model:

    action_logits, value, classification_logits = model(grid, scalars)

forward() takes already-batched tensors (the shape a training loop will
actually use). InfernoModel.obs_to_tensors(obs) bridges a single raw
InfernoEnv observation dict (as returned by reset()/step()) into that
batched form, for exactly the ad hoc forward-pass check in
test_model_forward.py.

NO TRAINING HERE -- forward pass only, verified for correct shapes.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.inferno_env import SCALAR_KEYS, flatten_scalars  # noqa: E402
from models.actor_critic import ActorCritic, N_RESOURCE_TYPES, N_ZONES  # noqa: E402
from models.classification_head import ClassificationHead  # noqa: E402
from models.cnn_branch import PER_CELL_CHANNELS, POOLED_DIM, CNNBranch  # noqa: E402
from models.mlp_branch import OUTPUT_DIM as MLP_OUTPUT_DIM  # noqa: E402
from models.mlp_branch import MLPBranch  # noqa: E402


class InfernoModel(nn.Module):
    def __init__(self, n_grid_channels, n_scalars=len(SCALAR_KEYS),
                 n_resource_types=N_RESOURCE_TYPES, n_zones=N_ZONES, n_value_heads=1):
        super().__init__()
        self.cnn = CNNBranch(in_channels=n_grid_channels)
        self.mlp = MLPBranch(in_features=n_scalars)
        fused_dim = POOLED_DIM + MLP_OUTPUT_DIM
        self.actor_critic = ActorCritic(in_features=fused_dim, n_resource_types=n_resource_types, n_zones=n_zones,
                                         n_value_heads=n_value_heads)
        self.classifier = ClassificationHead(in_channels=PER_CELL_CHANNELS)

    def forward(self, grid, scalars, value_head_idx=0):
        """grid: (B, n_grid_channels, H, W), scalars: (B, n_scalars) ->
        (action_logits dict, value (B, 1), classification_logits (B, 4, H, W))
        value_head_idx: see ActorCritic -- ignored unless the model was built
        with n_value_heads>1 (train_actor_critic_multi_v3.py's separate-
        value-head-per-scenario experiment)."""
        per_cell_features, pooled_cnn = self.cnn(grid)
        mlp_features = self.mlp(scalars)
        fused = torch.cat([pooled_cnn, mlp_features], dim=1)
        action_logits, value = self.actor_critic(fused, value_head_idx=value_head_idx)
        classification_logits = self.classifier(per_cell_features)
        return action_logits, value, classification_logits

    @staticmethod
    def obs_to_tensors(obs, device=None):
        """Bridge a single raw InfernoEnv observation dict (numpy, no batch
        dim) into batched (batch size 1) tensors for forward()."""
        grid = torch.from_numpy(np.ascontiguousarray(obs["grid"])).unsqueeze(0)
        scalars = torch.from_numpy(flatten_scalars(obs["scalars"])).unsqueeze(0)
        if device is not None:
            grid = grid.to(device)
            scalars = scalars.to(device)
        return grid, scalars
