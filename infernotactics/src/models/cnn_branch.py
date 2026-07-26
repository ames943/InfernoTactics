"""
CNN branch: consumes the InfernoEnv 'grid' observation (7 static terrain/
infrastructure layers + current fire state, one channel per layer) and
produces two things:
  - a full-resolution per-cell feature map, for the classification head
    (needs per-cell resolution -- this path never downsamples)
  - a pooled, fixed-size feature vector, for fusion with the MLP branch

Kept intentionally small/shallow: this is a course project fusing 4 required
ML concepts, not a bid for state-of-the-art wildfire modeling. Depth/width
can be revisited once real memory/speed numbers are in from
test_model_forward.py.
"""

import torch.nn as nn

PER_CELL_CHANNELS = 32  # channels in the full-resolution feature map (-> classification head)
POOLED_DIM = 128        # size of the flattened/pooled feature vector (-> fusion with MLP branch)


class CNNBranch(nn.Module):
    def __init__(self, in_channels, per_cell_channels=PER_CELL_CHANNELS, pooled_dim=POOLED_DIM):
        super().__init__()

        # Full-resolution path: stride-1, same-padding convs only, so the
        # output feature map matches the input grid's (H, W) exactly.
        self.per_cell_trunk = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, per_cell_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Pooled path: branches off the per-cell features, downsamples with
        # a couple of maxpool+conv steps (188K input cells is too much to
        # flatten directly), then adaptive-pools to a small fixed spatial
        # size regardless of input H/W so this works at any grid resolution.
        self.pool_trunk = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(per_cell_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.pool_fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, pooled_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, grid):
        """grid: (B, in_channels, H, W) ->
        (per_cell_features: (B, per_cell_channels, H, W), pooled: (B, pooled_dim))"""
        per_cell_features = self.per_cell_trunk(grid)
        pooled = self.pool_trunk(per_cell_features)
        pooled = pooled.flatten(1)
        pooled = self.pool_fc(pooled)
        return per_cell_features, pooled
