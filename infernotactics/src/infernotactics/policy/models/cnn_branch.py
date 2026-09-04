"""
CNN branch: consumes the environment grid observation (8 static terrain/
infrastructure/population layers + current fire state, one channel per
layer) and produces two things:
  - a full-resolution per-cell feature map, for the classification head
    (needs per-cell resolution -- this path never downsamples)
  - a pooled, fixed-size feature vector, for fusion with the MLP branch

Kept intentionally small/shallow: this is a course project fusing 4 required
ML concepts, not a bid for state-of-the-art wildfire modeling. Depth/width
can be revisited once real memory/speed numbers are in from
test_model_forward.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from infernotactics.domain.zones import build_zone_boundaries

PER_CELL_CHANNELS = 32  # channels in the full-resolution feature map (-> classification head)
POOLED_DIM = 128        # size of the flattened/pooled feature vector (-> fusion with MLP branch)
ADAPTIVE_POOL_SIZE = (4, 4)
def zone_boundaries(height, width):
    """Return the same canonical 4x8 boundaries used by the environment."""
    return [
        (zone.row_start, zone.row_stop, zone.col_start, zone.col_stop)
        for zone in build_zone_boundaries(height, width)
    ]


class CNNBranch(nn.Module):
    def __init__(self, in_channels, per_cell_channels=PER_CELL_CHANNELS, pooled_dim=POOLED_DIM,
                 adaptive_pool_size=ADAPTIVE_POOL_SIZE):
        super().__init__()
        self.adaptive_pool_size = adaptive_pool_size

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
        # AdaptiveAvgPool2d itself is applied separately in forward() (not
        # part of this Sequential) since it needs a dynamic zero-pad step
        # first -- see forward()'s comment.
        self.pool_downsample = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(per_cell_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d(adaptive_pool_size)
        self.pool_fc = nn.Sequential(
            nn.Linear(64 * adaptive_pool_size[0] * adaptive_pool_size[1], pooled_dim),
            nn.ReLU(inplace=True),
        )
        self._cached_boundaries = None  # (height, width) -> list[(r0,r1,c0,c1)], computed once per shape seen

    def _zone_boundaries_for(self, height, width):
        if self._cached_boundaries is None or self._cached_boundaries[0] != (height, width):
            self._cached_boundaries = ((height, width), zone_boundaries(height, width))
        return self._cached_boundaries[1]

    def forward(self, grid):
        """grid: (B, in_channels, H, W) -> (
            per_cell_features: (B, per_cell_channels, H, W),
            pooled: (B, pooled_dim),
            zone_pooled: (B, n_zones, per_cell_channels) -- per_cell_features
                average-pooled within each macro-zone rectangle (zero new
                conv params, just a spatial mean per zone), for the actor's
                zone head to see actual per-zone spatial detail instead of
                only the single globally-pooled vector.
        )"""
        per_cell_features = self.per_cell_trunk(grid)

        height, width = per_cell_features.shape[-2:]
        boundaries = self._zone_boundaries_for(height, width)
        zone_pooled = torch.stack(
            [per_cell_features[:, :, r0:r1, c0:c1].mean(dim=(2, 3)) for r0, r1, c0, c1 in boundaries],
            dim=1,
        )  # (B, n_zones, per_cell_channels)

        downsampled = self.pool_downsample(per_cell_features)

        # AdaptiveAvgPool2d on MPS requires the input's H and W to each be
        # evenly divisible by the output size (here 4, 4); CPU has no such
        # restriction. E.g. on the current 316x595 grid, downsampled is
        # (79, 148): 148 divides evenly but 79 doesn't (79 is prime).
        # Zero-pad up to the next multiple of 4 on each spatial dim (never
        # crop, so no feature data is discarded -- at most 3 padding rows/
        # cols, a negligible boundary effect on ~20x37-cell pooling bins).
        # Computed dynamically from the actual tensor shape (not hardcoded
        # to today's grid resolution) and applied identically on both
        # devices, so CPU and MPS now do the exact same computation instead
        # of silently differing.
        h, w = downsampled.shape[-2:]
        pad_h = (-h) % self.adaptive_pool_size[0]
        pad_w = (-w) % self.adaptive_pool_size[1]
        if pad_h or pad_w:
            downsampled = F.pad(downsampled, (0, pad_w, 0, pad_h))

        pooled = self.adaptive_pool(downsampled)
        pooled = pooled.flatten(1)
        pooled = self.pool_fc(pooled)
        return per_cell_features, pooled, zone_pooled
