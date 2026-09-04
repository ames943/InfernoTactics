import numpy as np
import pytest
import torch

from infernotactics.simulation.fire_sim import BLAZE, SAFE
from infernotactics.operations.environment import TRENCH_BREAK_HOLD_BONUS, InfernoEnv
from infernotactics.domain.weather import (
    meteorological_wind_to_cartesian,
    meteorological_wind_to_grid,
)
from infernotactics.domain.zones import build_zone_boundaries, zone_id_for_cell
from infernotactics.policy.models import RelativeInfernoModel
from infernotactics.policy.targets import NOOP_TARGET_INDEX, N_TARGET_TYPES, TARGET_FEATURE_DIM


def test_canonical_zones_cover_grid_without_overlap():
    height, width = 316, 595
    zones = build_zone_boundaries(height, width)
    coverage = np.zeros((height, width), dtype=np.uint8)
    for zone in zones:
        coverage[zone.row_start:zone.row_stop, zone.col_start:zone.col_stop] += 1

    assert len(zones) == 32
    assert np.all(coverage == 1)
    assert zones[0].row_range == (0, 79)
    assert zones[0].col_range == (0, 74)
    assert zones[-1].row_range == (237, 316)
    assert zones[-1].col_range == (520, 595)
    assert zone_id_for_cell(315, 594, height, width) == 31


@pytest.mark.parametrize(
    ("direction", "cartesian", "grid"),
    [
        (0.0, (0.0, -1.0), (1.0, 0.0)),
        (90.0, (-1.0, 0.0), (0.0, -1.0)),
        (180.0, (0.0, 1.0), (-1.0, 0.0)),
        (270.0, (1.0, 0.0), (0.0, 1.0)),
    ],
)
def test_meteorological_wind_conversion(direction, cartesian, grid):
    assert meteorological_wind_to_cartesian(direction) == pytest.approx(cartesian)
    assert meteorological_wind_to_grid(direction) == pytest.approx(grid)


def test_noop_target_is_not_masked_as_invalid():
    model = RelativeInfernoModel(
        n_grid_channels=9,
        n_scalars=11,
        n_resources=4,
        n_zones=32,
    )
    grid = torch.zeros((1, 9, 8, 16), dtype=torch.float32)
    scalars = torch.zeros((1, 11), dtype=torch.float32)
    target_zones = torch.full((1, 4, N_TARGET_TYPES), -1, dtype=torch.long)
    target_features = torch.zeros((1, 4, N_TARGET_TYPES, TARGET_FEATURE_DIM))

    logits, _value, _classification = model(
        grid, scalars, target_zones, target_features
    )

    assert torch.isfinite(logits["target"][:, :, NOOP_TARGET_INDEX]).all()
    invalid_non_noop = logits["target"][:, :, :NOOP_TARGET_INDEX]
    assert torch.all(invalid_non_noop <= -1e8)


def test_trench_hold_bonus_is_reachable():
    env = InfernoEnv(seed=404)
    env.reset(seed=404)
    row, col = 100, 100
    env.sim.state[row, col] = SAFE
    env.sim.ignitability[row, col] = 0.0
    env.trench_mask[row, col] = True
    env.sim.state[row, col + 1] = BLAZE
    env.sim.blaze_age[row, col + 1] = 1

    _obs, reward, _done, info = env.step([])

    assert info["trench_held"] == 1
    assert info["reward_components"]["trench_bonus"] == TRENCH_BREAK_HOLD_BONUS
    assert reward >= info["reward_components"]["trench_bonus"] + info["reward_components"]["fire_penalty"]
    assert reward == pytest.approx(sum(info["reward_components"].values()))
