from types import SimpleNamespace

import numpy as np

from infernotactics.simulation.fire_sim import BLAZE, FUEL, SAFE, THREAT
from infernotactics.operations import apply_rescue, apply_trench, apply_water, select_effect_point


def _sim(size=7):
    return SimpleNamespace(
        height=size,
        width=size,
        state=np.full((size, size), FUEL, dtype=np.uint8),
        ignitability=np.ones((size, size), dtype=np.float32),
        building_density=np.zeros((size, size), dtype=np.float32),
    )


def test_water_and_trench_effects_modify_only_eligible_cells():
    sim = _sim()
    sim.state[3, 3] = BLAZE
    sim.state[3, 4] = THREAT
    assert apply_water(sim, 3, 3, radius=1) == 2
    assert sim.state[3, 3] == SAFE
    assert sim.state[3, 4] == SAFE

    trench_mask = np.zeros_like(sim.state, dtype=bool)
    affected = apply_trench(sim, 1, 1, radius=1, trench_mask=trench_mask)
    assert affected == 5
    assert trench_mask.sum() == affected
    assert np.all(sim.ignitability[trench_mask] == 0)


def test_rescue_and_effect_point_follow_current_fire():
    sim = _sim()
    sim.state[2, 3] = BLAZE
    sim.building_density[2, 3] = 0.8
    evacuated = set()

    assert apply_rescue(
        sim, 2, 3, evacuated, radius=1, building_threshold=0.1
    ) == 1
    assert evacuated == {(2, 3)}
    zone = {
        "row_range": (0, 5),
        "col_range": (0, 5),
        "centroid_row": 2,
        "centroid_col": 2,
    }
    assert select_effect_point(sim, zone) == (2, 3)
