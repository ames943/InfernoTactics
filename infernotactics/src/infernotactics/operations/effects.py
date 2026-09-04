"""Pure, independently testable suppression and evacuation effects."""

import numpy as np

from infernotactics.domain.fire import BLAZE, FUEL, SAFE, THREAT
from infernotactics.simulation import FireEngine


def _disk_slice(center_row, center_col, radius, height, width):
    row_start = max(0, center_row - radius)
    row_stop = min(height, center_row + radius + 1)
    col_start = max(0, center_col - radius)
    col_stop = min(width, center_col + radius + 1)
    rows, cols = np.ogrid[row_start:row_stop, col_start:col_stop]
    mask = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2
    return row_start, row_stop, col_start, col_stop, mask


def apply_water(sim: FireEngine, row, col, *, radius):
    """Extinguish Threat/Blaze cells in a local disk and return the count."""
    row_start, row_stop, col_start, col_stop, disk = _disk_slice(
        row, col, radius, sim.height, sim.width
    )
    region = sim.state[row_start:row_stop, col_start:col_stop]
    active = disk & np.isin(region, (THREAT, BLAZE))
    affected = int(active.sum())
    if affected:
        region[active] = SAFE
    return affected


def apply_trench(sim: FireEngine, row, col, *, radius, trench_mask=None):
    """Create a permanent firebreak if the footprint is not already burning."""
    row_start, row_stop, col_start, col_stop, disk = _disk_slice(
        row, col, radius, sim.height, sim.width
    )
    region = sim.state[row_start:row_stop, col_start:col_stop]
    if (disk & np.isin(region, (THREAT, BLAZE))).any():
        return 0
    fuel = disk & (region == FUEL)
    affected = int(fuel.sum())
    if affected:
        region[fuel] = SAFE
        sim.ignitability[row_start:row_stop, col_start:col_stop][fuel] = 0.0
        if trench_mask is not None:
            trench_mask[row_start:row_stop, col_start:col_stop][fuel] = True
    return affected


def apply_rescue(sim: FireEngine, row, col, evacuated_cells, *, radius, building_threshold):
    """Record threatened building cells covered by an evacuation operation."""
    row_start, row_stop, col_start, col_stop, disk = _disk_slice(
        row, col, radius, sim.height, sim.width
    )
    buildings = sim.building_density[row_start:row_stop, col_start:col_stop]
    state = sim.state[row_start:row_stop, col_start:col_stop]
    threatened = disk & (buildings > building_threshold) & np.isin(state, (THREAT, BLAZE))
    rows, cols = np.where(threatened)
    evacuated_cells.update(
        (row_start + int(local_row), col_start + int(local_col))
        for local_row, local_col in zip(rows, cols)
    )
    return len(rows)


def select_effect_point(sim: FireEngine, zone):
    """Aim at current active fire in a zone, or its centroid when unburned."""
    row_start, row_stop = zone["row_range"]
    col_start, col_stop = zone["col_range"]
    region = sim.state[row_start:row_stop, col_start:col_stop]
    active_rows, active_cols = np.where(np.isin(region, (THREAT, BLAZE)))
    if len(active_rows):
        return (
            row_start + int(round(active_rows.mean())),
            col_start + int(round(active_cols.mean())),
        )
    return zone["centroid_row"], zone["centroid_col"]
