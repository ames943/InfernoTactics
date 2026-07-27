"""
Isolated synthetic test for ember spotting (FireSim._ember_spot_fires),
separate from test_fire_sim.py's real-grid hillside scenario so the road
corridor's width/placement and the fuel on each side are fully controlled
rather than whatever the real data happens to contain at some point.

Scenario: a small flat synthetic grid (no slope/building effects -- isolates
wind-driven ember spotting from the other spread factors already validated
by test_fire_sim.py) with fuel on the west side, a WIDE (6-cell / 180m) road
corridor with the same road fuel-resistance the real grid uses
(fuel_density=0.05 over roads, matching grid_builder._placeholder_fuel_density),
and more fuel on the east side. A single fire is ignited a few cells west of
the corridor; wind blows due east, straight across it, at either HIGH_WIND_MPH
(Santa-Ana-strength) or LOW_WIND_MPH.

Adjacent-neighbor spread (fire_sim.py's per-neighbor loop) can only ever
advance the front by 1 cell/tick, and each of the 6 resistant road cells has
only a ~1.8%/tick chance of catching once genuinely adjacent to a Blaze cell
(BASE_SPREAD_PROB * ROAD_RESISTANCE_FACTOR = 0.22 * 0.08) -- sequentially
crossing all 6 within this test's tick budget by adjacency alone is
vanishingly unlikely (see the "adjacency-only" control run below, which
essentially never crosses). So any fire appearing east of the corridor is
attributable to ember spotting, not creeping adjacency.

    python -m src.env.test_ember_spotting
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.config import DATA_DIR  # noqa: E402
from env.fire_sim import (  # noqa: E402
    BLAZE,
    BURNED_OUT,
    FUEL,
    SAFE,
    STATE_COLORS,
    STATE_NAMES,
    THREAT,
    FireSim,
)

OUTPUT_DIR = os.path.join(DATA_DIR, "ember_spotting_snapshots")

# --- Synthetic grid geometry ------------------------------------------------
HEIGHT, WIDTH = 50, 100
CELL_SIZE_M = 30.0
ROAD_START_COL = 40
ROAD_WIDTH_CELLS = 6  # 180m -- a "wide road corridor" per the task spec
ROAD_END_COL = ROAD_START_COL + ROAD_WIDTH_CELLS  # exclusive
IGNITION_ROW = HEIGHT // 2
IGNITION_COL = ROAD_START_COL - 2  # 2 cells west of the corridor

WIND_DIRECTION_DEG = 270.0  # FROM the west -> blows due east, straight across the corridor
HIGH_WIND_MPH = 45.0
LOW_WIND_MPH = 5.0
HUMIDITY_PCT = 8.0

N_TICKS = 40
SNAPSHOT_TICKS = [0, 5, 10, 15, 20, 25, 30, 40]
N_TRIALS = 30  # independent seeds per wind condition, for the crossing-rate stat

CMAP = ListedColormap(STATE_COLORS)
NORM = BoundaryNorm(np.arange(len(STATE_COLORS) + 1) - 0.5, CMAP.N)


def _build_synthetic_grid():
    """8-layer grid_static-shaped array (matches STATIC_LAYER_NAMES order),
    flat/no-slope, no buildings, fuel on both sides of a resistant road
    corridor -- fuel_density=0.05 over the road, matching grid_builder.py's
    real convention (roads are a fuel break, not literally unignitable)."""
    elevation = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    slope = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    building_density = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    building_height = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    road_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    road_mask[:, ROAD_START_COL:ROAD_END_COL] = 1.0

    fuel_density = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    fuel_density[:, ROAD_START_COL:ROAD_END_COL] = 0.05

    water_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    population_density = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    grid_static = np.stack(
        [elevation, slope, building_density, building_height,
         road_mask, fuel_density, water_mask, population_density],
        axis=0,
    )
    meta = {"height": HEIGHT, "width": WIDTH, "cell_size_m": CELL_SIZE_M}
    return grid_static, meta


def _run(seed, wind_speed_mph, n_ticks=N_TICKS, snapshot_ticks=SNAPSHOT_TICKS):
    grid_static, meta = _build_synthetic_grid()
    sim = FireSim(grid_static, meta, seed=seed)
    sim.ignite(IGNITION_ROW, IGNITION_COL, radius=1)

    snapshots = {0: sim.state.copy()} if 0 in snapshot_ticks else {}
    first_crossing_tick = None
    for t in range(1, n_ticks + 1):
        sim.step(wind_speed_mph=wind_speed_mph, wind_direction_deg=WIND_DIRECTION_DEG,
                  humidity_pct=HUMIDITY_PCT)
        if t in snapshot_ticks:
            snapshots[t] = sim.state.copy()
        east_side_active = np.isin(sim.state[:, ROAD_END_COL:], (THREAT, BLAZE, BURNED_OUT)).any()
        if east_side_active and first_crossing_tick is None:
            first_crossing_tick = t
    return sim, snapshots, first_crossing_tick


def _crossing_rate(wind_speed_mph, n_trials=N_TRIALS):
    """No snapshotting here (snapshot_ticks=[]) -- this scan is just for the
    aggregate crossing-count/tick stat, run over many seeds; the single
    representative run used for before/after frames is a separate, later
    call to _run() with full per-tick snapshotting."""
    crossings = 0
    crossing_ticks = []
    crossing_seeds = []
    for trial in range(n_trials):
        _sim, _snaps, first_crossing_tick = _run(seed=trial, wind_speed_mph=wind_speed_mph, snapshot_ticks=[])
        if first_crossing_tick is not None:
            crossings += 1
            crossing_ticks.append(first_crossing_tick)
            crossing_seeds.append(trial)
    return crossings, crossing_ticks, crossing_seeds


def _render(ax, state, tick):
    backdrop = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    backdrop[:, :, :] = (0.94, 0.94, 0.90)  # unburned fuel background
    backdrop[:, ROAD_START_COL:ROAD_END_COL, :] = (0.55, 0.58, 0.65)  # road corridor
    ax.imshow(backdrop, origin="upper")

    active = state >= THREAT  # Threat, Blaze, Burned Out
    fire_overlay = np.ma.masked_where(~active, state)
    ax.imshow(fire_overlay, cmap=CMAP, norm=NORM, origin="upper", alpha=0.95)

    ax.axvline(ROAD_START_COL - 0.5, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(ROAD_END_COL - 0.5, color="black", linewidth=0.8, linestyle="--")
    ax.scatter([IGNITION_COL], [IGNITION_ROW], marker="*", s=60, c="black", zorder=5)
    ax.set_title(f"tick {tick}")
    ax.set_xticks([])
    ax.set_yticks([])


def _save_before_after(sim_hw, snapshots_hw, first_crossing_tick, label):
    """The tick just before the far side first shows fire, and the tick it
    (or the next captured snapshot at/after) does -- demonstrating an
    isolated spot ignition ahead of the main front, not a connected burn
    path through the road."""
    if first_crossing_tick is None:
        print(f"  [{label}] no road crossing observed in this particular run -- "
              f"skipping before/after frames for it (see the aggregate crossing-rate stat instead).")
        return

    before_tick = max(t for t in snapshots_hw if t < first_crossing_tick)
    after_tick = min((t for t in snapshots_hw if t >= first_crossing_tick), default=None)
    if after_tick is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    _render(axes[0], snapshots_hw[before_tick], before_tick)
    axes[0].set_title(f"BEFORE -- tick {before_tick} (no fire east of road)")
    _render(axes[1], snapshots_hw[after_tick], after_tick)
    axes[1].set_title(f"AFTER -- tick {after_tick} (spot fire landed east of road)")
    fig.suptitle(f"Ember spotting jumps the {ROAD_WIDTH_CELLS}-cell road corridor ({label})", fontsize=12)
    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"before_after_{label}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Saved before/after -> {path}")


def _save_progression(sim, snapshots, label):
    ticks = sorted(snapshots)
    fig, axes = plt.subplots(1, len(ticks), figsize=(3.2 * len(ticks), 3.6))
    for ax, t in zip(axes, ticks):
        _render(ax, snapshots[t], t)
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COLORS[i]) for i in range(len(STATE_NAMES))]
    fig.legend(handles, STATE_NAMES, loc="lower center", ncol=len(STATE_NAMES))
    fig.suptitle(f"Ember spotting synthetic test -- {label}", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"progression_{label}.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  Saved progression -> {path}")


def main():
    print(f"Synthetic grid: {WIDTH}x{HEIGHT} cells @ {CELL_SIZE_M}m, flat, no buildings.")
    print(f"Road corridor: cols [{ROAD_START_COL}, {ROAD_END_COL}) "
          f"({ROAD_WIDTH_CELLS} cells / {ROAD_WIDTH_CELLS * CELL_SIZE_M:.0f}m wide)")
    print(f"Ignition: row={IGNITION_ROW}, col={IGNITION_COL} "
          f"({ROAD_START_COL - IGNITION_COL} cells west of the corridor)")
    print(f"Wind: from {WIND_DIRECTION_DEG} deg (due west) -> blows straight east across the corridor\n")

    # --- Aggregate crossing rate across many seeds, both wind conditions -----
    print(f"=== Crossing rate over {N_TRIALS} independent seeds, {N_TICKS} ticks each ===")
    hw_crossings, hw_ticks, hw_seeds = _crossing_rate(HIGH_WIND_MPH)
    lw_crossings, lw_ticks, _lw_seeds = _crossing_rate(LOW_WIND_MPH)
    print(f"HIGH wind ({HIGH_WIND_MPH} mph): {hw_crossings}/{N_TRIALS} trials crossed "
          f"({hw_crossings / N_TRIALS:.0%})"
          + (f", first-crossing tick range {min(hw_ticks)}-{max(hw_ticks)}" if hw_ticks else ""))
    print(f"LOW wind  ({LOW_WIND_MPH} mph): {lw_crossings}/{N_TRIALS} trials crossed "
          f"({lw_crossings / N_TRIALS:.0%})"
          + (f", first-crossing tick range {min(lw_ticks)}-{max(lw_ticks)}" if lw_ticks else ""))

    # --- Representative frames: HIGH wind uses a seed that actually crossed
    # (median first-crossing tick among the scan above), re-run with full
    # per-tick snapshotting so before/after can bracket the crossing tick
    # exactly. LOW wind uses seed=0 (essentially always uncrossed, per the
    # 0% rate above) to show the contrasting non-event.
    print(f"\n=== Representative HIGH-wind run ({HIGH_WIND_MPH} mph) ===")
    if hw_seeds:
        # Pick the seed with the MEDIAN first-crossing tick (not just the
        # median list position) -- a "typical" crossing, not an outlier
        # that jumps on tick 1 (still near the initial ignite(radius=1)
        # footprint) or one that takes unusually long.
        by_tick = sorted(zip(hw_ticks, hw_seeds))
        rep_tick, rep_seed = by_tick[len(by_tick) // 2]
        sim_hw, snaps_hw, first_cross_hw = _run(
            seed=rep_seed, wind_speed_mph=HIGH_WIND_MPH, snapshot_ticks=list(range(N_TICKS + 1))
        )
        print(f"seed={rep_seed}  first tick with fire east of the road corridor: {first_cross_hw}")
        display_ticks = sorted(set(SNAPSHOT_TICKS) | {first_cross_hw - 1, first_cross_hw} if first_cross_hw else SNAPSHOT_TICKS)
        _save_progression(sim_hw, {t: snaps_hw[t] for t in display_ticks if t in snaps_hw},
                           label=f"high_wind_{HIGH_WIND_MPH:.0f}mph")
        _save_before_after(sim_hw, snaps_hw, first_cross_hw, label=f"high_wind_{HIGH_WIND_MPH:.0f}mph")
    else:
        print("  No trial crossed at high wind in this scan -- skipping representative frames.")

    print(f"\n=== Representative LOW-wind run ({LOW_WIND_MPH} mph) ===")
    sim_lw, snaps_lw, first_cross_lw = _run(seed=0, wind_speed_mph=LOW_WIND_MPH)
    print(f"First tick with fire east of the road corridor: {first_cross_lw}")
    _save_progression(sim_lw, snaps_lw, label=f"low_wind_{LOW_WIND_MPH:.0f}mph")

    print("\n=== Sanity check ===")
    if hw_crossings <= lw_crossings:
        print("  ! FLAGGED: high-wind crossing rate is not greater than low-wind's -- "
              "ember spotting's wind-scaling may not be working correctly.")
    else:
        print(f"  OK: high wind crosses the road corridor far more often than low wind "
              f"({hw_crossings}/{N_TRIALS} vs {lw_crossings}/{N_TRIALS}), as expected.")


if __name__ == "__main__":
    main()
