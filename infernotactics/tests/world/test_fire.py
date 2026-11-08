"""Numerical behavior of the pinned physical backend through its public adapter.

These are synthetic software checks, not historical spread/suppression validation.
The solver-boundary spy in the continuation regression still runs Pyretechnics;
it supplies a pending spot so the regression does not depend on a rare ember.
"""

import copy
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pytest
from pyproj import Transformer
from pyretechnics import eulerian_level_set as els

from src.world.contracts import Treatment
from src.world.fire import FireEngine
from src.world.synthetic import make_synthetic


def landscape(*, size=65, cell_m=10.0, duration_s=1800, wind_m_s=0.0, wind_from=270.0):
    scenario = make_synthetic(seed=7, size=size, cell_size_m=cell_m, duration_s=duration_s)
    scenario.weather["wind_speed_m_s"][:] = wind_m_s
    scenario.weather["upwind_direction_deg"][:] = wind_from
    # Dry, explicitly synthetic fuels keep numerical checks small without a
    # fabricated spread multiplier or replacing the real physics backend.
    scenario.weather["fuel_moisture_dead_1hr"][:] = 0.03
    scenario.weather["fuel_moisture_dead_10hr"][:] = 0.04
    scenario.weather["fuel_moisture_dead_100hr"][:] = 0.05
    scenario.weather["fuel_moisture_live_herbaceous"][:] = 0.30
    scenario.validate()
    return scenario


def projected(scenario, row, col):
    a, b, c, d, e, f = scenario.transform
    return c + a * (col + 0.5) + b * (row + 0.5), f + d * (col + 0.5) + e * (row + 0.5)


def lonlat(scenario, row, col):
    return Transformer.from_crs(scenario.crs, "EPSG:4326", always_xy=True).transform(
        *projected(scenario, row, col)
    )


def ignite_center(engine, scenario):
    height, width = scenario.landscape["valid"].shape
    engine.ignite(*lonlat(scenario, height // 2, width // 2))


def assert_snapshot_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        np.testing.assert_array_equal(left[name], right[name], err_msg=name)


def enable_spotting(scenario, *, brands=5e-11):
    scenario.provenance["spotting"] = {
        "enabled": True,
        "source": "illustrative numerical regression parameters, not calibrated",
        "firebrands_per_unit_heat": brands,
        "downwind_distance_mean": 11.7,
        "fireline_intensity_exponent": 0.3,
        "wind_speed_exponent": 0.55,
        "downwind_variance_mean_ratio": 10.0,
        "crosswind_distance_stdev": 10.0,
        "decay_distance": 500.0,
    }


def test_real_no_wind_fire_expands_and_preserves_physical_outputs():
    scenario = landscape()
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    initial = engine.snapshot()
    assert initial["burned"].sum() == 1
    assert initial["arrival_time_s"][32, 32] == 0

    engine.advance(1800, [])
    result = engine.snapshot()
    assert result["burned"].sum() > initial["burned"].sum()
    rows, cols = np.nonzero(result["burned"])
    assert rows.min() < 32 < rows.max()
    assert cols.min() < 32 < cols.max()
    arrived = result["arrival_time_s"][result["burned"]]
    assert np.isfinite(arrived).all()
    assert np.all((arrived >= 0) & (arrived <= 1800))
    newly_burned = result["burned"] & ~initial["burned"]
    assert np.all(result["fire_type"][newly_burned] > 0)
    assert np.any(result["fireline_intensity"][newly_burned] > 0)
    assert np.any(result["spread_rate_m_s"][newly_burned] > 0)
    assert result["frontier"].dtype == np.bool_
    assert result["frontier"].any()


@pytest.mark.parametrize("wind_from,axis,down_sign", [(270.0, 1, 1), (180.0, 0, -1)])
def test_wind_drives_downwind_asymmetry_in_north_up_grid(wind_from, axis, down_sign):
    scenario = landscape(size=81, duration_s=600, wind_m_s=8.0, wind_from=wind_from)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    engine.advance(300, [])
    burned = engine.snapshot()["burned"]
    positions = np.nonzero(burned)[axis]
    center = 40
    downwind = positions.max() - center if down_sign > 0 else center - positions.min()
    upwind = center - positions.min() if down_sign > 0 else positions.max() - center
    assert downwind > upwind
    assert np.count_nonzero((positions - center) * down_sign > 0) > np.count_nonzero(
        (positions - center) * down_sign < 0
    )


@pytest.mark.parametrize("reason", ["nonburnable", "nodata", "outside", "nan", "infinite"])
def test_invalid_ignition_is_rejected_without_mutation(reason):
    scenario = landscape(duration_s=120)
    row, col = 5, 8
    if reason == "nonburnable":
        scenario.landscape["fuel_model"][row, col] = 98
    elif reason == "nodata":
        scenario.landscape["valid"][row, col] = False
    coordinate = lonlat(scenario, row, col)
    if reason == "outside":
        coordinate = lonlat(scenario, -3, col)
    elif reason == "nan":
        coordinate = (float("nan"), coordinate[1])
    elif reason == "infinite":
        coordinate = (coordinate[0], float("inf"))
    engine = FireEngine(scenario, seed=7)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.ignite(*coordinate)
    assert_snapshot_equal(before, engine.snapshot())


def test_arbitrary_cell_ignition_multiple_points_and_repeat_are_idempotent():
    scenario = landscape(duration_s=120)
    engine = FireEngine(scenario, seed=7)
    points = [(7, 11), (41, 45)]
    for row, col in points:
        engine.ignite(*lonlat(scenario, row, col))
    before = engine.snapshot()
    assert before["burned"].sum() == 2
    for row, col in points:
        assert before["burned"][row, col]
        assert before["arrival_time_s"][row, col] == 0
    engine.ignite(*lonlat(scenario, *points[0]))
    assert_snapshot_equal(before, engine.snapshot())
    events = engine.advance(60, [])
    assert sum(event.kind == "ignition_ignored" for event in events) == 1
    engine.advance(120, [])
    assert all(engine.snapshot()["arrival_time_s"][row, col] == 0 for row, col in points)


def test_same_seed_and_extra_render_snapshots_do_not_change_fire_or_events():
    scenario = landscape(size=65, duration_s=600, wind_m_s=4.0)
    enable_spotting(scenario)
    bulk = FireEngine(scenario, seed=7)
    rendered = FireEngine(scenario, seed=7)
    ignite_center(bulk, scenario)
    ignite_center(rendered, scenario)
    bulk_events = bulk.advance(600, [])
    rendered_events = []
    for until_s in range(60, 601, 60):
        before = rendered.snapshot()
        assert_snapshot_equal(before, rendered.snapshot())
        rendered_events.extend(rendered.advance(until_s, []))
        for _ in range(3):
            rendered.snapshot()
    assert_snapshot_equal(bulk.snapshot(), rendered.snapshot())
    assert [asdict(event) for event in bulk_events] == [asdict(event) for event in rendered_events]


def test_snapshots_are_owned_and_cannot_mutate_engine_buffers():
    scenario = landscape(duration_s=120)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    original = engine.snapshot()
    detached = engine.snapshot()
    for name, array in detached.items():
        assert not np.shares_memory(array, original[name])
        # Owned arrays may be readonly; even opting back into writes must not
        # expose the engine's authoritative state.
        array.setflags(write=True)
        array.fill(0)
    assert_snapshot_equal(original, engine.snapshot())


def test_water_attenuates_new_growth_without_erasing_burn_history():
    scenario = landscape(duration_s=1800, wind_m_s=3.0)
    treated = FireEngine(scenario, seed=7)
    control = FireEngine(scenario, seed=7)
    for engine in (treated, control):
        ignite_center(engine, scenario)
        engine.advance(600, [])
    history = treated.snapshot()
    assert history["burned"].sum() > 1
    x, y = projected(scenario, 32, 32)
    water = Treatment(x=x, y=y, radius_m=10000.0, kind="water", strength=1000.0,
                      expires_s=1800.0, time_s=600.0)
    treated.advance(1200, [water])
    control.advance(1200, [])
    result = treated.snapshot()
    old = history["burned"]
    assert result["burned"][old].all()
    for name in ("arrival_time_s", "fire_type", "fireline_intensity"):
        np.testing.assert_array_equal(result[name][old], history[name][old])
    assert result["burned"].sum() < control.snapshot()["burned"].sum()
    # An expired treatment no longer stops growth. Never infer permanent
    # containment from a single quiet coupling interval.
    treated.advance(1800, [])
    assert treated.snapshot()["burned"].sum() >= result["burned"].sum()


def test_completed_line_blocks_spread_without_erasing_history():
    scenario = landscape(size=65, duration_s=600, wind_m_s=6.0)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    engine.advance(60, [])
    before = engine.snapshot()
    x, y = projected(scenario, 32, 32)
    line = Treatment(x=x, y=y, radius_m=10000.0, kind="line", strength=1.0,
                     expires_s=600.0, time_s=60.0)
    engine.advance(600, [line])
    after = engine.snapshot()
    np.testing.assert_array_equal(before["burned"], after["burned"])
    np.testing.assert_array_equal(before["arrival_time_s"], after["arrival_time_s"])


def test_nonburnable_barrier_stops_fire_when_spotting_disabled():
    scenario = landscape(size=65, cell_m=5.0, duration_s=1200, wind_m_s=8.0)
    scenario.landscape["fuel_model"][:, 30:34] = 98
    assert scenario.provenance["spotting"]["enabled"] is False
    engine = FireEngine(scenario, seed=7)
    engine.ignite(*lonlat(scenario, 32, 20))
    engine.advance(1200, [])
    result = engine.snapshot()
    assert result["burned"].sum() > 1
    assert result["burned"][:, 29].any(), "fire must actually reach the barrier"
    assert not result["burned"][:, 30:].any()
    assert not np.isfinite(result["arrival_time_s"][:, 30:]).any()


@pytest.mark.parametrize("until_s", [-60, 30, 61, 151])
def test_advance_rejects_noncanonical_or_out_of_horizon_instants_atomically(until_s):
    scenario = landscape(duration_s=150)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.advance(until_s, [])
    assert engine.time_s == 0
    assert_snapshot_equal(before, engine.snapshot())


def test_bulk_canonical_steps_and_partial_final_interval_are_allowed():
    scenario = landscape(duration_s=150)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    engine.advance(120, [])
    assert engine.time_s == 120
    with pytest.raises(ValueError):
        engine.advance(60, [])
    assert engine.time_s == 120
    engine.advance(150, [])
    assert engine.time_s == 150


def minute_weather(scenario, *, change_at_s, speed):
    bands = scenario.duration_s // 60
    scenario.weather_step_s = 60
    scenario.weather = {key: np.repeat(array[:1], bands, axis=0) for key, array in scenario.weather.items()}
    scenario.weather["wind_speed_m_s"][change_at_s // 60:] = speed
    start = datetime.fromisoformat(scenario.start_utc.replace("Z", "+00:00"))
    iso = lambda value: value.isoformat().replace("+00:00", "Z")
    scenario.provenance["weather_intervals"] = [
        {"start_utc": iso(start + timedelta(seconds=i * 60)),
         "end_utc": iso(start + timedelta(seconds=(i + 1) * 60))}
        for i in range(bands)
    ]
    scenario.validate()


def test_weather_changes_at_timestamp_not_before_and_uses_wind_from_convention():
    changing = landscape(size=81, duration_s=600)
    minute_weather(changing, change_at_s=120, speed=8.0)
    calm = copy.deepcopy(changing)
    calm.weather["wind_speed_m_s"][:] = 0.0
    engines = [FireEngine(scenario, seed=7) for scenario in (changing, calm)]
    for engine, scenario in zip(engines, (changing, calm)):
        ignite_center(engine, scenario)
        engine.advance(120, [])
    np.testing.assert_array_equal(engines[0].snapshot()["burned"], engines[1].snapshot()["burned"])
    np.testing.assert_array_equal(engines[0].snapshot()["arrival_time_s"], engines[1].snapshot()["arrival_time_s"])
    for engine in engines:
        engine.advance(600, [])
    windy_cols = np.nonzero(engines[0].snapshot()["burned"])[1]
    calm_cols = np.nonzero(engines[1].snapshot()["burned"])[1]
    assert windy_cols.max() > calm_cols.max()
    assert windy_cols.max() - 40 > 40 - windy_cols.min()


def test_seed_substreams_and_pending_spots_survive_real_solver_reentry(monkeypatch):
    scenario = landscape(duration_s=180)
    # Disable fresh random embers but keep spotting machinery enabled. An
    # explicitly queued ember isolates pending-state continuation from chance.
    enable_spotting(scenario, brands=0.0)
    solver = els.spread_fire_with_phi_field
    calls = []
    remote_row, remote_col = 8, 10
    upstream_cell = (64 - remote_row, remote_col)
    scheduled_min = 1.5

    def recording_solver(*args, **kwargs):
        pending = kwargs["spot_ignitions"]
        calls.append({
            "seed": kwargs["spot_config"]["random_seed"],
            "pending": copy.deepcopy(pending),
            "pending_object": pending,
            "start_time": kwargs["start_time"],
            "max_duration": kwargs["max_duration"],
            "max_cells_per_timestep": kwargs["max_cells_per_timestep"],
        })
        result = solver(*args, **kwargs)
        if len(calls) == 1:
            result["spot_ignitions"] = {scheduled_min: {upstream_cell}}
        return result

    monkeypatch.setattr(els, "spread_fire_with_phi_field", recording_solver)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    engine.advance(60, [])
    assert not engine.snapshot()["burned"][remote_row, remote_col]
    engine.advance(120, [])
    arrival = engine.snapshot()["arrival_time_s"][remote_row, remote_col]
    assert 90.0 <= arrival <= 120.0
    engine.advance(180, [])
    assert len(calls) == 3
    assert [call["seed"] for call in calls] == [
        int(np.random.SeedSequence([7, index]).generate_state(1)[0]) for index in range(3)
    ]
    assert len({call["seed"] for call in calls}) == 3
    assert calls[0]["pending"] == {}
    assert calls[1]["pending"] == {scheduled_min: {upstream_cell}}
    assert calls[2]["pending"] == {}
    assert len({id(call["pending_object"]) for call in calls}) == 3
    assert [call["start_time"] for call in calls] == [0.0, 1.0, 2.0]
    assert all(call["max_duration"] == 1.0 for call in calls)
    assert all(call["max_cells_per_timestep"] == 0.4 for call in calls)


def test_domain_exit_is_reported_not_mistaken_for_containment():
    scenario = landscape(size=9, cell_m=5.0, duration_s=600, wind_m_s=8.0)
    engine = FireEngine(scenario, seed=7)
    ignite_center(engine, scenario)
    events = engine.advance(600, [])
    assert any(event.kind == "domain_exit" for event in events)
    result = engine.snapshot()
    assert result["burned"][0].any() or result["burned"][-1].any() or result["burned"][:, 0].any() or result["burned"][:, -1].any()
    assert engine.time_s == 600
