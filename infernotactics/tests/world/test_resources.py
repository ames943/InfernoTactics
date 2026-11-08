"""Synthetic operation accounting checks; not operational validation."""
import copy

import numpy as np
import pytest

from src.world.contracts import Action
from src.world.fire import FireEngine, treatment_mask
from src.world.resources import ResourceEngine
from src.world.synthetic import make_synthetic


def fixture(*, kinds=("engine",), duration_s=1200):
    scenario = make_synthetic(size=81, cell_size_m=10.0, duration_s=duration_s)
    a, _, c, _, e, f = scenario.transform
    home = (c + 105.0, f + e * 40.5)
    target = (home[0] + 600.0, home[1])
    scenario.weather["wind_speed_m_s"][:] = 0.0
    scenario.roads = {
        "nodes": [{"id": "home", "x": home[0], "y": home[1]},
                  {"id": "goal", "x": target[0], "y": target[1]}],
        "edges": [{"id": name, "source": start, "target": end,
                   "geometry": [list(p0), list(p1)], "length_m": 600.0,
                   "speed_m_s": 10.0, "speed_source": "synthetic test",
                   "closed": False}
                  for name, start, end, p0, p1 in
                  [("out", "home", "goal", home, target),
                   ("back", "goal", "home", target, home)]],
    }
    prototypes = {unit["kind"]: unit for unit in scenario.resources}
    scenario.resources = []
    for index, kind in enumerate(kinds):
        unit = copy.deepcopy(prototypes[kind])
        unit.update(id=f"unit-{index}", home_x=home[0], home_y=home[1],
                    x=home[0], y=home[1], speed_m_s=10.0,
                    refill_rate_l_s=20.0 if kind != "handcrew" else 0.0)
        if kind != "handcrew":
            unit.update(capacity_l=1000.0, supply_l=1000.0)
        scenario.resources.append(unit)
    scenario.validate()
    return scenario, home, target


def state(engine, unit_id="unit-0"):
    return next(unit for unit in engine.snapshot() if unit["id"] == unit_id)


def dispatch(target, unit_id="unit-0", operation="water"):
    return Action(unit_id, operation, *target)


def test_exact_arrival_work_and_next_boundary_activation():
    scenario, home, target = fixture()
    engine = ResourceEngine(scenario)
    events = engine.submit([dispatch(target)], 0)
    assert [event.kind for event in events] == ["dispatch"]
    assert state(engine)["status"] == "mobilizing"
    events, treatments = engine.advance(89)
    assert not treatments
    assert state(engine)["status"] == "traveling"
    assert state(engine)["supply_l"] == 1000
    assert home[0] < state(engine)["x"] < target[0]
    events, treatments = engine.advance(90)
    assert not treatments
    assert [(event.kind, event.time_s) for event in events if event.kind == "arrival"] == [("arrival", 90)]
    assert state(engine)["status"] == "working"
    assert state(engine)["remaining_work_s"] == 60
    _, treatments = engine.advance(120)
    assert not treatments
    assert state(engine)["supply_l"] == pytest.approx(700)
    events, treatments = engine.advance(150)
    assert len(treatments) == 1
    water = treatments[0]
    assert water.kind == "water" and water.time_s == 150
    assert water.radius_m == scenario.provenance["suppression"]["radius_m"]
    audit = next(event for event in events if event.kind == "treatment_completed")
    assert audit.details["delivered_l"] == pytest.approx(600)
    assert water.strength * audit.details["area_m2"] == pytest.approx(600)
    assert audit.details["footprint_cells"] == int(treatment_mask(scenario, water).sum())
    assert state(engine)["status"] == "returning"
    fire = FireEngine(scenario, seed=3)
    fire.advance(180, [])
    activation = [event for event in fire.advance(240, treatments) if event.kind == "treatment_activated"]
    assert len(activation) == 1
    assert activation[0].time_s == 180
    assert activation[0].details["coupling_delay_s"] == 30


def test_return_refill_and_elapsed_busy_integral():
    scenario, home, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    events, treatments = engine.advance(300)
    assert sum(event.details.get("resource_time_s", 0) for event in events) == pytest.approx(240)
    assert len(treatments) == 1
    assert [(event.kind, event.time_s) for event in events if event.kind in
            {"arrival", "work_completed", "home_arrival", "refill_started", "refill_completed", "available"}] == [
        ("arrival", 90), ("work_completed", 150), ("home_arrival", 210),
        ("refill_started", 210), ("available", 240), ("refill_completed", 240)]
    unit = state(engine)
    assert unit["status"] == "available" and unit["assignment"] is None
    assert (unit["x"], unit["y"]) == home
    assert unit["supply_l"] == unit["capacity_l"] == 1000
    assert unit["event_timestamps"]["arrival"] == 90


def test_atomic_invalid_batch_and_validation_purity():
    scenario, _, target = fixture(kinds=("engine", "handcrew"))
    engine = ResourceEngine(scenario)
    before = engine.snapshot()
    engine.validate_actions([dispatch(target)], 0)
    assert engine.snapshot() == before
    with pytest.raises(ValueError):
        engine.submit([dispatch(target), dispatch(target, "unit-1")], 0)
    assert engine.snapshot() == before
    assert engine.time_s == 0
    assert scenario.resources[0]["status"] == "available"
    with pytest.raises(ValueError, match="duplicate"):
        engine.submit([dispatch(target), dispatch(target)], 0)
    assert engine.snapshot() == before


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True, 1])
def test_action_timestamp_must_be_current(bad):
    scenario, _, target = fixture()
    engine = ResourceEngine(scenario)
    with pytest.raises(ValueError):
        engine.submit([dispatch(target)], bad)


def test_busy_dispatch_is_rejected_and_wait_does_not_interrupt():
    scenario, _, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="busy"):
        engine.submit([dispatch(target)], 0)
    assert engine.snapshot() == before
    assert engine.submit([Action()], 0) == []
    assert engine.snapshot() == before


def test_return_interruption_uses_current_route_position():
    scenario, home, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    engine.advance(60)
    assert state(engine)["x"] == pytest.approx(home[0] + 300)
    engine.submit([Action("unit-0", "return")], 60)
    assert state(engine)["x"] == pytest.approx(home[0] + 300)
    events, treatments = engine.advance(90)
    assert not treatments
    assert state(engine)["status"] == "available"
    assert state(engine)["x"] == pytest.approx(home[0])
    assert next(event for event in events if event.kind == "home_arrival").time_s == 90


def test_return_during_work_publishes_only_earned_water():
    scenario, _, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    engine.advance(120)
    engine.submit([Action("unit-0", "return")], 120)
    events, treatments = engine.advance(120)
    assert len(treatments) == 1 and treatments[0].time_s == 120
    assert next(event for event in events if event.kind == "treatment_completed").details["delivered_l"] == 300
    assert state(engine)["supply_l"] == 700
    _, later = engine.advance(300)
    assert not later


def test_shared_base_refill_is_fifo_not_parallel():
    scenario, _, target = fixture(kinds=("engine", "engine"))
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target, "unit-1"), dispatch(target)], 0)
    events, _ = engine.advance(225)
    assert state(engine)["supply_l"] == 700
    assert not state(engine)["refill_waiting"]
    assert state(engine, "unit-1")["supply_l"] == 400
    assert state(engine, "unit-1")["refill_waiting"]
    later, _ = engine.advance(360)
    starts = [(event.unit_id, event.time_s) for event in events + later if event.kind == "refill_started"]
    assert starts == [("unit-0", 210), ("unit-1", 240)]
    assert sum(event.details.get("resource_time_s", 0) for event in events + later) == pytest.approx(510)
    assert all(unit["status"] == "available" for unit in engine.snapshot())


def test_distinct_base_ids_have_independent_refill_slots():
    scenario, _, target = fixture(kinds=("engine", "engine"))
    scenario.resources[1]["base_id"] = "other-base"
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target), dispatch(target, "unit-1")], 0)
    events, _ = engine.advance(300)
    assert [(event.unit_id, event.time_s) for event in events if event.kind == "refill_started"] == [
        ("unit-0", 210), ("unit-1", 210)]
    assert all(unit["status"] == "available" for unit in engine.snapshot())


def test_closure_blocks_remaining_route_without_teleport_or_supply_loss():
    scenario, home, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    engine.advance(45)
    position = state(engine)["x"]
    for edge in scenario.roads["edges"]:
        edge["closed"] = True
    events, _ = engine.advance(90)
    blocked = next(event for event in events if event.kind == "route_blocked")
    assert blocked.time_s == 60
    assert position <= state(engine)["x"] < target[0]
    assert state(engine)["blocked"] and state(engine)["supply_l"] == 1000
    stopped = state(engine)["x"]
    with pytest.raises(ValueError):
        engine.submit([Action("unit-0", "return")], 90)
    assert state(engine)["x"] == stopped
    engine.advance(120)
    assert state(engine)["x"] == stopped
    for edge in scenario.roads["edges"]:
        edge["closed"] = False
    events, _ = engine.advance(180)
    assert any(event.kind == "route_replanned" for event in events)
    assert not state(engine)["blocked"]


def test_air_wind_rejects_dispatch_atomically():
    scenario, _, target = fixture(kinds=("helicopter",))
    scenario.weather["wind_speed_m_s"][:] = 30
    engine = ResourceEngine(scenario)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.submit([dispatch(target)], 0)
    assert engine.snapshot() == before


def test_air_work_suspends_at_boundary_and_conserves_supplies():
    scenario, home, _ = fixture(kinds=("helicopter",))
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(home)], 0)
    engine.advance(45)
    assert state(engine)["status"] == "working"
    scenario.weather["wind_speed_m_s"][:] = 30
    events, treatments = engine.advance(60)
    assert not treatments
    assert next(event for event in events if event.kind == "work_suspended").time_s == 60
    supply = state(engine)["supply_l"]
    remaining = state(engine)["remaining_work_s"]
    events, treatments = engine.advance(120)
    assert not treatments
    assert state(engine)["supply_l"] == supply
    assert state(engine)["remaining_work_s"] == remaining
    assert sum(event.details.get("resource_time_s", 0) for event in events) == 60
    with pytest.raises(ValueError):
        engine.submit([Action("unit-0", "return")], 120)
    scenario.weather["wind_speed_m_s"][:] = 0
    events, treatments = engine.advance(180)
    assert any(event.kind == "work_resumed" for event in events)
    assert len(treatments) == 1
    assert next(event for event in events if event.kind == "work_completed").time_s == 150


def test_line_progress_is_a_segment_not_target_disk():
    scenario, home, _ = fixture(kinds=("handcrew",))
    unit = scenario.resources[0]
    unit.update(line_rate_m_s=1.0, work_duration_s=60.0)
    target = (home[0] + 300, home[1] + 60)
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target, operation="line")], 0)
    events, treatments = engine.advance(60)
    arrival = next(event.time_s for event in events if event.kind == "arrival")
    assert arrival == pytest.approx(60)
    assert not treatments
    events, treatments = engine.advance(80)
    assert treatments
    assert all(treatment.kind == "line" and treatment.radius_m == 0 for treatment in treatments)
    assert all(treatment.time_s > arrival and treatment.time_s <= 80 for treatment in treatments)
    assert all(treatment.y < target[1] for treatment in treatments)
    assert state(engine)["y"] == pytest.approx(home[1] + 20)
    assert state(engine)["supply_l"] == 0
    _, later = engine.advance(120)
    assert later
    assert state(engine)["status"] == "returning"
    all_cells = np.zeros(scenario.landscape["valid"].shape, dtype=bool)
    for treatment in treatments + later:
        all_cells |= treatment_mask(scenario, treatment)
    assert all_cells.sum() <= 7
    assert np.count_nonzero(all_cells.any(axis=0)) == 1


def test_depleted_water_rejected_and_low_supply_limits_work():
    scenario, _, target = fixture()
    scenario.resources[0]["supply_l"] = 0
    engine = ResourceEngine(scenario)
    with pytest.raises(ValueError, match="depleted"):
        engine.submit([dispatch(target)], 0)
    scenario.resources[0]["supply_l"] = 50
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    events, treatments = engine.advance(100)
    assert treatments[0].time_s == 95
    assert next(event for event in events if event.kind == "treatment_completed").details["delivered_l"] == 50
    assert state(engine)["supply_l"] == 0


def test_invalid_advance_and_owned_snapshot():
    scenario, _, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    snapshot = engine.snapshot()
    snapshot[0]["route"].clear()
    snapshot[0]["event_timestamps"]["dispatch"] = 999
    assert state(engine)["route"]
    assert state(engine)["event_timestamps"]["dispatch"] == 0
    engine.advance(30)
    for until in (-1, 29, 1201, float("nan"), True):
        with pytest.raises(ValueError):
            engine.advance(until)
    assert engine.time_s == 30


def test_partitioning_advance_conserves_work_and_busy_time():
    scenario, _, target = fixture()
    one, many = ResourceEngine(scenario), ResourceEngine(scenario)
    one.submit([dispatch(target)], 0)
    many.submit([dispatch(target)], 0)
    events, treatments = one.advance(300)
    partition_events, partition_treatments = [], []
    for until in (17, 45, 91, 119, 150, 215, 261, 300):
        next_events, next_treatments = many.advance(until)
        partition_events.extend(next_events)
        partition_treatments.extend(next_treatments)
    assert many.snapshot() == one.snapshot()
    assert partition_treatments == treatments
    assert sum(event.details.get("resource_time_s", 0) for event in events) == pytest.approx(
        sum(event.details.get("resource_time_s", 0) for event in partition_events))


def test_unreachable_batch_is_atomic():
    scenario, home, target = fixture(kinds=("engine", "engine"))
    engine = ResourceEngine(scenario)
    unreachable = (home[0], home[1] + 150)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.submit([dispatch(target), dispatch(unreachable, "unit-1")], 0)
    assert engine.snapshot() == before


def test_shift_end_suspends_without_forced_return():
    scenario, home, _ = fixture()
    scenario.resources[0]["shift_end_s"] = 60.0
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(home)], 0)
    events, treatments = engine.advance(120)
    assert not treatments
    assert next(event for event in events if event.kind == "work_suspended").time_s == 60
    unit = state(engine)
    assert unit["status"] == "working" and unit["suspended"]
    assert unit["supply_l"] == 700
    assert unit["remaining_work_s"] == 30
    assert (unit["x"], unit["y"]) == home
    engine.submit([Action("unit-0", "return")], 120)
    _, treatments = engine.advance(120)
    assert treatments[0].time_s == 120


def test_engine_cannot_construct_line_and_air_corridor_rejects():
    scenario, _, target = fixture(kinds=("engine", "helicopter"))
    scenario.resources[1]["air_corridor_available"] = False
    engine = ResourceEngine(scenario)
    before = engine.snapshot()
    with pytest.raises(ValueError, match="handcrew"):
        engine.submit([dispatch(target, operation="line")], 0)
    with pytest.raises(ValueError):
        engine.submit([dispatch(target, "unit-1")], 0)
    assert engine.snapshot() == before


def test_line_water_and_slope_cells_reject_before_dispatch():
    scenario, home, _ = fixture(kinds=("handcrew",))
    target = (home[0] + 300, home[1] + 60)
    a, _, c, _, e, f = scenario.transform
    row, col = int((target[1] - f) / e), int((target[0] - c) / a)
    scenario.landscape["fuel_model"][row, col] = 98
    engine = ResourceEngine(scenario)
    before = engine.snapshot()
    with pytest.raises(ValueError):
        engine.submit([dispatch(target, operation="line")], 0)
    assert engine.snapshot() == before
    scenario.landscape["fuel_model"][row, col] = 101
    scenario.landscape["slope"][row, col] = 2.0
    with pytest.raises(ValueError):
        engine.submit([dispatch(target, operation="line")], 0)
    assert engine.snapshot() == before


def test_no_reuse_before_refill_completion_240_seconds():
    scenario, _, target = fixture()
    engine = ResourceEngine(scenario)
    engine.submit([dispatch(target)], 0)
    engine.advance(239)
    assert state(engine)["status"] == "refilling"
    with pytest.raises(ValueError, match="busy"):
        engine.submit([dispatch(target)], 239)
    engine.advance(240)
    assert state(engine)["status"] == "available"
    assert state(engine)["supply_l"] == 1000
    engine.submit([dispatch(target)], 240)
    assert state(engine)["status"] == "mobilizing"
