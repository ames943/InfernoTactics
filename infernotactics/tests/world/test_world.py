"""Observable world timing/isolation checks, not historical validation."""
import copy

import numpy as np
from pyproj import Transformer
import pytest

from src.world.actions import candidate_actions, heuristic_action
from src.world.contracts import Action, event_sort_key
from src.world.synthetic import make_synthetic
from src.world.world import World


def fixture(duration_s=180):
    scenario = make_synthetic(size=31, cell_size_m=30., duration_s=duration_s)
    a, _, c, _, e, f = scenario.transform
    home = (c + 5.5 * a, f + 15.5 * e)
    target = (c + 25.5 * a, home[1])
    scenario.roads = {
        "nodes": [{"id": "home", "x": home[0], "y": home[1]},
                  {"id": "end", "x": target[0], "y": target[1]}],
        "edges": [{"id": name, "source": start, "target": end,
                   "geometry": [list(p), list(q)], "length_m": 600.,
                   "speed_m_s": 10., "speed_source": "synthetic fixture", "closed": False}
                  for name, start, end, p, q in
                  [("out", "home", "end", home, target),
                   ("back", "end", "home", target, home)]],
    }
    for unit in scenario.resources:
        unit.update(x=home[0], y=home[1], home_x=home[0], home_y=home[1],
                    mobilization_s=10., work_duration_s=20., speed_m_s=10.)
        if unit["kind"] != "handcrew":
            unit.update(capacity_l=1000., supply_l=1000., water_rate_l_s=10.,
                        refill_rate_l_s=20.)
    scenario.weather["wind_speed_m_s"][:] = 0.
    scenario.validate()
    return scenario, home, target


def ignition(world, row=15, col=15):
    a, _, c, _, e, f = world.scenario.transform
    xy = (c + (col + .5) * a, f + (row + .5) * e)
    world.ignite(*Transformer.from_crs(world.scenario.crs, "EPSG:4326", always_xy=True).transform(*xy))
    return xy


def engine_id(scenario):
    return next(unit["id"] for unit in scenario.resources if unit["kind"] == "engine")


def test_minute_horizon_and_quiet_duration_are_not_containment():
    scenario, _, _ = fixture(duration_s=125)
    world = World(scenario)
    for expected in (60, 120, 125):
        obs, reward, done, info = world.step([])
        assert obs["time_s"] == expected
        assert obs["no_new_growth_s"] == expected
        assert reward == 0.
        assert done == (expected == 125)
        assert not info["domain_exit"]
        assert not info["censored"]
        assert "contained" not in obs
    with pytest.raises(ValueError, match="ended"):
        world.step([])


def test_resource_audit_exact_reward_and_next_boundary_treatments():
    scenario, home, _ = fixture()
    world = World(scenario)
    action = Action(engine_id(scenario), "water", *home)
    obs, reward, done, info = world.step([action])
    # Mobilization 0..10, water 10..30, return at home, refill 30..40.
    assert info["resource_time_s"] == pytest.approx(40.)
    assert reward == pytest.approx(-.01 * 40. / (len(scenario.resources) * scenario.duration_s))
    assert world.treatments
    assert max(item.time_s for item in world.treatments) == 30.
    assert not any(event.kind == "treatment_activated" for event in info["events"])
    unit = next(unit for unit in obs["units"] if unit["id"] == action.unit_id)
    assert unit["status"] == "available"
    assert unit["supply_l"] == 1000.
    _, reward, _, info = world.step([])
    effects = [event for event in info["events"] if event.kind == "treatment_activated"]
    assert effects and all(event.time_s == 60. for event in effects)
    assert effects[-1].details["completed_s"] == 30.
    assert reward == 0.
    assert world.events == sorted(world.events, key=event_sort_key)


def test_partial_water_consumes_supply_but_publishes_only_completed_work():
    scenario, home, _ = fixture()
    unit_id = engine_id(scenario)
    next(unit for unit in scenario.resources if unit["id"] == unit_id)["work_duration_s"] = 100.
    world = World(scenario)
    _, reward, _, info = world.step([Action(unit_id, "water", *home)])
    assert not world.treatments
    assert not any(event.kind == "treatment_completed" for event in info["events"])
    assert info["resource_time_s"] == pytest.approx(60.)
    assert reward == pytest.approx(-.01 * 60. / (len(scenario.resources) * scenario.duration_s))
    unit = next(unit for unit in world.observe()["units"] if unit["id"] == unit_id)
    assert unit["status"] == "working"
    assert unit["supply_l"] == 500.
    _, _, _, info = world.step([])
    assert max(treatment.time_s for treatment in world.treatments) == 110.
    assert not any(event.kind == "treatment_activated" for event in info["events"])
    _, _, _, info = world.step([])
    effects = [event for event in info["events"] if event.kind == "treatment_activated"]
    assert effects and all(event.time_s == 120. for event in effects)
    assert all(event.details["completed_s"] == 110. for event in effects)


def test_invalid_batch_is_atomic_for_fire_resources_and_world():
    scenario, home, _ = fixture()
    world = World(scenario)
    ignition(world)
    before = world.observe()
    with pytest.raises(ValueError, match="unknown unit"):
        world.step([Action(engine_id(scenario), "water", *home), Action("missing", "return")])
    after = world.observe()
    assert world.time_s == 0
    assert after["units"] == before["units"]
    assert after["events"] == before["events"]
    np.testing.assert_array_equal(after["fire"]["burned"], before["fire"]["burned"])
    assert not world.treatments


def test_observation_arrays_and_nested_state_do_not_alias_world():
    scenario, _, _ = fixture()
    world = World(scenario)
    ignition(world)
    obs = world.observe()
    original_elevation = world.scenario.landscape["elevation"].copy()
    obs["grid"]["elevation"][:] = -999.
    obs["grid"]["burned"][:] = False
    obs["fire"]["burned"].setflags(write=True)
    obs["fire"]["burned"][:] = False
    obs["weather"]["wind_speed_m_s"][...] = 100.
    obs["units"][0]["x"] += 1000.
    obs["roads"]["edges"][0]["closed"] = True
    scenario.landscape["elevation"][:] = -123.
    fresh = world.observe()
    np.testing.assert_array_equal(fresh["grid"]["elevation"], original_elevation)
    assert fresh["fire"]["burned"].any()
    assert not np.any(fresh["weather"]["wind_speed_m_s"])
    assert not fresh["roads"]["edges"][0]["closed"]
    assert fresh["units"][0]["x"] != obs["units"][0]["x"]


def test_extra_observations_do_not_change_seeded_event_stream():
    scenario, _, _ = fixture()
    first, second = World(scenario, seed=31), World(scenario, seed=31)
    ignition(first)
    ignition(second)
    for _ in range(3):
        first.observe()
        first.observe()
        a, ar, ad, ai = first.step([])
        b, br, bd, bi = second.step([])
        np.testing.assert_array_equal(a["fire"]["arrival_time_s"], b["fire"]["arrival_time_s"])
        assert ar == br and ad == bd and ai == bi
        if ad:
            break


def test_domain_exit_censors_instead_of_claiming_containment():
    scenario, _, _ = fixture(duration_s=120)
    world = World(scenario)
    ignition(world, row=0)
    obs, _, done, info = world.step([])
    assert done and obs["censored"] and info["domain_exit"]
    assert any(event.kind == "domain_exit" for event in info["events"])
    assert obs["time_s"] == 60
    with pytest.raises(ValueError, match="ended"):
        world.step([])


def test_future_weather_and_hidden_effects_cannot_change_current_candidates():
    scenario, home, target = fixture(duration_s=7200)
    altered = copy.deepcopy(scenario)
    altered.weather["wind_speed_m_s"][1:] = 100.
    altered.provenance["suppression"]["k"] *= 9
    altered.provenance["suppression"]["decay_s"] *= 3
    worlds = [World(item) for item in (scenario, altered)]
    for world in worlds:
        ignition(world)
    left, right = [world.observe() for world in worlds]
    assert left["candidates"] == right["candidates"]
    assert left["candidate_metadata"] == right["candidate_metadata"]
    assert "suppression" not in left["provenance"] and "future_weather" not in left
    assert set(left["operations"]) == {"water_radius_m", "air_corridor_available"}
    for key, values in left["weather"].items():
        np.testing.assert_array_equal(values, scenario.weather[key][0])
    # A long aerial route must not inspect the weather at its future arrival.
    for world in worlds:
        heli = next(unit for unit in world.resources.snapshot() if unit["kind"] == "helicopter")
        world.resources._units[heli["id"]]["speed_m_s"] = .1
        world.resources._units[heli["id"]]["min_ground_speed_m_s"] = .1
        world.resources.validate_actions([Action(heli["id"], "water", *target)], 0)
    # Even when handed an altered scenario, generator consumes observed weather.
    assert candidate_actions(left, altered) == left["candidates"]


def test_candidates_wait_sorted_feasible_and_kind_specific():
    scenario, _, _ = fixture()
    world = World(scenario)
    ignition(world)
    obs = world.observe()
    assert obs["candidates"][0] == Action()
    ids = [action.unit_id for action in obs["candidates"][1:]]
    assert ids == sorted(ids)
    for unit in obs["units"]:
        actions = [action for action in obs["candidates"] if action.unit_id == unit["id"]]
        assert len(actions) <= 16
        for action in actions:
            world.resources.validate_actions([action], 0)
            a, _, c, _, e, f = scenario.transform
            row, col = int((action.y - f) / e), int((action.x - c) / a)
            if unit["kind"] == "helicopter":
                assert obs["fire"]["frontier"][row, col]
            elif unit["kind"] == "handcrew":
                assert not obs["fire"]["burned"][row, col]
            else:
                assert action.y == pytest.approx(unit["home_y"])
    assert candidate_actions(obs, scenario) == obs["candidates"]
    work = [(data["eta_s"], index, action) for index, (action, data) in
            enumerate(zip(obs["candidates"], obs["candidate_metadata"]))
            if action.operation in {"water", "line"}]
    assert work
    assert heuristic_action(obs, scenario) == min(work, key=lambda item: (item[0], item[1]))[2]


def test_busy_unit_has_feasible_return_and_no_new_work():
    scenario, _, target = fixture()
    world = World(scenario)
    unit_id = engine_id(scenario)
    obs, _, _, _ = world.step([Action(unit_id, "water", *target)])
    actions = [action for action in obs["candidates"] if action.unit_id == unit_id]
    assert actions == [Action(unit_id, "return")]
    world.resources.validate_actions(actions, world.time_s)


def test_equal_arc_samples_trace_boundary_instead_of_row_major_thinning():
    from src.world.actions import _frontier_samples
    burned = np.zeros((25, 25), dtype=bool)
    burned[2:23, 2:23] = True
    frontier = np.zeros_like(burned)
    frontier[2, 2:23] = frontier[22, 2:23] = True
    frontier[2:23, 2] = frontier[2:23, 22] = True
    obs = {"fire": {"burned": burned, "frontier": frontier},
           "grid": {"valid": np.ones_like(burned),
                    "fuel_model": np.full(burned.shape, 101)}}
    samples = _frontier_samples(obs)
    assert len(samples) == 16
    assert samples == _frontier_samples(obs)
    # Equal edge lengths place four sites on each side of this square.
    assert sum(row == 2 for row, col in samples) == 4
    assert sum(row == 22 for row, col in samples) == 4
    assert sum(col == 2 for row, col in samples) == 4
    assert sum(col == 22 for row, col in samples) == 4


def test_reward_uses_new_area_and_exposure_not_cumulative_burns():
    scenario, _, _ = fixture()
    scenario.landscape["asset_value"][:] = 2.
    world = World(scenario)
    ignition(world)
    for _ in range(2):
        _, reward, done, info = world.step([])
        expected = -(info["new_burned_m2"] / world.valid_burnable_m2)
        expected -= info["new_asset_exposure_m2"] / max(world.total_asset_m2, 1.)
        assert reward == pytest.approx(expected)
        assert info["resource_time_s"] == 0.
        if done:
            break


def test_six_hundred_metre_dispatch_arrives_ninety_completes_one_fifty():
    scenario, _, target = fixture(duration_s=300)
    unit_id = engine_id(scenario)
    configured = next(unit for unit in scenario.resources if unit["id"] == unit_id)
    configured.update(mobilization_s=30., work_duration_s=60.)
    world = World(scenario)
    obs, _, _, _ = world.step([Action(unit_id, "water", *target)])
    unit = next(unit for unit in obs["units"] if unit["id"] == unit_id)
    assert obs["time_s"] == 60 and unit["status"] == "traveling"
    assert not world.treatments
    obs, _, _, info = world.step([])
    unit = next(unit for unit in obs["units"] if unit["id"] == unit_id)
    assert unit["status"] == "working"
    assert unit["event_timestamps"]["work_started"] == 90.
    assert not world.treatments
    obs, _, _, info = world.step([])
    assert max(treatment.time_s for treatment in world.treatments) == 150.
    assert not any(event.kind == "treatment_activated" for event in info["events"])
    unit = next(unit for unit in obs["units"] if unit["id"] == unit_id)
    assert unit["status"] == "returning"
    _, _, _, info = world.step([])
    effects = [event for event in info["events"] if event.kind == "treatment_activated"]
    assert effects and all(event.time_s == 180. for event in effects)
    assert all(event.details["completed_s"] == 150. for event in effects)
