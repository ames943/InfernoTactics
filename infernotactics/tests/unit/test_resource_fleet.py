import math

import pytest

from infernotactics.operations import ResourceFleet, new_resource_unit


RESOURCE_TYPES = ("water_team", "trench_crew", "rescue_vehicle", "helicopter")
GROUND_TYPES = RESOURCE_TYPES[:3]
DELAYS = {
    resource_type: {
        "dispatch_delay_ticks": 1,
        "arrival_setup_delay_ticks": 1,
        "post_effect_busy_ticks": 2,
    }
    for resource_type in RESOURCE_TYPES
}


def _fleet(*, affected=3):
    reserved = []
    released = []
    effects = []
    stations = [
        {"station_id": "far", "roster": {"water_team": 1}},
        {"station_id": "near", "roster": {"water_team": 1}},
    ]
    travel = {
        "far": [100.0, math.inf, math.inf],
        "near": [50.0, 80.0, math.inf],
    }

    def dynamic_route(station_id, zone_id):
        seconds = travel[station_id][zone_id]
        return seconds + 10.0, [(station_id, zone_id, 0)] if math.isfinite(seconds) else []

    def apply_effect(resource_type, zone_id):
        effects.append((resource_type, zone_id))
        return affected, 4, 5

    fleet = ResourceFleet(
        resource_types=RESOURCE_TYPES,
        ground_resource_types=GROUND_TYPES,
        stations=stations,
        stations_by_type={"water_team": ["far", "near"]},
        station_travel_time_s=travel,
        delay_config=DELAYS,
        traffic_mode="synthetic",
        tick_duration_seconds=60,
        dynamic_ground_route=dynamic_route,
        reserve_route=lambda resource_type, edges: reserved.append((resource_type, list(edges))),
        release_route=lambda resource_type, edges: released.append((resource_type, list(edges))),
        apply_effect=apply_effect,
        travel_penalty_per_second=0.02,
        response_delay_penalty_per_second=0.02,
        wasted_penalty=10,
        suppression_reward=50,
        legacy_busy_ticks={resource_type: 2 for resource_type in RESOURCE_TYPES},
    )
    return fleet, reserved, released, effects


def test_resource_unit_schema_is_stable_and_available():
    unit = new_resource_unit("station-1")
    assert unit["station_id"] == "station-1"
    assert unit["state"] == "available"
    assert unit["target_zone"] is None
    assert unit["route_edges"] == []


def test_dispatch_selects_nearest_unit_and_reports_delay_costs():
    fleet, _, _, _ = _fleet()

    result = fleet.try_dispatch("water_team", 0, tick=7)

    assert result["status"] == "dispatched"
    assert result["station_id"] == "near"
    assert result["travel_time_s"] == 60
    assert result["traffic_delay_s"] == 10
    assert result["eta_ticks"] == 2
    assert result["reward_delta"] == pytest.approx(-2.4)
    unit = fleet.resources["water_team"][1]
    assert unit["state"] == "preparing"
    assert unit["dispatch_tick"] == 7


def test_lifecycle_reserves_route_applies_effect_and_recovers_unit():
    fleet, reserved, released, effects = _fleet()
    fleet.try_dispatch("water_team", 0, tick=0)

    assert fleet.advance(tick=1) == (0.0, [])
    assert reserved == [("water_team", [("near", 0, 0)])]
    assert fleet.advance(tick=2) == (0.0, [])
    reward, events = fleet.advance(tick=3)

    assert reward == 50
    assert effects == [("water_team", 0)]
    assert released == reserved
    assert events == [{
        "resource_type": "water_team",
        "zone": 0,
        "cells_affected": 3,
        "success": True,
        "row": 4,
        "col": 5,
    }]
    assert fleet.resources["water_team"][1]["state"] == "deployed"

    fleet.advance(tick=4)
    fleet.advance(tick=5)
    unit = fleet.resources["water_team"][1]
    assert unit["state"] == "available"
    assert unit["available_again_tick"] == 5


def test_dispatch_distinguishes_busy_from_unreachable_and_reset_restores_inventory():
    fleet, _, _, _ = _fleet()
    assert fleet.try_dispatch("water_team", 0, tick=0)["station_id"] == "near"
    assert fleet.try_dispatch("water_team", 0, tick=0)["station_id"] == "far"

    busy = fleet.try_dispatch("water_team", 0, tick=0)
    assert busy["status"] == "no_unit_available"
    assert busy["reward_delta"] == -10

    fleet.reset()
    unreachable = fleet.try_dispatch("water_team", 2, tick=0)
    assert unreachable["status"] == "zone_unreachable"
    assert all(unit["state"] == "available" for unit in fleet.resources["water_team"])


def test_failed_arrival_is_penalized():
    fleet, _, _, _ = _fleet(affected=0)
    fleet.try_dispatch("water_team", 0, tick=0)
    fleet.advance(tick=1)
    fleet.advance(tick=2)

    reward, events = fleet.advance(tick=3)

    assert reward == -10
    assert events[0]["success"] is False
