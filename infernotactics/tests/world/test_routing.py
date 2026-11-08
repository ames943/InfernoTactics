"""Routing geometry, directed reachability and interrupted-position contracts."""

import copy
import math

import numpy as np
import pytest

from src.world.routing import (
    check_air_constraints, line_access_point, plan_route, route_invalid_reason,
    validate_ground_segment,
)
from src.world.synthetic import make_synthetic


@pytest.fixture
def scenario():
    result = make_synthetic(size=21, duration_s=7200, cell_size_m=10)
    result.transform = (10.0, 0.0, 0.0, 0.0, -10.0, 210.0)
    result.weather["wind_speed_m_s"][:] = 0
    result.roads = {"nodes": [], "edges": []}
    return result


def road(scenario, name, source, target, geometry, speed=10):
    nodes = {node["id"] for node in scenario.roads["nodes"]}
    for identity, point in ((source, geometry[0]), (target, geometry[-1])):
        if identity not in nodes:
            scenario.roads["nodes"].append({"id": identity, "x": point[0], "y": point[1]})
            nodes.add(identity)
    edge = {"id": name, "source": source, "target": target,
            "geometry": geometry, "speed_m_s": speed,
            "length_m": sum(math.dist(a, b) for a, b in zip(geometry, geometry[1:])),
            "closed": False, "speed_source": "test illustrative"}
    scenario.roads["edges"].append(edge)
    return edge


def unit(scenario, kind="engine", x=15, y=15, **kwargs):
    result = copy.deepcopy(next(u for u in scenario.resources if u["kind"] == kind))
    result.update(x=x, y=y, home_x=x, home_y=y, max_access_m=0,
                  speed_m_s=10, offroad_speed_m_s=2)
    result.update(kwargs)
    return result


def seconds(route):
    return sum(segment["seconds"] for segment in route)


def length(route):
    return sum(segment["length_m"] for segment in route)


def test_road_follows_polyline_not_node_chord(scenario):
    road(scenario, "curve", "a", "b", [[15, 15], [15, 65], [95, 65]])
    result = plan_route(scenario, unit(scenario), 95, 65, 0)
    assert [(s["x"], s["y"]) for s in result] == [(15, 65), (95, 65)]
    assert length(result) == pytest.approx(130)
    assert seconds(result) == pytest.approx(13)
    assert all(segment["edge_id"] == "curve" for segment in result)


def test_minimum_time_not_minimum_distance_and_unit_speed_cache(scenario):
    road(scenario, "short-slow", "a", "b", [[15, 15], [95, 15]], speed=1)
    road(scenario, "fast-1", "a", "c", [[15, 15], [15, 55]], speed=20)
    road(scenario, "fast-2", "c", "b", [[15, 55], [95, 55], [95, 15]], speed=20)
    fast = unit(scenario, speed_m_s=20)
    fast_route = plan_route(scenario, fast, 95, 15, 0)
    assert seconds(fast_route) == pytest.approx(8)
    assert {s["edge_id"] for s in fast_route} == {"fast-1", "fast-2"}
    slow = unit(scenario, speed_m_s=.5)
    slow_route = plan_route(scenario, slow, 95, 15, 0)
    assert seconds(slow_route) == pytest.approx(160)
    assert {s["edge_id"] for s in slow_route} == {"short-slow"}
    assert seconds(plan_route(scenario, fast, 95, 15, 0)) == pytest.approx(8)


def test_interior_edge_destination_and_bounded_lastmile(scenario):
    road(scenario, "long", "a", "b", [[15, 15], [195, 15]])
    engine = unit(scenario, max_access_m=11)
    result = plan_route(scenario, engine, 105, 25, 0)
    assert result[-1]["mode"] == "offroad"
    assert (result[-2]["x"], result[-2]["y"]) == (105, 15)
    assert length(result) == pytest.approx(100)
    assert seconds(result) == pytest.approx(14)
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 105, 35, 0)
    assert line_access_point(scenario, engine, 105, 25, 0) == (105, 15)


def test_oneway_cannot_reverse_or_shortcut_backwards(scenario):
    road(scenario, "oneway", "a", "b", [[15, 15], [195, 15]])
    engine = unit(scenario, x=105, max_access_m=100)
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 35, 15, 0)
    forward = plan_route(scenario, engine, 155, 15, 0)
    assert length(forward) == pytest.approx(50)


def test_return_uses_actual_interior_position_and_reverse_geometry(scenario):
    road(scenario, "out", "a", "b", [[15, 15], [15, 65], [95, 65]])
    road(scenario, "back", "b", "a", [[95, 65], [15, 65], [15, 15]])
    engine = unit(scenario, x=45, y=65)
    result = plan_route(scenario, engine, 15, 15, 4)
    assert (result[0]["start_x"], result[0]["start_y"]) == (45, 65)
    assert [(s["x"], s["y"]) for s in result] == [(15, 65), (15, 15)]
    assert length(result) == pytest.approx(80)
    assert seconds(result) == pytest.approx(8)


def test_oneway_return_takes_directed_loop_from_actual_position(scenario):
    road(scenario, "out", "a", "b", [[15, 15], [95, 15]])
    road(scenario, "return", "b", "a", [[95, 15], [95, 65], [15, 65], [15, 15]])
    engine = unit(scenario, x=55, y=15)
    result = plan_route(scenario, engine, 15, 15, 4)
    assert length(result) == pytest.approx(220)
    assert result[0]["edge_id"] == "out"
    assert result[-1]["edge_id"] == "return"


def test_closure_version_and_speed_change_invalidate_cache(scenario):
    edge = road(scenario, "only", "a", "b", [[15, 15], [95, 15]])
    engine = unit(scenario)
    planned = plan_route(scenario, engine, 95, 15, 0)
    assert route_invalid_reason(scenario, engine, planned, 1) is None
    edge["speed_m_s"] = 5
    assert route_invalid_reason(scenario, engine, planned, 1) is not None
    assert seconds(plan_route(scenario, engine, 95, 15, 1)) == pytest.approx(16)
    edge["closed"] = True
    assert route_invalid_reason(scenario, engine, planned, 1) is not None
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 95, 15, 1)
    edge["closed"] = False
    assert seconds(plan_route(scenario, engine, 95, 15, 2)) == pytest.approx(16)


def test_unit_speed_change_invalidates_existing_route(scenario):
    road(scenario, "only", "a", "b", [[15, 15], [95, 15]])
    engine = unit(scenario)
    planned = plan_route(scenario, engine, 95, 15, 0)
    engine["speed_m_s"] = 2
    assert "speed" in route_invalid_reason(scenario, engine, planned, 1)
    assert seconds(plan_route(scenario, engine, 95, 15, 1)) == pytest.approx(40)


def test_engine_cannot_cross_nodata_or_water_lastmile(scenario):
    road(scenario, "road", "a", "b", [[15, 15], [195, 15]])
    engine = unit(scenario, max_access_m=50)
    scenario.landscape["fuel_model"][18, :] = 98  # y=20..30 water barrier
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 105, 45, 0)
    scenario.landscape["fuel_model"][18, :] = 101
    scenario.landscape["valid"][18, :] = False
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 105, 45, 0)


def test_crew_dem_route_detours_water_and_obeys_distance_bound(scenario):
    crew = unit(scenario, "handcrew", x=15, y=15, max_access_m=150)
    scenario.landscape["fuel_model"][17:21, 4] = 98
    result = plan_route(scenario, crew, 75, 15, 0)
    assert any(segment["y"] >= 45 for segment in result)
    assert length(result) > 60
    assert length(result) <= 150
    crew["max_access_m"] = 70
    with pytest.raises(ValueError):
        plan_route(scenario, crew, 75, 15, 0)


def test_crew_forbids_nodata_slope_and_dem_grade_barriers(scenario):
    crew = unit(scenario, "handcrew", max_access_m=200, max_slope=.5)
    scenario.landscape["valid"][:, 4] = False
    with pytest.raises(ValueError):
        plan_route(scenario, crew, 75, 15, 0)
    scenario.landscape["valid"][:, 4] = True
    scenario.landscape["slope"][:, 4] = 1
    with pytest.raises(ValueError):
        plan_route(scenario, crew, 75, 15, 0)
    scenario.landscape["slope"][:] = 0
    scenario.landscape["elevation"][:, 4:] += 100
    with pytest.raises(ValueError):
        plan_route(scenario, crew, 75, 15, 0)


def test_crew_grade_relation_is_directional(scenario):
    scenario.landscape["elevation"][:] = np.arange(21)[None, :] * 2
    crew = unit(scenario, "handcrew", max_access_m=100, max_slope=.5)
    uphill = plan_route(scenario, crew, 75, 15, 0)
    crew.update(x=75, y=15)
    downhill = plan_route(scenario, crew, 15, 15, 0)
    assert seconds(uphill) > seconds(downhill)
    assert seconds(uphill) > length(uphill) / crew["offroad_speed_m_s"]


def test_straight_line_validation_checks_crossed_cell_not_sparse_samples(scenario):
    crew = unit(scenario, "handcrew")
    scenario.landscape["valid"][19, 4] = False
    with pytest.raises(ValueError, match="nodata"):
        validate_ground_segment(scenario, crew, 15, 15, 195, 15)


def test_air_speed_projects_wind_and_uses_positive_minimum(scenario):
    helicopter = unit(scenario, "helicopter", speed_m_s=10, max_wind_m_s=30,
                      min_ground_speed_m_s=2)
    scenario.weather["wind_speed_m_s"][:] = 20
    scenario.weather["upwind_direction_deg"][:] = 90  # from east, headwind eastbound
    east = plan_route(scenario, helicopter, 95, 15, 0)
    assert seconds(east) == pytest.approx(40)
    helicopter.update(x=95)
    west = plan_route(scenario, helicopter, 15, 15, 0)
    assert seconds(west) == pytest.approx(80 / 30)


def test_air_rejects_wind_closed_corridor_and_no_weather_even_colocated(scenario):
    helicopter = unit(scenario, "helicopter")
    scenario.weather["wind_speed_m_s"][:] = 20
    with pytest.raises(ValueError, match="wind"):
        plan_route(scenario, helicopter, 15, 15, 0)
    scenario.weather["wind_speed_m_s"][:] = 0
    helicopter["air_corridor_available"] = False
    with pytest.raises(ValueError, match="corridor"):
        plan_route(scenario, helicopter, 95, 15, 0)
    helicopter["air_corridor_available"] = True
    with pytest.raises(ValueError, match="coverage"):
        check_air_constraints(scenario, helicopter, 7201, 15, 15)


def test_air_checks_entire_spatial_corridor(scenario):
    helicopter = unit(scenario, "helicopter")
    corridor = np.ones((21, 21), dtype=bool)
    corridor[19, 4] = False
    scenario.landscape["air_corridor_available"] = corridor
    with pytest.raises(ValueError, match="corridor"):
        plan_route(scenario, helicopter, 95, 15, 0)


def test_air_weather_boundary_and_wind_replanning_after_interruption(scenario):
    helicopter = unit(scenario, "helicopter", speed_m_s=10)
    scenario.weather["wind_speed_m_s"][1] = 5
    scenario.weather["upwind_direction_deg"][:] = 270
    result = plan_route(scenario, helicopter, 95, 15, 3596)
    assert seconds(result) == pytest.approx(8.)  # future band is not dispatch knowledge
    first = plan_route(scenario, helicopter, 95, 15, 0)
    helicopter["x"] = 35
    assert route_invalid_reason(scenario, helicopter, first[2:], 2) is None
    scenario.weather["wind_speed_m_s"][0] = 5
    assert "wind" in route_invalid_reason(scenario, helicopter, first[2:], 2)


def test_closed_interior_edge_does_not_snap_to_nearby_open_road(scenario):
    blocked = road(scenario, "blocked", "a", "b", [[15, 15], [195, 15]])
    road(scenario, "nearby", "c", "d", [[15, 25], [195, 25]])
    engine = unit(scenario, x=105, y=15, max_access_m=30)
    blocked["closed"] = True
    with pytest.raises(ValueError, match="unreachable"):
        plan_route(scenario, engine, 195, 25, 0)


def test_air_rejects_spatial_wind_barrier_and_nodata(scenario):
    helicopter = unit(scenario, "helicopter")
    wind = np.zeros((2, 21, 21))
    wind[:, 19, 4] = 25
    scenario.weather["wind_speed_m_s"] = wind
    with pytest.raises(ValueError, match="wind"):
        plan_route(scenario, helicopter, 95, 15, 0)
    scenario.weather["wind_speed_m_s"][:] = 0
    scenario.landscape["valid"][19, 4] = False
    with pytest.raises(ValueError, match="nodata"):
        plan_route(scenario, helicopter, 95, 15, 0)
