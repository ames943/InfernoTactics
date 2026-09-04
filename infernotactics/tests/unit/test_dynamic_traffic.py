import networkx as nx
import pytest

from infernotactics.operations import DynamicTrafficRouter


def _router(*, mode="synthetic"):
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, key=0, travel_time=10.0, highway="residential")
    graph.add_edge(1, 2, key=1, travel_time=20.0, highway="primary")
    graph.add_edge(2, 3, key=0, travel_time=10.0, highway="residential")
    return DynamicTrafficRouter(
        road_graph=graph,
        stations=[{"station_id": "s1", "road_node": 1}],
        zone_road_nodes=[3, 99],
        traffic_mode=mode,
        tick_duration_minutes=2,
        road_multipliers={"residential": 1.2, "primary": 1.0, "unclassified": 1.3},
        road_capacities={"residential": 1.0, "primary": 2.0, "unclassified": 1.0},
        resource_weights={"water_team": 1.0, "helicopter": 0.0},
        bpr_alpha=0.15,
        bpr_beta=4.0,
    )


def test_router_selects_effective_shortest_path_and_returns_edges():
    router = _router()

    travel_s, edges = router.route("s1", 0, tick=0)

    assert travel_s == pytest.approx(26.4)
    assert edges == [(1, 2, 0), (2, 3, 0)]


def test_route_occupancy_increases_congestion_and_can_be_released():
    router = _router()
    base, edges = router.route("s1", 0, tick=0)

    router.reserve("water_team", edges)
    congested, _ = router.route("s1", 0, tick=0)
    assert congested > base
    assert sum(router.edge_resource_load.values()) == 2.0

    router.release("water_team", edges)
    assert router.edge_resource_load == {}
    assert router.route("s1", 0, tick=0)[0] == pytest.approx(base)


def test_legacy_mode_uses_free_flow_and_unreachable_zone_is_safe():
    router = _router(mode="legacy")

    assert router.route("s1", 0, tick=0)[0] == 20.0
    travel_s, edges = router.route("s1", 1, tick=0)
    assert travel_s == float("inf")
    assert edges == []


def test_air_resources_do_not_reserve_road_capacity():
    router = _router()
    _, edges = router.route("s1", 0, tick=0)
    router.reserve("helicopter", edges)
    assert router.edge_resource_load == {}
