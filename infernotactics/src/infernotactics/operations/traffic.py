"""Dynamic road routing and emergency-vehicle congestion."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import networkx as nx


class DynamicTrafficRouter:
    """Calculate tick-specific routes and track active route occupancy."""

    def __init__(
        self,
        *,
        road_graph,
        stations: Sequence[Mapping[str, Any]],
        zone_road_nodes: Sequence[int],
        traffic_mode: str,
        tick_duration_minutes: float,
        road_multipliers: Mapping[str, float],
        road_capacities: Mapping[str, float],
        resource_weights: Mapping[str, float],
        bpr_alpha: float,
        bpr_beta: float,
        simulation_start_hour: float = 10.5,
    ):
        if traffic_mode not in {"legacy", "synthetic"}:
            raise ValueError("traffic_mode must be 'legacy' or 'synthetic'")
        self.road_graph = road_graph
        self.stations_by_id = {str(station["station_id"]): station for station in stations}
        self.zone_road_nodes = tuple(int(node) for node in zone_road_nodes)
        self.traffic_mode = traffic_mode
        self.tick_duration_minutes = float(tick_duration_minutes)
        self.road_multipliers = dict(road_multipliers)
        self.road_capacities = dict(road_capacities)
        self.resource_weights = dict(resource_weights)
        self.bpr_alpha = float(bpr_alpha)
        self.bpr_beta = float(bpr_beta)
        self.simulation_start_hour = float(simulation_start_hour)
        self.edge_resource_load: dict[tuple[Any, Any, Any], float] = {}

    def reset(self) -> None:
        self.edge_resource_load.clear()

    @staticmethod
    def road_class(data: Mapping[str, Any]) -> str:
        highway = data.get("highway", "unclassified")
        if isinstance(highway, (list, tuple)):
            highway = highway[0] if highway else "unclassified"
        return str(highway).lower()

    def background_factor(self, tick: int) -> float:
        """Deterministic time-of-day traffic profile."""
        hour = (
            self.simulation_start_hour
            + tick * self.tick_duration_minutes / 60.0
        ) % 24.0
        if 7.0 <= hour < 9.5 or 16.0 <= hour < 19.0:
            return 1.25
        if 11.0 <= hour < 14.0:
            return 1.05
        return 1.10

    def edge_effective_time(self, u, v, data, *, tick: int) -> float:
        """NetworkX weight callback with road class and active-unit load."""
        if "travel_time" in data:
            base_time = float(data["travel_time"])
            edge_data = data
            edge_key = data.get("key")
        else:
            candidates = [
                self.edge_effective_time(u, v, {**attrs, "key": key}, tick=tick)
                for key, attrs in data.items()
                if isinstance(attrs, dict)
            ]
            return min(candidates, default=math.inf)

        if self.traffic_mode == "legacy":
            return base_time
        road_class = self.road_class(edge_data)
        background = self.road_multipliers.get(
            road_class, self.road_multipliers["unclassified"]
        ) * self.background_factor(tick)
        load = (
            self.edge_resource_load.get((u, v, edge_key), 0.0)
            if edge_key is not None
            else 0.0
        )
        capacity = self.road_capacities.get(
            road_class, self.road_capacities["unclassified"]
        )
        congestion = 1.0 + self.bpr_alpha * (load / max(capacity, 1.0)) ** self.bpr_beta
        return base_time * background * congestion

    def route_edges_from_nodes(self, nodes: Sequence[Any], *, tick: int) -> list[tuple[Any, Any, Any]]:
        edges = []
        for u, v in zip(nodes, nodes[1:]):
            records = self.road_graph.get_edge_data(u, v) or {}
            edge_key, _ = min(
                records.items(),
                key=lambda item: self.edge_effective_time(
                    u, v, {**item[1], "key": item[0]}, tick=tick
                ),
                default=(None, None),
            )
            if edge_key is not None:
                edges.append((u, v, edge_key))
        return edges

    def route(self, station_id: str, target_zone_id: int, *, tick: int):
        station = self.stations_by_id[station_id]
        source = int(station["road_node"])
        target = self.zone_road_nodes[target_zone_id]
        weight = lambda u, v, data: self.edge_effective_time(u, v, data, tick=tick)
        try:
            travel_s, nodes = nx.single_source_dijkstra(
                self.road_graph, source, target, weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf, []
        return float(travel_s), self.route_edges_from_nodes(nodes, tick=tick)

    def reserve(self, resource_type: str, route_edges: Sequence[tuple[Any, Any, Any]]) -> None:
        weight = self.resource_weights[resource_type]
        if weight <= 0:
            return
        for edge in route_edges:
            self.edge_resource_load[edge] = self.edge_resource_load.get(edge, 0.0) + weight

    def release(self, resource_type: str, route_edges: Sequence[tuple[Any, Any, Any]]) -> None:
        weight = self.resource_weights[resource_type]
        if weight <= 0:
            return
        for edge in route_edges:
            remaining = self.edge_resource_load.get(edge, 0.0) - weight
            if remaining > 1e-9:
                self.edge_resource_load[edge] = remaining
            else:
                self.edge_resource_load.pop(edge, None)
