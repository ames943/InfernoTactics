"""Reusable immutable routing preparation shared by simulation sessions."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingContext:
    """Expensive, read-only routing products for one world/station roster."""

    road_graph: Any
    world_fingerprint: str
    zone_road_nodes: tuple[int, ...]
    stations: tuple[dict, ...]
    stations_by_type: dict[str, tuple[str, ...]]
    station_travel_time_s: dict[str, tuple[float, ...]]
    zone_travel_time_s: dict[str, tuple[float, ...]]
