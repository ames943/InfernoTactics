"""Resource dispatch and lifecycle state machine.

The fleet deliberately knows nothing about fire physics, map arrays, or the
reinforcement-learning environment.  Routing and on-arrival effects enter as
callbacks, which keeps this state machine deterministic and independently
testable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


RouteEdge = tuple[Any, Any, Any]
DynamicRoute = Callable[[str, int], tuple[float, list[RouteEdge]]]
RouteOccupancy = Callable[[str, Sequence[RouteEdge]], None]
ApplyEffect = Callable[[str, int], tuple[int, int, int]]


def new_resource_unit(station_id: str) -> dict[str, Any]:
    """Return the stable, serializable state record used for one unit."""
    return {
        "state": "available",
        "remaining_ticks": 0,
        "target_zone": None,
        "station_id": station_id,
        "route_edges": [],
        "travel_time_s": 0.0,
        "traffic_delay_s": 0.0,
        "dispatch_tick": None,
        "arrival_tick": None,
        "effect_tick": None,
        "available_again_tick": None,
    }


class ResourceFleet:
    """Own resource inventory, dispatch selection, and unit transitions."""

    def __init__(
        self,
        *,
        resource_types: Iterable[str],
        ground_resource_types: Iterable[str],
        stations: Sequence[Mapping[str, Any]],
        stations_by_type: Mapping[str, Sequence[str]],
        station_travel_time_s: Mapping[str, Sequence[float]],
        delay_config: Mapping[str, Mapping[str, int]],
        traffic_mode: str,
        tick_duration_seconds: float,
        dynamic_ground_route: DynamicRoute,
        reserve_route: RouteOccupancy,
        release_route: RouteOccupancy,
        apply_effect: ApplyEffect,
        travel_penalty_per_second: float,
        response_delay_penalty_per_second: float,
        wasted_penalty: float,
        suppression_reward: float,
        legacy_busy_ticks: Mapping[str, int],
    ):
        self.resource_types = tuple(resource_types)
        self.ground_resource_types = frozenset(ground_resource_types)
        self.stations = stations
        self.stations_by_type = stations_by_type
        self.station_travel_time_s = station_travel_time_s
        self.delay_config = {
            resource_type: dict(delay_config[resource_type])
            for resource_type in self.resource_types
        }
        self.traffic_mode = traffic_mode
        self.tick_duration_seconds = float(tick_duration_seconds)
        self.dynamic_ground_route = dynamic_ground_route
        self.reserve_route = reserve_route
        self.release_route = release_route
        self.apply_effect = apply_effect
        self.travel_penalty_per_second = float(travel_penalty_per_second)
        self.response_delay_penalty_per_second = float(response_delay_penalty_per_second)
        self.wasted_penalty = float(wasted_penalty)
        self.suppression_reward = float(suppression_reward)
        self.legacy_busy_ticks = dict(legacy_busy_ticks)
        self.resources: dict[str, list[dict[str, Any]]] = {}
        self.reset()

    def reset(self) -> dict[str, list[dict[str, Any]]]:
        """Restore every station's full roster to an available state."""
        self.resources = {resource_type: [] for resource_type in self.resource_types}
        for station in self.stations:
            station_id = str(station["station_id"])
            for resource_type, count in station["roster"].items():
                self.resources[resource_type].extend(
                    new_resource_unit(station_id) for _ in range(int(count))
                )
        return self.resources

    def try_dispatch(self, resource_type: str, target_zone_id: int, *, tick: int) -> dict[str, Any]:
        """Assign the fastest currently available, reachable unit."""
        station_ids = self.stations_by_type.get(resource_type, ())
        roster = self.resources[resource_type]
        best_unit = None
        best_travel_s = math.inf
        best_free_flow_s = math.inf
        best_route: list[RouteEdge] = []

        for unit in roster:
            if unit["state"] != "available":
                continue
            station_id = unit["station_id"]
            if resource_type in self.ground_resource_types and self.traffic_mode == "synthetic":
                travel_s, route_edges = self.dynamic_ground_route(station_id, target_zone_id)
                free_flow_s = self.station_travel_time_s[station_id][target_zone_id]
            else:
                travel_s = self.station_travel_time_s[station_id][target_zone_id]
                route_edges = []
                free_flow_s = travel_s
            if math.isfinite(travel_s) and travel_s < best_travel_s:
                best_unit = unit
                best_travel_s = float(travel_s)
                best_free_flow_s = float(free_flow_s)
                best_route = list(route_edges)

        if best_unit is None:
            zone_reachable = any(
                math.isfinite(self.station_travel_time_s[station_id][target_zone_id])
                for station_id in station_ids
            )
            return {
                "resource_type": resource_type,
                "target_zone": target_zone_id,
                "status": "zone_unreachable" if not zone_reachable else "no_unit_available",
                "reward_delta": -self.wasted_penalty,
            }

        eta_ticks = max(1, math.ceil(best_travel_s / self.tick_duration_seconds))
        delay = self.delay_config[resource_type]
        dispatch_delay_ticks = delay["dispatch_delay_ticks"] if self.traffic_mode == "synthetic" else 0
        best_unit.update(
            state="preparing" if dispatch_delay_ticks else "traveling",
            remaining_ticks=dispatch_delay_ticks or eta_ticks,
            target_zone=target_zone_id,
            route_edges=best_route,
            travel_time_s=best_travel_s,
            traffic_delay_s=max(0.0, best_travel_s - best_free_flow_s),
            pending_travel_ticks=eta_ticks,
            dispatch_tick=tick,
        )
        if not dispatch_delay_ticks and resource_type in self.ground_resource_types \
                and self.traffic_mode == "synthetic":
            self.reserve_route(resource_type, best_route)

        return {
            "resource_type": resource_type,
            "target_zone": target_zone_id,
            "status": "dispatched",
            "travel_time_s": best_travel_s,
            "traffic_delay_s": best_unit["traffic_delay_s"],
            "response_delay_ticks": dispatch_delay_ticks,
            "eta_ticks": dispatch_delay_ticks + eta_ticks,
            "station_id": best_unit["station_id"],
            "reward_delta": (
                -self.travel_penalty_per_second * best_travel_s
                -self.response_delay_penalty_per_second
                * dispatch_delay_ticks
                * self.tick_duration_seconds
            ),
        }

    def advance(self, *, tick: int) -> tuple[float, list[dict[str, Any]]]:
        """Advance all busy units by one tick and execute due effects."""
        reward = 0.0
        events: list[dict[str, Any]] = []
        for resource_type in self.resource_types:
            for unit in self.resources[resource_type]:
                if unit["state"] == "available":
                    continue
                unit["remaining_ticks"] -= 1
                if unit["remaining_ticks"] > 0:
                    continue

                if unit["state"] == "preparing":
                    unit["state"] = "traveling"
                    unit["remaining_ticks"] = unit["pending_travel_ticks"]
                    if resource_type in self.ground_resource_types and self.traffic_mode == "synthetic":
                        self.reserve_route(resource_type, unit["route_edges"])
                    continue

                if unit["state"] == "traveling":
                    if resource_type in self.ground_resource_types and self.traffic_mode == "synthetic":
                        self.release_route(resource_type, unit["route_edges"])
                    unit["arrival_tick"] = tick
                    setup_ticks = (
                        self.delay_config[resource_type]["arrival_setup_delay_ticks"]
                        if self.traffic_mode == "synthetic"
                        else 0
                    )
                    if setup_ticks:
                        unit["state"] = "arrival_setup"
                        unit["remaining_ticks"] = setup_ticks
                        continue

                if unit["state"] in {"traveling", "arrival_setup"}:
                    affected, row, col = self.apply_effect(resource_type, unit["target_zone"])
                    success = affected > 0
                    if success and resource_type in {"water_team", "helicopter"}:
                        reward += self.suppression_reward
                    elif not success:
                        reward -= self.wasted_penalty
                    events.append({
                        "resource_type": resource_type,
                        "zone": unit["target_zone"],
                        "cells_affected": affected,
                        "success": success,
                        "row": int(row),
                        "col": int(col),
                    })
                    unit["effect_tick"] = tick
                    unit["state"] = "deployed"
                    unit["remaining_ticks"] = (
                        self.delay_config[resource_type]["post_effect_busy_ticks"]
                        if self.traffic_mode == "synthetic"
                        else self.legacy_busy_ticks[resource_type]
                    )
                    continue

                if unit["state"] == "deployed":
                    unit["state"] = "available"
                    unit["target_zone"] = None
                    unit["route_edges"] = []
                    unit["available_again_tick"] = tick

        return reward, events
