"""Shared domain definitions used across data, simulation, and decision code."""

from .actions import DispatchAction, parse_dispatch_actions
from .fire import ACTIVE_FIRE_STATES, BLAZE, BURNED_OUT, FUEL, SAFE, STATE_NAMES, THREAT
from .resources import AIR_RESOURCE_TYPES, GROUND_RESOURCE_TYPES, RESOURCE_TYPES
from .observations import SCALAR_KEYS, flatten_scalars
from .weather import meteorological_wind_to_cartesian, meteorological_wind_to_grid
from .zones import ZoneBounds, build_zone_boundaries, zone_id_for_cell

__all__ = [
    "DispatchAction",
    "AIR_RESOURCE_TYPES",
    "ACTIVE_FIRE_STATES",
    "BLAZE",
    "BURNED_OUT",
    "FUEL",
    "GROUND_RESOURCE_TYPES",
    "RESOURCE_TYPES",
    "SCALAR_KEYS",
    "SAFE",
    "STATE_NAMES",
    "THREAT",
    "ZoneBounds",
    "build_zone_boundaries",
    "flatten_scalars",
    "meteorological_wind_to_cartesian",
    "meteorological_wind_to_grid",
    "parse_dispatch_actions",
    "zone_id_for_cell",
]
