"""Weather-vector conversions shared by simulation and decision code."""

import math


def meteorological_wind_to_cartesian(direction_from_deg: float) -> tuple[float, float]:
    """Convert meteorological wind direction to an east/north travel vector.

    Meteorological direction describes where wind comes *from*. The returned
    unit vector describes where it travels *toward*.
    """
    direction_to_deg = (float(direction_from_deg) + 180.0) % 360.0
    theta = math.radians(direction_to_deg)
    return math.sin(theta), math.cos(theta)


def meteorological_wind_to_grid(direction_from_deg: float) -> tuple[float, float]:
    """Return the downwind unit vector as ``(row_delta, col_delta)``."""
    east, north = meteorological_wind_to_cartesian(direction_from_deg)
    return -north, east
