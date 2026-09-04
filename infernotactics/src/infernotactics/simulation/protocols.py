"""Structural interface required by operational response mechanics."""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FireEngine(Protocol):
    height: int
    width: int
    state: np.ndarray
    ignitability: np.ndarray
    building_density: np.ndarray

    def ignite(self, row: int, col: int, radius: int = 0) -> None: ...

    def step(
        self,
        wind_speed_mph: float = 0.0,
        wind_direction_deg: float = 0.0,
        humidity_pct: float = 30.0,
    ) -> None: ...

    def state_counts(self) -> dict[str, int]: ...
