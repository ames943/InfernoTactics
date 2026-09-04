"""Fire-spread engine implementations and contracts."""

from .fire_sim import FireSim
from .protocols import FireEngine

__all__ = ["FireEngine", "FireSim"]
