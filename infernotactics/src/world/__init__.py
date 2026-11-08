"""Versioned, georeferenced wildfire-world contracts.

This package defines data interchange, not a claim of physical or operational
validation. Simulation engines consume these contracts without policy mutation.
"""

from .contracts import Action, Event, Scenario, Treatment
from .scenario import load_scenario, save_scenario
from .world import World

__all__ = ["Action", "Event", "Scenario", "Treatment", "World", "load_scenario", "save_scenario"]
