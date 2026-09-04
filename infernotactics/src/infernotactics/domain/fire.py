"""Shared fire-state vocabulary independent of a simulator implementation."""


SAFE = 0
FUEL = 1
THREAT = 2
BLAZE = 3
BURNED_OUT = 4

STATE_NAMES = ("Safe", "Fuel", "Threat", "Blaze", "Burned Out")
STATE_COLORS = ("#e8e8e8", "#2ca02c", "#ff7f0e", "#d62728", "#3a2a20")
ACTIVE_FIRE_STATES = (THREAT, BLAZE)
