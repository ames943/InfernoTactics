"""Weather-series loading and deterministic simulation-time lookup."""

import csv
from datetime import datetime, timezone

import numpy as np


def load_weather_series(path):
    epochs, values = [], []
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            timestamp = datetime.strptime(
                row["timestamp"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            epochs.append(timestamp.timestamp())
            values.append((
                float(row["wind_speed_mph"]),
                float(row["wind_direction_deg"]),
                float(row["humidity_pct"]),
            ))
    if not epochs:
        raise ValueError(f"Weather series is empty: {path}")
    order = np.argsort(epochs)
    return (
        np.asarray(epochs, dtype=np.float64)[order],
        np.asarray(values, dtype=np.float64)[order],
    )


def weather_at(start_utc, elapsed_seconds, weather_epochs, weather_values):
    """Return the latest observation at a simulation time, clamped at ends."""
    target_epoch = start_utc.timestamp() + elapsed_seconds
    index = np.searchsorted(weather_epochs, target_epoch, side="right") - 1
    index = int(np.clip(index, 0, len(weather_epochs) - 1))
    wind_speed, wind_direction, humidity = weather_values[index]
    return float(wind_speed), float(wind_direction), float(humidity)


def synthetic_santa_ana(tick):
    """Historical fixed debugging schedule retained for regression tests."""
    return min(10.0 + 3.5 * tick, 45.0), 45.0, 8.0
