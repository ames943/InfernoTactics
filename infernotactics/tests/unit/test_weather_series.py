from datetime import datetime, timezone

import numpy as np

from infernotactics.operations.weather import synthetic_santa_ana, weather_at


def test_weather_lookup_is_stepwise_and_clamped():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    base = start.timestamp()
    epochs = np.array([base, base + 60, base + 120])
    values = np.array([[10, 20, 30], [11, 21, 31], [12, 22, 32]])

    assert weather_at(start, -10, epochs, values) == (10.0, 20.0, 30.0)
    assert weather_at(start, 90, epochs, values) == (11.0, 21.0, 31.0)
    assert weather_at(start, 999, epochs, values) == (12.0, 22.0, 32.0)


def test_synthetic_schedule_caps_wind_speed():
    assert synthetic_santa_ana(0) == (10.0, 45.0, 8.0)
    assert synthetic_santa_ana(100)[0] == 45.0
