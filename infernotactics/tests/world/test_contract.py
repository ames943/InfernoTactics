"""Consumer-visible validation and safe on-disk scenario contract checks."""

import copy
import json

import numpy as np
import pytest

from src.world.scenario import load_scenario, save_scenario
from src.world.synthetic import make_synthetic


@pytest.fixture
def scenario():
    return make_synthetic(seed=7, size=9, duration_s=7200)


def test_synthetic_defaults_and_dimensions():
    scenario = make_synthetic()
    assert scenario.landscape["valid"].shape == (96, 96)
    assert scenario.duration_s == 21600
    assert scenario.transform[0] == 30.0
    assert scenario.transform[4] == -30.0
    assert scenario.crs == "EPSG:5070"
    assert np.all(scenario.landscape["fuel_model"] == 101)
    assert np.all(scenario.landscape["canopy_height"] == 0)
    assert {unit["kind"] for unit in scenario.resources} == {"engine", "handcrew", "helicopter"}
    assert len({unit["base_id"] for unit in scenario.resources}) == 1
    assert scenario.provenance["scenario_kind"] == "synthetic"
    assert scenario.provenance["suppression_validation"] == "experimental"
    assert len(scenario.provenance["weather_intervals"]) == 6
    scenario.validate()


def test_factory_is_deterministic_and_owns_arrays(scenario):
    other = make_synthetic(seed=7, size=9, duration_s=7200)
    assert scenario.provenance == other.provenance
    assert scenario.resources == other.resources
    assert scenario.roads == other.roads
    for name, array in scenario.landscape.items():
        np.testing.assert_array_equal(array, other.landscape[name])
        assert not np.shares_memory(array, other.landscape[name])
    for name, array in scenario.weather.items():
        np.testing.assert_array_equal(array, other.weather[name])


def test_partial_final_hour_has_explicit_coverage():
    scenario = make_synthetic(duration_s=3601, size=5)
    assert scenario.weather["wind_speed_m_s"].shape == (2, 1, 1)
    assert scenario.provenance["weather_intervals"][-1]["end_utc"] == "2025-01-01T02:00:00Z"
    scenario.validate()


@pytest.mark.parametrize("field,value", [
    ("crs", "EPSG:4326"),
    ("transform", (30.0, 1.0, 0.0, 0.0, -30.0, 0.0)),
    ("transform", (30.0, 0.0, 0.0, 0.0, 30.0, 0.0)),
    ("transform", (30.0, 0.0, 0.0, 0.0, -60.0, 0.0)),
    ("start_utc", "2025-01-01T00:00:00"),
    ("duration_s", 0),
])
def test_rejects_invalid_scalar_contract(scenario, field, value):
    setattr(scenario, field, value)
    with pytest.raises(ValueError):
        scenario.validate()


@pytest.mark.parametrize("key,value", [
    ("slope", np.zeros((2, 2))),
    ("elevation", np.full((9, 9), np.nan)),
    ("valid", np.ones((9, 9), dtype=np.int8)),
    ("fuel_model", np.full((9, 9), 101.0)),
    ("canopy_cover", np.full((9, 9), 2.0)),
])
def test_rejects_invalid_landscape(scenario, key, value):
    scenario.landscape[key] = value
    with pytest.raises(ValueError):
        scenario.validate()


def test_missing_required_layer_is_rejected(scenario):
    del scenario.landscape["elevation"]
    with pytest.raises(ValueError):
        scenario.validate()


def test_nodata_remains_explicitly_masked(scenario):
    scenario.landscape["valid"][0, 0] = False
    scenario.landscape["elevation"][0, 0] = np.nan
    scenario.validate()
    assert not scenario.landscape["valid"][0, 0]
    assert np.isnan(scenario.landscape["elevation"][0, 0])


@pytest.mark.parametrize("mutation", ["missing", "short", "nonfinite", "direction", "negative_moisture", "gap"])
def test_rejects_invalid_weather(scenario, mutation):
    if mutation == "missing":
        del scenario.weather["foliar_moisture"]
    elif mutation == "short":
        scenario.weather = {key: value[:1] for key, value in scenario.weather.items()}
        scenario.provenance["weather_intervals"] = scenario.provenance["weather_intervals"][:1]
    elif mutation == "nonfinite":
        scenario.weather["temperature_c"][0, 0, 0] = np.nan
    elif mutation == "direction":
        scenario.weather["upwind_direction_deg"][0, 0, 0] = 360.0
    elif mutation == "negative_moisture":
        scenario.weather["fuel_moisture_dead_1hr"][0, 0, 0] = -0.1
    else:
        scenario.provenance["weather_intervals"][1]["start_utc"] = "2025-01-01T01:01:00Z"
    with pytest.raises(ValueError):
        scenario.validate()


@pytest.mark.parametrize("mutation", ["duplicate_unit", "missing_source", "bad_edge", "bad_geometry", "negative_speed"])
def test_rejects_invalid_operations_inputs(scenario, mutation):
    if mutation == "duplicate_unit":
        scenario.resources.append(copy.deepcopy(scenario.resources[0]))
    elif mutation == "missing_source":
        scenario.resources[0]["parameter_sources"] = {}
    elif mutation == "bad_edge":
        scenario.roads["edges"][0]["target"] = "missing-node"
    elif mutation == "bad_geometry":
        scenario.roads["edges"][0]["geometry"][0] = [0.0, 0.0]
    else:
        scenario.roads["edges"][0]["speed_m_s"] = -1.0
    with pytest.raises(ValueError):
        scenario.validate()


def test_scenario_roundtrip(scenario, tmp_path):
    path = tmp_path / "scenario"
    save_scenario(scenario, path)
    restored = load_scenario(path)
    for name in ("name", "crs", "transform", "start_utc", "duration_s", "weather_step_s", "seed",
                 "provenance", "roads", "resources"):
        assert getattr(restored, name) == getattr(scenario, name)
    for collection in ("landscape", "weather"):
        before, after = getattr(scenario, collection), getattr(restored, collection)
        assert before.keys() == after.keys()
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
            assert before[key].dtype == after[key].dtype
    manifest = json.loads((path / "scenario.json").read_text())
    assert manifest["schema_version"] == 1
    for filename in ("landscape.npz", "weather.npz"):
        with np.load(path / filename, allow_pickle=False) as arrays:
            assert arrays.files
            for key in arrays.files:
                assert not arrays[key].dtype.hasobject


def test_save_rejects_overwrite(scenario, tmp_path):
    path = tmp_path / "scenario"
    save_scenario(scenario, path)
    with pytest.raises((ValueError, FileExistsError)):
        save_scenario(scenario, path)


def test_load_rejects_incompatible_version(scenario, tmp_path):
    path = tmp_path / "scenario"
    save_scenario(scenario, path)
    manifest_path = path / "scenario.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 999
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_scenario(path)


def test_load_rejects_pickle_arrays(scenario, tmp_path):
    path = tmp_path / "scenario"
    save_scenario(scenario, path)
    np.savez(path / "landscape.npz", elevation=np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError):
        load_scenario(path)


@pytest.mark.parametrize("kwargs", [{"size": 0}, {"duration_s": 0}, {"cell_size_m": float("nan")}, {"seed": -1}])
def test_factory_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        make_synthetic(**kwargs)


def test_load_rejects_array_path_escape(scenario, tmp_path):
    path = tmp_path / "scenario"
    save_scenario(scenario, path)
    manifest_path = path / "scenario.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["landscape"] = "../outside.npz"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_scenario(path)
