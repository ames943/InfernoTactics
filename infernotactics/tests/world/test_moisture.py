"""Offline contract tests; native integration requires the pinned compiled CLI."""
import pytest
from datetime import datetime, timedelta, timezone

from src.world.moisture import _fw21_timestamp, _station_timezone

from src.world.moisture import MOISTURE_FIELDS, parse_moisture_csv, run_nfdrs4, sample_moisture_prior


def manifest():
    return {"moisture_priors": {name: {"distribution": "uniform", "low": 0.05,
            "high": 0.3, "provenance": "explicit sensitivity interval, not observations"}
            for name in MOISTURE_FIELDS}}


def test_missing_history_requires_explicit_prior():
    with pytest.raises(ValueError, match="interval prior required"):
        run_nfdrs4([], start_utc="2025-01-01T00:00:00Z", end_utc="2025-01-01T02:00:00Z",
                   station={}, binary="not-a-binary", attestation={}, cache_dir="unused", foliar=None)


def test_missing_history_samples_prior_without_executable(tmp_path):
    result = run_nfdrs4([], start_utc="2025-01-01T00:00:00Z", end_utc="2025-01-01T02:00:00Z",
                        station={}, binary="not-a-binary", attestation={}, cache_dir=tmp_path,
                        foliar=None, uncertainty_manifest=manifest(), seed=4)
    assert result["provenance"]["moisture_source"] == "assumed_prior"
    assert result["provenance"]["historical_readiness_certified"] is False
    assert set(result["fields"]) == set(MOISTURE_FIELDS)
    assert all(len(values) == 2 and all(0.05 <= v <= 0.3 for v in values)
               for values in result["fields"].values())
    assert result["fields"] == sample_moisture_prior(manifest(), 2, seed=4)["fields"]
    assert result["fields"] != sample_moisture_prior(manifest(), 2, seed=5)["fields"]


def test_prior_rejects_missing_foliar_and_point_estimate():
    prior = manifest()
    del prior["moisture_priors"]["foliar_moisture"]
    with pytest.raises(ValueError, match="foliar_moisture"):
        sample_moisture_prior(prior, 1)
    prior = manifest()
    prior["moisture_priors"]["foliar_moisture"]["high"] = 0.05
    with pytest.raises(ValueError, match="nonempty interval"):
        sample_moisture_prior(prior, 1)


def test_real_upstream_csv_schema_percent_conversion():
    # Exact pinned FW21 field names, not a subprocess echo/mocked executable.
    text = "StationID,DateTime,1HourDFM(%),10HourDFM(%),100HourDFM(%),1000HourDFM(%),HerbLFM(%),WoodyLFM(%)\nS,2025-01-01T00:00:00Z,5,7,9,12,120,85\n"
    result = next(iter(parse_moisture_csv(text, station_id="S").values()))
    assert result == {"fuel_moisture_dead_1hr": .05, "fuel_moisture_dead_10hr": .07,
                      "fuel_moisture_dead_100hr": .09, "fuel_moisture_live_herbaceous": 1.2,
                      "fuel_moisture_live_woody": .85}
    assert "foliar_moisture" not in result
    with pytest.raises(ValueError, match="another station"):
        parse_moisture_csv(text, station_id="other")
    with pytest.raises(ValueError, match="required columns"):
        parse_moisture_csv("DateTime,HerbLFM(%)\n", station_id="S")
    with pytest.raises(ValueError, match="invalid"):
        parse_moisture_csv(text.replace(",5,7,", ",nan,7,"), station_id="S")


def test_local_fw21_daywrap_preserves_every_utc_hour():
    zone = _station_timezone('timeZoneOffset = "-6";')
    start = datetime(2024, 12, 31, 20, tzinfo=timezone.utc)
    expected = [start + timedelta(hours=i) for i in range(36)]
    transported = [_fw21_timestamp(stamp, zone) for stamp in expected]
    assert transported[0] == "2024-12-31T14:00:00-06:00"
    assert transported[10] == "2025-01-01T00:00:00-06:00"
    # Pinned DateToOriginal emits compact local ISO dates and retains the offset.
    header = "StationID,DateTime,1HourDFM(%),10HourDFM(%),100HourDFM(%),HerbLFM(%),WoodyLFM(%)\n"
    rows = [f'S,{stamp.replace("-", "", 2).replace(":", "", 2)},5,7,9,120,85\n'
            for stamp in transported]
    assert list(parse_moisture_csv(header + "".join(rows), station_id="S")) == expected


@pytest.mark.parametrize("config", ["", 'timeZoneOffset = "-6.5";',
    'timeZoneOffset = "-24";', 'timeZoneOffset = "-6";\ntimeZoneOffset = "-7";'])
def test_station_timezone_requires_one_unambiguous_integer(config):
    with pytest.raises(ValueError, match="timeZoneOffset"):
        _station_timezone(config)
