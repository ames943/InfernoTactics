"""Pinned NFDRS4 FW21 adapter; moisture fractions are kg water / kg dry fuel.

The executable must be built from firelab/NFDRS4 commit below. Build attestations
are supplied by the caller, not inferred from a filename. Elevation documents the
native weather location (NFDRS4's initialization format has no elevation field).
The model output is not observational or historical-readiness certification.
"""
from __future__ import annotations

import calendar
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

import re
SOURCE_COMMIT = "84223b95fd395a62ea344257f80a9949cee71e6e"
OUTPUT_FIELDS = {
    "1HourDFM(%)": "fuel_moisture_dead_1hr",
    "10HourDFM(%)": "fuel_moisture_dead_10hr",
    "100HourDFM(%)": "fuel_moisture_dead_100hr",
    "HerbLFM(%)": "fuel_moisture_live_herbaceous",
    "WoodyLFM(%)": "fuel_moisture_live_woody",
}
MOISTURE_FIELDS = (*OUTPUT_FIELDS.values(), "foliar_moisture")


def _utc(value):
    date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if date.tzinfo is None or date.utcoffset() != timedelta(0):
        raise ValueError("moisture timestamps must be UTC")
    return date


def _station_timezone(init_config):
    assignments = re.findall(r'^\s*timeZoneOffset\s*=\s*"([^"\n]*)"\s*;', init_config, re.MULTILINE)
    if len(assignments) != 1 or not re.fullmatch(r'[+-]?\d+', assignments[0]):
        raise ValueError("init_config requires exactly one integer timeZoneOffset")
    offset = int(assignments[0])
    if not -23 <= offset <= 23:
        raise ValueError("timeZoneOffset must be between -23 and 23 hours")
    return timezone(timedelta(hours=offset))


def _fw21_timestamp(stamp, station_timezone):
    # Explicit offsets bypass pinned upstream's faulty UTC day-wrap arithmetic.
    return stamp.astimezone(station_timezone).isoformat(timespec="seconds")


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value, name, minimum=0):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def sample_moisture_prior(uncertainty_manifest, count, *, seed=0):
    """Sample explicitly declared uniform intervals; never silently use midpoints."""
    priors = (uncertainty_manifest or {}).get("moisture_priors", {})
    rng = np.random.default_rng(seed)
    fields = {}
    for name in MOISTURE_FIELDS:
        prior = priors.get(name, {})
        if prior.get("distribution") != "uniform" or not prior.get("provenance"):
            raise ValueError(f"explicit provenance-tagged uniform interval prior required: {name}")
        low = _finite(prior.get("low"), name)
        high = _finite(prior.get("high"), name)
        if high <= low:
            raise ValueError(f"nonempty interval required: {name}")
        fields[name] = rng.uniform(low, high, count).tolist()
    return {"fields": fields, "provenance": {
        "moisture_source": "assumed_prior", "historical_readiness_certified": False,
        "uncertainty_manifest": uncertainty_manifest, "seed": seed,
    }}


def parse_moisture_csv(text, *, station_id):
    """Parse the pinned CLI's real fuelMoisturesOutputFile, including percent units."""
    reader = csv.DictReader(io.StringIO(text))
    required = {"StationID", "DateTime", *OUTPUT_FIELDS}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("NFDRS4 moisture CSV is missing required columns")
    result = {}
    for row in reader:
        if row["StationID"] != station_id:
            raise ValueError("NFDRS4 returned another station")
        stamp = datetime.fromisoformat(row["DateTime"].replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("NFDRS4 output timestamp requires an explicit offset")
        stamp = stamp.astimezone(timezone.utc)
        if stamp in result:
            raise ValueError("duplicate NFDRS4 output timestamp")
        result[stamp] = {target: _finite(row[source], source) / 100
                         for source, target in OUTPUT_FIELDS.items()}
    return result


def run_nfdrs4(weather, *, start_utc, end_utc, station, binary, attestation,
               cache_dir, foliar, uncertainty_manifest=None, seed=0):
    """Execute one native-cell/vegetation run, caching complete reproducible inputs.

    ``station.init_config`` is the complete upstream NFDRSInit configuration,
    including vegetation/GSI parameters, matching latitude and one integer
    timeZoneOffset. Station identity is passed in the run configuration. UTC
    weather is transported as explicit local-offset FW21 timestamps to bypass
    the pinned CLI's UTC output day-wrap bug; output is normalized back to UTC.
    ``foliar`` contains value (fraction), provenance, start_utc and end_utc;
    it is never taken from HerbLFM.
    Missing history or foliar requires all six explicit ensemble priors.
    """
    start, end = _utc(start_utc), _utc(end_utc)
    if end <= start or start.minute or start.second or start.microsecond:
        raise ValueError("positive hour-aligned moisture interval required")
    hours = (end - start).total_seconds() / 3600
    if not hours.is_integer():
        raise ValueError("moisture interval must contain whole hours")
    count = int(hours)
    prior_year = start.year - 1
    history_start = start.replace(year=prior_year, day=min(start.day, calendar.monthrange(prior_year, start.month)[1]))
    rows = list(weather or [])
    times = [_utc(row["time_utc"]) for row in rows]
    required_times = [history_start + timedelta(hours=i) for i in range(int((end-history_start).total_seconds()/3600))]
    valid_foliar = bool(foliar and foliar.get("provenance") and
                        _utc(foliar["start_utc"]) <= start and _utc(foliar["end_utc"]) >= end)
    if times != required_times or not valid_foliar:
        return sample_moisture_prior(uncertainty_manifest, count, seed=seed)
    foliar_value = _finite(foliar["value"], "foliar_moisture")
    for key in ("native_cell_id", "vegetation_config_id", "station_id", "init_config"):
        if not isinstance(station.get(key), str) or not station[key].strip():
            raise ValueError(f"station requires {key}")
    latitude = _finite(station["latitude"], "latitude", -90)
    if latitude > 90:
        raise ValueError("latitude exceeds 90")
    _finite(station["elevation_m"], "elevation_m", -500)
    binary = Path(binary).resolve()
    binary_hash = _sha(binary)
    if attestation.get("source_commit") != SOURCE_COMMIT or attestation.get("binary_sha256") != binary_hash:
        raise ValueError("pinned source commit and matching binary SHA256 attestation required")
    if "build_patch_sha256" not in attestation:
        raise ValueError("attestation must declare build_patch_sha256 (null for pristine build)")
    payload = {"weather": rows, "station": station, "attestation": attestation,
               "start_utc": start.isoformat(), "end_utc": end.isoformat(), "foliar": foliar}
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    key = hashlib.sha256(encoded).hexdigest()
    root = Path(cache_dir) / key
    root.mkdir(parents=True, exist_ok=True)
    (root / "inputs.json").write_bytes(encoded)
    output = root / "moistures.csv"
    if not output.exists():
        # Unique output avoids upstream append semantics and partial-cache reuse.
        with tempfile.TemporaryDirectory(dir=root) as work:
            work = Path(work)
            station_timezone = _station_timezone(station["init_config"])
            fw = work / "weather.fw21"
            with fw.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["StationID", "DateTime", "Temperature(C)", "RelativeHumidity(%)", "Precipitation(mm)", "WindSpeed(kph)", "WindAzimuth(degrees)", "SolarRadiation(W/m2)", "SnowFlag"])
                for row, stamp in zip(rows, times):
                    temp = _finite(row["temperature_c"], "temperature_c", -100)
                    rh = _finite(row["relative_humidity_pct"], "relative_humidity_pct")
                    if rh > 100:
                        raise ValueError("relative humidity exceeds 100 percent")
                    writer.writerow([station["station_id"], _fw21_timestamp(stamp, station_timezone), temp, rh,
                                     _finite(row["precipitation_mm"], "precipitation_mm"),
                                     _finite(row["wind_speed_m_s"], "wind_speed_m_s") * 3.6,
                                     _finite(row["upwind_direction_deg"], "upwind_direction_deg"),
                                     _finite(row["shortwave_radiation_w_m2"], "shortwave_radiation_w_m2"), 0])
            init = work / "init.txt"
            # The pinned CLI takes stationID in run config, not initialization.
            init.write_text(station["init_config"])
            match = re.search(r'^\s*latitude\s*=\s*"([^"\n]+)"\s*;', station["init_config"], re.MULTILINE)
            if not match or float(match.group(1)) != float(station["latitude"]):
                raise ValueError("init_config must explicitly match station latitude")
            config = {"initFile": str(init), "wxFile": str(fw), "loadFromStateFile": "", "saveToStateFile": "", "allOutputsFile": "", "indexOutputFile": "", "fuelMoisturesOutputFile": str(work / "output.csv"), "outputInterval": "0", "useStoredOutputs": "0", "stationID": station["station_id"]}
            cfg = work / "run.txt"
            cfg.write_text("\n".join(f"{name} = {json.dumps(value)};" for name, value in config.items()))
            (root / "weather.fw21").write_bytes(fw.read_bytes())
            (root / "init.txt").write_text(station["init_config"])
            (root / "run.txt").write_text(cfg.read_text())
            completed = subprocess.run([str(binary), str(cfg)], capture_output=True, text=True, check=False)
            (root / "execution.json").write_text(json.dumps({"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}))
            if completed.returncode:
                raise RuntimeError(f"NFDRS4 failed: {completed.stderr}\n{completed.stdout}")
            parsed = parse_moisture_csv((work / "output.csv").read_text(), station_id=station["station_id"])
            if any(start + timedelta(hours=i) not in parsed for i in range(count)):
                raise ValueError("NFDRS4 output does not cover requested hours")
            (work / "output.csv").replace(output)
    parsed = parse_moisture_csv(output.read_text(), station_id=station["station_id"])
    fields = {name: [parsed[start + timedelta(hours=i)][name] for i in range(count)] for name in OUTPUT_FIELDS.values()}
    fields["foliar_moisture"] = [foliar_value] * count
    return {"fields": fields, "provenance": {"moisture_source": "nfdrs4", "source_commit": SOURCE_COMMIT,
            "binary_sha256": binary_hash, "attestation": attestation, "cache_key": key,
            "native_cell_id": station["native_cell_id"], "vegetation_config_id": station["vegetation_config_id"],
            "station": station, "foliar": foliar, "historical_readiness_certified": False}}
