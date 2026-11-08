"""Acquire NOAA HRRR hourly f00 analyses, not forecasts or fabricated weather.

Published .idx offsets select native GRIB messages. PRATE is instantaneous
kg/m2/s, NEVER an hourly accumulation. Archive bytes and source timestamps are
hashed. ecCodes decodes native Lambert grids; earth-relative U/V remain vectors.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import requests

from .fetch_landfire import AcquisitionError, _write_json, download_cached, sha256_file, utc_now

ARCHIVE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FIELDS = {
    "u_wind": ("UGRD", "10 m above ground", "m/s"),
    "v_wind": ("VGRD", "10 m above ground", "m/s"),
    "temperature_k": ("TMP", "2 m above ground", "K"),
    "relative_humidity": ("RH", "2 m above ground", "%"),
    "precipitation_rate": ("PRATE", "surface", "kg/m2/s"),
    "shortwave": ("DSWRF", "surface", "W/m2"),
}


def parse_utc(value) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AcquisitionError(f"invalid UTC timestamp: {value}") from exc
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise AcquisitionError("timestamps must be timezone-aware UTC")
    if value.minute or value.second or value.microsecond:
        raise AcquisitionError("HRRR acquisition timestamps must lie on hourly boundaries")
    return value.astimezone(timezone.utc)


def analysis_url(valid_time) -> str:
    valid_time = parse_utc(valid_time)
    return f"{ARCHIVE}/hrrr.{valid_time:%Y%m%d}/conus/hrrr.t{valid_time:%H}z.wrfsfcf00.grib2"


def select_index(text: str) -> list[dict]:
    """Require exactly one analysis record for every requested native variable."""
    records = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 6:
            raise AcquisitionError(f"malformed HRRR index record: {line}")
        try:
            offset = int(parts[1])
        except ValueError as exc:
            raise AcquisitionError(f"invalid HRRR byte offset: {line}") from exc
        if offset < 0 or (records and offset <= records[-1]["start"]):
            raise AcquisitionError("HRRR index offsets are not strictly increasing")
        records.append({"start": offset, "parameter": parts[3], "level": parts[4],
                        "step": parts[5], "index_line": line, "reference": parts[2]})
    selected = []
    for field, (parameter, level, _) in FIELDS.items():
        matches = [(i, record) for i, record in enumerate(records)
                   if (record["parameter"], record["level"], record["step"]) == (parameter, level, "anl")]
        if len(matches) != 1:
            raise AcquisitionError(f"HRRR missing/ambiguous analysis {parameter}:{level}:anl")
        i, record = matches[0]
        selected.append({**record, "field": field,
                         "end": records[i + 1]["start"] - 1 if i + 1 < len(records) else None})
    return sorted(selected, key=lambda item: item["start"])


def acquire_grib(valid_time, cache_dir, session=None) -> tuple[Path, dict]:
    """Fetch selected ranges, using full files only for servers ignoring Range."""
    valid_time = parse_utc(valid_time)
    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = analysis_url(valid_time)
    index_path, index_metadata = download_cached(url + ".idx", cache_dir, session)
    records = select_index(index_path.read_text())
    expected_reference = f"d={valid_time:%Y%m%d%H}"
    if any(record["reference"] != expected_reference for record in records):
        raise AcquisitionError("HRRR index reference time does not match requested analysis")
    key = hashlib.sha256((url + index_metadata["sha256"]).encode()).hexdigest()[:24]
    path = cache_dir / f"{key}_analysis.grib2"
    sidecar = path.with_suffix(".json")
    if path.exists():
        if not sidecar.exists():
            raise AcquisitionError(f"unverified HRRR cache: {path}")
        metadata = json.loads(sidecar.read_text())
        if metadata.get("sha256") != sha256_file(path) or metadata.get("source_url") != url:
            raise AcquisitionError(f"HRRR cache checksum mismatch: {path}")
        return path, metadata
    temporary = path.with_name(path.name + ".part")
    full_path = None
    mode = "http_ranges"
    try:
        with temporary.open("wb") as target:
            for record in records:
                start, end = record["start"], record["end"]
                if full_path is None:
                    range_header = f"bytes={start}-{end if end is not None else ''}"
                    with session.get(url, headers={"Range": range_header, "Accept-Encoding": "identity"},
                                     stream=True, timeout=180) as response:
                        response.raise_for_status()
                        if response.status_code == 200:
                            # The server ignored Range: this response IS the full file.
                            full_path = cache_dir / (key + "_full.grib2")
                            full_temp = full_path.with_name(full_path.name + ".part")
                            try:
                                size = 0
                                with full_temp.open("wb") as full:
                                    for chunk in response.iter_content(1024 * 1024):
                                        if chunk:
                                            full.write(chunk)
                                            size += len(chunk)
                                if response.headers.get("Content-Length") and size != int(response.headers["Content-Length"]):
                                    raise AcquisitionError("truncated full HRRR download")
                                full_temp.replace(full_path)
                            finally:
                                full_temp.unlink(missing_ok=True)
                            mode = "full_file_range_unsupported"
                        elif response.status_code == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", response.headers.get("Content-Range", ""))
                            if not match or int(match[1]) != start or (end is not None and int(match[2]) != end):
                                raise AcquisitionError("HRRR response Content-Range differs from published index")
                            expected_size = int(match[2]) - start + 1
                            size = 0
                            for chunk in response.iter_content(1024 * 1024):
                                if chunk:
                                    target.write(chunk)
                                    size += len(chunk)
                            if size != expected_size:
                                raise AcquisitionError("truncated HRRR byte range")
                            continue
                        else:
                            raise AcquisitionError(f"HRRR unexpected range response {response.status_code}")
                with full_path.open("rb") as full:
                    full.seek(start)
                    remaining = end - start + 1 if end is not None else full_path.stat().st_size - start
                    if remaining <= 0:
                        raise AcquisitionError("HRRR index exceeds full-file length")
                    while remaining:
                        chunk = full.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise AcquisitionError("HRRR full file truncated at published range")
                        target.write(chunk)
                        remaining -= len(chunk)
        metadata = {"source_url": url, "sha256": sha256_file(temporary), "retrieved_utc": utc_now(),
                    "index_url": url + ".idx", "index_sha256": index_metadata["sha256"],
                    "valid_time_utc": valid_time.isoformat().replace("+00:00", "Z"),
                    "retrieval_mode": mode, "records": records}
        temporary.replace(path)
        _write_json(sidecar, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return path, metadata


def rotate_grid_winds(u, v, convergence_degrees):
    """Convert grid-east/north to true earth-east/north using convergence."""
    angle = np.deg2rad(convergence_degrees)
    return u * np.cos(angle) + v * np.sin(angle), -u * np.sin(angle) + v * np.cos(angle)


def wind_from_degrees(u, v):
    """Meteorological direction FROM, clockwise from north (calm is 0)."""
    direction = np.mod(np.degrees(np.arctan2(-np.asarray(u), -np.asarray(v))), 360.0)
    return np.where((np.asarray(u) == 0) & (np.asarray(v) == 0), 0.0, direction)


def decode_grib(path, expected_time) -> tuple[dict, dict]:
    """Decode real GRIB2 via ecCodes, preserving native resolution and valid time."""
    try:
        import eccodes as ec
        from pyproj import CRS, Proj, Transformer
    except ImportError as exc:
        raise AcquisitionError("HRRR decoding requires eccodes and pyproj (install requirements.txt)") from exc
    expected_time = parse_utc(expected_time)
    fields, grid, first_lat, first_lon, vector_flag = {}, None, None, None, None
    # WMO discipline/category/number are stable across ecCodes short-name tables.
    identities = {
        (0, 2, 2, "heightAboveGround", 10): "u_wind",
        (0, 2, 3, "heightAboveGround", 10): "v_wind",
        (0, 0, 0, "heightAboveGround", 2): "temperature_k",
        (0, 1, 1, "heightAboveGround", 2): "relative_humidity",
        (0, 1, 7, "surface", 0): "precipitation_rate",
        (0, 4, 7, "surface", 0): "shortwave",
        (0, 4, 192, "surface", 0): "shortwave",  # NCEP local DSWRF table
    }
    with Path(path).open("rb") as stream:
        while True:
            message = ec.codes_grib_new_from_file(stream)
            if message is None:
                break
            try:
                get = lambda key: ec.codes_get(message, key)
                identity = (int(get("discipline")), int(get("parameterCategory")),
                            int(get("parameterNumber")), get("typeOfLevel"), int(get("level")))
                field = identities.get(identity)
                if field is None or field in fields:
                    raise AcquisitionError(f"unexpected or duplicate HRRR GRIB field: {identity}")
                if int(get("dataDate")) != int(expected_time.strftime("%Y%m%d")) or int(get("dataTime")) != expected_time.hour * 100 or int(get("forecastTime")) != 0:
                    raise AcquisitionError("HRRR message is not the requested f00 analysis")
                if int(get("validityDate")) != int(expected_time.strftime("%Y%m%d")) or int(get("validityTime")) != expected_time.hour * 100:
                    raise AcquisitionError("HRRR message validity time differs from requested analysis")
                if get("gridType") != "lambert" or int(get("alternativeRowScanning")) or int(get("jPointsAreConsecutive")):
                    raise AcquisitionError("unsupported HRRR native grid/scanning order")
                nx, ny = int(get("Nx")), int(get("Ny"))
                dx, dy = float(get("DxInMetres")), float(get("DyInMetres"))
                if not (math.isclose(dx, 3000) and math.isclose(dy, 3000)):
                    raise AcquisitionError("HRRR native grid is not the expected 3 km product")
                radius = float(get("radius"))
                crs = CRS.from_proj4(f"+proj=lcc +lat_1={get('Latin1InDegrees')} +lat_2={get('Latin2InDegrees')} +lat_0={get('LaDInDegrees')} +lon_0={get('LoVInDegrees')} +R={radius} +units=m +no_defs")
                lon0, lat0 = float(get("longitudeOfFirstGridPointInDegrees")), float(get("latitudeOfFirstGridPointInDegrees"))
                x0, y0 = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True).transform(lon0, lat0)
                sx = -dx if int(get("iScansNegatively")) else dx
                sy = dy if int(get("jScansPositively")) else -dy
                current_grid = {"native_crs": crs.to_wkt(), "native_transform": [sx, 0, x0 - sx / 2, 0, sy, y0 - sy / 2],
                                "shape": [ny, nx], "native_resolution_m": 3000}
                if grid is not None and current_grid != grid:
                    raise AcquisitionError("HRRR fields do not share one native grid")
                if grid is None:
                    grid = current_grid
                    first_lat = np.asarray(ec.codes_get_array(message, "latitudes")).reshape(ny, nx)
                    first_lon = np.asarray(ec.codes_get_array(message, "longitudes")).reshape(ny, nx)
                    first_lon = (first_lon + 180) % 360 - 180
                values = np.asarray(ec.codes_get_values(message), dtype=np.float32).reshape(ny, nx)
                if int(get("bitmapPresent")):
                    bitmap = np.asarray(ec.codes_get_array(message, "bitmap")).reshape(ny, nx)
                    values[bitmap == 0] = np.nan
                if not np.isfinite(values).all():
                    raise AcquisitionError(f"HRRR {field} contains missing/nonfinite values")
                fields[field] = values
                if field in ("u_wind", "v_wind"):
                    flag = int(get("uvRelativeToGrid"))
                    if flag not in (0, 1) or vector_flag is not None and flag != vector_flag:
                        raise AcquisitionError("inconsistent GRIB vector orientation")
                    vector_flag = flag
            finally:
                ec.codes_release(message)
    if set(fields) != set(FIELDS):
        raise AcquisitionError(f"HRRR decoded fields missing: {sorted(set(FIELDS) - set(fields))}")
    if vector_flag:
        convergence = Proj(grid["native_crs"]).get_factors(first_lon, first_lat).meridian_convergence
        fields["u_wind"], fields["v_wind"] = rotate_grid_winds(fields["u_wind"], fields["v_wind"], convergence)
    fields["lat"], fields["lon"] = first_lat, first_lon
    grid.update(vector_basis="earth_east_north", source_vector_basis="grid_relative" if vector_flag else "earth_east_north")
    return fields, grid


def fetch_hrrr(start, end, cache_dir, out, session=None) -> dict:
    """Write native NPZ analyses for [start,end), preserving source valid times."""
    start, end = parse_utc(start), parse_utc(end)
    if end <= start:
        raise AcquisitionError("HRRR end must be after start (exclusive)")
    out = Path(out)
    if out.suffix != ".npz":
        raise AcquisitionError("HRRR output must have .npz extension")
    out.parent.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    arrays = {field: [] for field in FIELDS}
    times, source_records, grid, lat, lon = [], [], None, None, None
    current = start
    while current < end:
        try:
            grib, source = acquire_grib(current, cache_dir, session)
            decoded, current_grid = decode_grib(grib, current)
        except (AcquisitionError, requests.RequestException, OSError, ValueError) as exc:
            raise AcquisitionError(f"HRRR {current.isoformat()}: {exc}") from exc
        if grid is not None and grid != current_grid:
            raise AcquisitionError("HRRR native grid changed during requested interval")
        grid = current_grid
        if lat is None:
            lat, lon = decoded["lat"], decoded["lon"]
        for field in FIELDS:
            arrays[field].append(decoded[field])
        times.append(current.isoformat().replace("+00:00", "Z"))
        source_records.append(source)
        current += timedelta(hours=1)
    temporary = out.with_name(out.name + ".part")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **{field: np.stack(values) for field, values in arrays.items()},
                                lat=lat, lon=lon, time_utc=np.asarray(times))
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {"path": str(out.resolve()), "sha256": sha256_file(out),
                "provider": "NOAA HRRR", "version": "HRRR operational hourly f00 analysis",
                "license": "US federal public-domain data", "retrieved_utc": utc_now(),
                "valid_from_utc": times[0], "valid_until_utc": end.isoformat().replace("+00:00", "Z"),
                "units": {field: descriptor[2] for field, descriptor in FIELDS.items()},
                "crs": grid["native_crs"], "transform": grid["native_transform"], **grid,
                "valid_times_utc": times, "source_records": source_records,
                "limitations": ["Native 3 km analysis, not terrain downscaling.",
                                "PRATE is instantaneous precipitation rate, not hourly accumulated precipitation.",
                                "Analysis is hindcast truth; future analysis bands must not be exposed to policies."]}
    result = {"sources": {"hrrr": metadata}}
    _write_json(out.with_suffix(".json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="Inclusive hourly UTC ISO timestamp")
    parser.add_argument("--end", required=True, help="Exclusive hourly UTC ISO timestamp")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        fetch_hrrr(args.start, args.end, args.cache, args.out)
    except (AcquisitionError, requests.RequestException, OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"HRRR acquisition failed: {exc}\n")


if __name__ == "__main__":
    main()
