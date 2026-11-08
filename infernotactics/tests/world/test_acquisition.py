"""Offline protocol fixtures; network and native decoding smoke are opt-in/manual."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.data_pipeline import fetch_hrrr as hrrr
from src.data_pipeline import fetch_landfire as landfire


class Response:
    def __init__(self, body=b"", status=200, headers=None):
        self.body = body.encode() if isinstance(body, str) else body
        self.status_code = status
        self.headers = headers or {}
        self.text = self.body.decode(errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, size):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Session:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.handler(url, kwargs)


def index_text():
    # Each fixture message has ten bytes, with one unrelated trailing record.
    return "\n".join(
        f"{i + 1}:{i * 10}:d=2025010700:{parameter}:{level}:anl:"
        for i, (parameter, level, _) in enumerate(hrrr.FIELDS.values())
    ) + "\n7:60:d=2025010700:APCP:surface:0-0 day acc fcst:\n"


def test_catalog_follows_published_pagination_and_checksum():
    root = "https://landfire.gov/catalog"
    page0 = """<a href='/data/LF2023_FBFM40_CONUS.zip'>fuel</a>
    <a href='/data/LF2023_FBFM40_CONUS.zip.sha256'>checksum</a>
    <a href='?page=1' rel='next'>Next</a>"""
    page1 = "".join(f"<a href='/data/LF2023_{product}_CONUS.zip'>{product}</a>"
                    for product in ("CC", "CH", "CBH", "CBD"))
    session = Session(lambda url, _: Response(page1 if "page=1" in url else page0))
    products = landfire.discover_products(session, root)
    assert set(products) == set(landfire.PRODUCTS)
    assert len(session.calls) == 2
    assert products["FBFM40"]["checksum_urls"] == ["https://landfire.gov/data/LF2023_FBFM40_CONUS.zip.sha256"]
    assert products["CC"]["url"] == "https://landfire.gov/data/LF2023_CC_CONUS.zip"


def test_catalog_cycle_is_named_failure():
    session = Session(lambda url, _: Response(f"<a href='{url}' rel='next'>next</a>"))
    with pytest.raises(landfire.AcquisitionError, match="pagination cycle"):
        landfire.discover_products(session)


def test_download_cache_hash_and_provider_checksum(tmp_path):
    url = "https://example.test/exact.zip"
    body = b"provider fixture"
    session = Session(lambda *_: Response(body, headers={"Content-Length": str(len(body))}))
    checksums = {"sha256": hashlib.sha256(body).hexdigest()}
    path, first = landfire.download_cached(url, tmp_path, session, checksums)
    same, second = landfire.download_cached(url, tmp_path, session, checksums)
    assert path == same and first == second and len(session.calls) == 1
    path.write_bytes(b"corruption")
    with pytest.raises(landfire.AcquisitionError, match="cache checksum mismatch"):
        landfire.download_cached(url, tmp_path, session, checksums)


def test_wrong_provider_checksum_does_not_publish_cache(tmp_path):
    session = Session(lambda *_: Response(b"wrong"))
    with pytest.raises(landfire.AcquisitionError, match="provider sha256 checksum mismatch"):
        landfire.download_cached("https://example.test/file.zip", tmp_path, session, {"sha256": "0" * 64})
    assert not list(tmp_path.glob("*.zip"))
    assert not list(tmp_path.glob("*.part"))


def make_raster(path, product="FBFM40"):
    data = np.arange(100, dtype="int16").reshape(10, 10)
    data[2, 2] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=10, height=10, count=1,
                       dtype="int16", crs="EPSG:4326", transform=from_origin(-120, 40, 1, 1),
                       nodata=-9999) as target:
        target.write(data, 1)
        target.update_tags(PRODUCT=product, VERSION="LF2023")
    return data


def test_zip_crop_reads_native_window_and_preserves_nodata(tmp_path):
    raw = tmp_path / "LF2023_FBFM40_CONUS.tif"
    data = make_raster(raw)
    archive = tmp_path / "LF2023_FBFM40_CONUS.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.write(raw, "nested/LF2023_FBFM40_CONUS.tif")
    out = tmp_path / "crop.tif"
    metadata = landfire.crop_product(archive, "FBFM40", (-118, 35, -115, 38), out)
    with rasterio.open(out) as source:
        assert source.shape == (3, 3)
        np.testing.assert_array_equal(source.read(1), data[2:5, 2:5])
        assert source.read(1, masked=True).mask[0, 0]
        assert source.tags()["VERSION"] == "LF2023"
    assert metadata["crs"] == "EPSG:4326"
    assert metadata["nodata"] == -9999


def test_exact_local_products_and_validity_required(tmp_path):
    products = {}
    for product in landfire.PRODUCTS:
        path = tmp_path / f"LF2023_{product}_CONUS.tif"
        make_raster(path, product)
        products[product] = path
    with pytest.raises(landfire.AcquisitionError, match="provider-supported"):
        landfire.fetch_landfire(tmp_path / "cache", (-118, 35, -115, 38), tmp_path / "out", products)
    result = landfire.fetch_landfire(tmp_path / "cache", (-118, 35, -115, 38), tmp_path / "out", products,
                                    valid_from_utc="2023-01-01T00:00:00Z", valid_until_utc="2024-01-01T00:00:00Z")
    # Dates here are explicit fixture metadata, not a claim about LF2023 validity.
    assert set(result["sources"]) == set(landfire.PRODUCTS)
    assert result["sources"]["CC"]["scale"] == .01
    assert result["sources"]["CH"]["scale"] == .1
    assert result["sources"]["CBD"]["units"] == "hundredth kg/m3"
    for source in result["sources"].values():
        assert source["sha256"] == landfire.sha256_file(source["path"])
        assert source["valid_from_utc"] == "2023-01-01T00:00:00Z"


def test_missing_catalog_product_never_substitutes(monkeypatch, tmp_path):
    monkeypatch.setattr(landfire, "discover_products", lambda *_: {"FBFM40": {"url": "unused"}})
    with pytest.raises(landfire.AcquisitionError, match="missing products: CBD, CBH, CC, CH"):
        landfire.fetch_landfire(tmp_path, (-118, 35, -115, 38), tmp_path / "out",
                                valid_from_utc="2023-01-01T00:00:00Z", valid_until_utc="2024-01-01T00:00:00Z")


def test_index_selects_analysis_rate_not_zero_duration_accumulation():
    selected = hrrr.select_index(index_text())
    assert len(selected) == 6
    assert selected[-2]["parameter"] == "PRATE"
    assert all(record["step"] == "anl" for record in selected)
    assert selected[-1]["end"] == 59
    with pytest.raises(hrrr.AcquisitionError, match="PRATE"):
        hrrr.select_index(index_text().replace("PRATE", "APCP"))


def test_index_rejects_duplicate_or_unordered_offsets():
    with pytest.raises(hrrr.AcquisitionError, match="strictly increasing"):
        hrrr.select_index(index_text().replace("2:10:", "2:0:"))
    with pytest.raises(hrrr.AcquisitionError, match="ambiguous analysis"):
        hrrr.select_index(index_text() + "8:70:d=2025010700:UGRD:10 m above ground:anl:\n")


@pytest.mark.parametrize("range_supported", [True, False])
def test_hrrr_range_and_full_file_fallback(tmp_path, range_supported):
    blob = b"".join(bytes([i]) * 10 for i in range(7))
    def handler(url, kwargs):
        if url.endswith(".idx"):
            return Response(index_text())
        if not range_supported:
            return Response(blob, headers={"Content-Length": str(len(blob))})
        start, end = kwargs["headers"]["Range"].removeprefix("bytes=").split("-")
        start, end = int(start), int(end)
        return Response(blob[start:end + 1], status=206,
                        headers={"Content-Range": f"bytes {start}-{end}/{len(blob)}"})
    session = Session(handler)
    path, metadata = hrrr.acquire_grib("2025-01-07T00:00:00Z", tmp_path, session)
    assert path.read_bytes() == blob[:60]
    assert metadata["retrieval_mode"] == ("http_ranges" if range_supported else "full_file_range_unsupported")
    assert metadata["valid_time_utc"] == "2025-01-07T00:00:00Z"
    assert len(session.calls) == (7 if range_supported else 2)
    hrrr.acquire_grib("2025-01-07T00:00:00Z", tmp_path, session)
    assert len(session.calls) == (7 if range_supported else 2)


def test_hrrr_rejects_wrong_content_range(tmp_path):
    def handler(url, kwargs):
        return Response(index_text()) if url.endswith(".idx") else Response(
            b"1234567890", status=206, headers={"Content-Range": "bytes 1-10/100"})
    with pytest.raises(hrrr.AcquisitionError, match="Content-Range"):
        hrrr.acquire_grib("2025-01-07T00:00:00Z", tmp_path, Session(handler))
    assert not list(tmp_path.glob("*_analysis.grib2"))


def test_grid_rotation_and_meteorological_direction():
    u, v = hrrr.rotate_grid_winds(np.array([1., 0.]), np.array([0., 1.]), 90)
    np.testing.assert_allclose(u, [0, 1], atol=1e-12)
    np.testing.assert_allclose(v, [-1, 0], atol=1e-12)
    np.testing.assert_allclose(hrrr.wind_from_degrees([1, 0, -1, 0, 0], [0, 1, 0, -1, 0]),
                               [270, 180, 90, 0, 0])


@pytest.mark.parametrize("value", ["2025-01-07T00:30:00Z", "2025-01-07T00:00:00", "2025-01-07T00:00:00-08:00"])
def test_hrrr_requires_hourly_explicit_utc(value):
    with pytest.raises(hrrr.AcquisitionError):
        hrrr.parse_utc(value)


def test_hrrr_native_npz_metadata_and_times(tmp_path, monkeypatch):
    def acquire(time, cache, session):
        return Path("fixture.grib2"), {"source_url": hrrr.analysis_url(time), "sha256": "fixture"}
    grid = {"native_crs": "EPSG:5070", "native_transform": [3000, 0, 0, 0, -3000, 6000],
            "native_resolution_m": 3000, "shape": [2, 2], "vector_basis": "earth_east_north"}
    def decode(path, time):
        return {**{field: np.full((2, 2), time.hour, dtype=np.float32) for field in hrrr.FIELDS},
                "lat": np.full((2, 2), 34.), "lon": np.full((2, 2), -118.)}, grid
    monkeypatch.setattr(hrrr, "acquire_grib", acquire)
    monkeypatch.setattr(hrrr, "decode_grib", decode)
    out = tmp_path / "weather.npz"
    result = hrrr.fetch_hrrr("2025-01-07T00:00:00Z", "2025-01-07T02:00:00Z", tmp_path, out)
    with np.load(out, allow_pickle=False) as data:
        assert data["u_wind"].shape == (2, 2, 2)
        assert data["time_utc"].tolist() == ["2025-01-07T00:00:00Z", "2025-01-07T01:00:00Z"]
        assert "wind_direction" not in data.files  # interpolate vectors before direction
    source = result["sources"]["hrrr"]
    assert source["native_resolution_m"] == 3000
    assert source["units"]["precipitation_rate"] == "kg/m2/s"
    assert source["vector_basis"] == "earth_east_north"
    assert source["valid_until_utc"] == "2025-01-07T02:00:00Z"
    assert source["sha256"] == landfire.sha256_file(out)
    assert json.loads(out.with_suffix(".json").read_text()) == result


def test_rotation_matches_projected_grid_north_geometry():
    from pyproj import CRS, Geod, Proj, Transformer

    crs = CRS.from_proj4("+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +R=6371229 +units=m")
    lon, lat = -118., 34.
    project = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
    unproject = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)
    x, y = project.transform(lon, lat)
    north_lon, north_lat = unproject.transform(x, y + 1)
    azimuth, _, _ = Geod(a=6371229, b=6371229).inv(lon, lat, north_lon, north_lat)
    convergence = Proj(crs).get_factors(lon, lat).meridian_convergence
    u, v = hrrr.rotate_grid_winds(0., 1., convergence)
    np.testing.assert_allclose([u, v], [np.sin(np.deg2rad(azimuth)), np.cos(np.deg2rad(azimuth))], atol=1e-7)


def test_acquisition_clis_report_named_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as failed:
        hrrr.main(["--start", "2025-01-07T00:30:00Z", "--end", "2025-01-07T01:00:00Z",
                   "--cache", str(tmp_path), "--out", str(tmp_path / "out.npz")])
    assert failed.value.code == 1
    assert "HRRR acquisition failed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as failed:
        landfire.main(["--cache", str(tmp_path), "--out", str(tmp_path / "out"),
                       "--bbox", "-118", "34", "-117", "35", "--local", "NOT_A_PRODUCT=x",
                       "--valid-from", "2023-01-01T00:00:00Z", "--valid-until", "2024-01-01T00:00:00Z"])
    assert failed.value.code == 2
    assert "unique PRODUCT=PATH" in capsys.readouterr().err
