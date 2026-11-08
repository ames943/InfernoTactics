"""Acquire exact LF2023 CONUS fuel products, retaining raw provider units.

Catalog links (including pagination), rather than constructed download names,
are authoritative. ZIPs stay cached; only the requested raster window is read.
The emitted sources.json can be used to assemble the explicit world INPUTS file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import unquote, urljoin, urlparse
import zipfile

import requests

CATALOG_URL = "https://landfire.gov/data/FullExtentDownloads?field_version_target_id=All&field_theme_target_id=All&field_region_id_target_id=4"
PRODUCTS = {
    "FBFM40": ("code", 1.0), "CC": ("percent", 0.01),
    "CH": ("decimeter", 0.1), "CBH": ("decimeter", 0.1),
    "CBD": ("hundredth kg/m3", 0.01),
}


class AcquisitionError(RuntimeError):
    """A named source could not be acquired or verified."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            values = dict(attrs)
            if values.get("href"):
                self.links.append(values)


def discover_products(session=None, catalog_url: str = CATALOG_URL) -> dict:
    """Follow published next links; return exact product and checksum URLs."""
    session = session or requests.Session()
    found, all_links, visited = {}, set(), set()
    current = catalog_url
    while current:
        if current in visited:
            raise AcquisitionError(f"LANDFIRE catalog pagination cycle: {current}")
        visited.add(current)
        response = session.get(current, timeout=90)
        response.raise_for_status()
        parser = _Links()
        parser.feed(response.text)
        next_pages = []
        for attrs in parser.links:
            url = urljoin(current, attrs["href"])
            all_links.add(url)
            filename = unquote(Path(urlparse(url).path).name)
            match = re.fullmatch(r"LF2023_(FBFM40|CC|CH|CBH|CBD)_CONUS\.zip", filename, re.I)
            if match:
                product = match.group(1).upper()
                if product in found and found[product]["url"] != url:
                    raise AcquisitionError(f"LANDFIRE {product}: ambiguous catalog links")
                found[product] = {"url": url}
            if "next" in attrs.get("rel", "").split() or attrs.get("title", "").lower() == "go to next page":
                if urlparse(url).netloc != urlparse(catalog_url).netloc:
                    raise AcquisitionError("LANDFIRE catalog next page leaves official host")
                next_pages.append(url)
        if len(set(next_pages)) > 1:
            raise AcquisitionError("LANDFIRE catalog has ambiguous next pages")
        current = next_pages[0] if next_pages else None
    for entry in found.values():
        checksum_links = [url for url in all_links if any(
            urlparse(url).path == urlparse(entry["url"]).path + suffix
            for suffix in (".sha256", ".sha256sum", ".md5", ".md5sum"))]
        entry["checksum_urls"] = sorted(checksum_links)
    return found


def _check_provider(path: Path, checksums: dict) -> None:
    for algorithm, expected in checksums.items():
        digest = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != expected.lower():
            raise AcquisitionError(f"{path.name}: provider {algorithm} checksum mismatch")


def download_cached(url: str, cache_dir: Path, session=None, checksums=None) -> tuple[Path, dict]:
    """Cache verified bytes; reject damaged caches instead of silently replacing."""
    session = session or requests.Session()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    path = cache_dir / (key + "_" + Path(urlparse(url).path).name)
    metadata_path = path.with_name(path.name + ".json")
    checksums = checksums or {}
    if path.exists():
        if not metadata_path.exists():
            raise AcquisitionError(f"unverified cache entry: {path}")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("source_url") != url or metadata.get("sha256") != sha256_file(path):
            raise AcquisitionError(f"cache checksum mismatch: {path}")
        _check_provider(path, checksums)
        return path, metadata
    temporary = path.with_name(path.name + ".part")
    try:
        with session.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            digest, size = hashlib.sha256(), 0
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if response.headers.get("Content-Length") and not response.headers.get("Content-Encoding"):
                if size != int(response.headers["Content-Length"]):
                    raise AcquisitionError(f"truncated download: {url}")
            _check_provider(temporary, checksums)
            metadata = {"source_url": url, "sha256": digest.hexdigest(),
                        "retrieved_utc": utc_now(), "size_bytes": size,
                        "provider_checksums": checksums}
        temporary.replace(path)
        _write_json(metadata_path, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return path, metadata


def _provider_checksums(entry, session):
    checksums = {}
    for url in entry.get("checksum_urls", []):
        response = session.get(url, timeout=90)
        response.raise_for_status()
        algorithm = "sha256" if ".sha256" in urlparse(url).path else "md5"
        length = 64 if algorithm == "sha256" else 32
        matches = re.findall(rf"\b[0-9a-fA-F]{{{length}}}\b", response.text)
        if len(matches) != 1:
            raise AcquisitionError(f"ambiguous provider checksum: {url}")
        checksums[algorithm] = matches[0].lower()
    return checksums


def _raster_path(path: Path, product: str) -> str:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            candidates = [name for name in archive.namelist()
                          if name.lower().endswith((".tif", ".tiff")) and
                          re.search(rf"(?:^|[_/]){product}(?:[_./]|$)", name, re.I)]
            # Official archives can use the long layer title internally.
            if not candidates:
                candidates = [name for name in archive.namelist()
                              if name.lower().endswith((".tif", ".tiff"))]
        if len(candidates) != 1:
            raise AcquisitionError(f"LANDFIRE {product}: expected one product raster in {path}")
        return f"zip://{path.resolve()}!{candidates[0]}"
    return str(path)


def crop_product(path: Path, product: str, bbox, destination: Path) -> dict:
    """Crop outward to native pixels; do not extract or read the CONUS raster."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds

    with rasterio.open(_raster_path(path, product)) as source:
        if source.crs is None or source.count != 1:
            raise AcquisitionError(f"LANDFIRE {product}: expected one georeferenced band")
        bounds = transform_bounds("EPSG:4326", source.crs, *bbox, densify_pts=41)
        requested = from_bounds(*bounds, transform=source.transform)
        window = Window(math.floor(requested.col_off), math.floor(requested.row_off),
                        math.ceil(requested.col_off + requested.width) - math.floor(requested.col_off),
                        math.ceil(requested.row_off + requested.height) - math.floor(requested.row_off))
        if window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > source.width or window.row_off + window.height > source.height:
            raise AcquisitionError(f"LANDFIRE {product}: requested bbox outside raster coverage")
        profile = source.profile.copy()
        profile.update(driver="GTiff", width=int(window.width), height=int(window.height),
                       transform=source.window_transform(window), compress="deflate")
        profile.pop("blockxsize", None)
        profile.pop("blockysize", None)
        profile.pop("tiled", None)
        data = source.read(1, window=window, masked=True)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(destination, "w", **profile) as target:
                target.write(data.filled(source.nodata if source.nodata is not None else 0), 1)
                target.write_mask(source.dataset_mask(window=window))
                target.update_tags(PRODUCT=product, VERSION="LF2023")
        return {"crs": source.crs.to_string(), "transform": list(profile["transform"])[:6],
                "nodata": source.nodata}


def fetch_landfire(cache_dir, bbox, out_dir, local_products=None, session=None,
                   valid_from_utc=None, valid_until_utc=None) -> dict:
    """Acquire all five exact products. Local products require identifiable names/tags."""
    import rasterio

    if valid_from_utc is None or valid_until_utc is None:
        raise AcquisitionError("LANDFIRE requires provider-supported valid_from_utc and valid_until_utc; retrieval date is not validity")
    try:
        valid_from = datetime.fromisoformat(valid_from_utc.replace("Z", "+00:00"))
        valid_until = datetime.fromisoformat(valid_until_utc.replace("Z", "+00:00"))
        if valid_from.tzinfo is None or valid_until.tzinfo is None or valid_until <= valid_from:
            raise ValueError("invalid interval")
    except (AttributeError, TypeError, ValueError) as exc:
        raise AcquisitionError("LANDFIRE validity must be increasing timezone-aware ISO timestamps") from exc
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox) or not (-180 <= bbox[0] < bbox[2] <= 180 and -90 <= bbox[1] < bbox[3] <= 90):
        raise AcquisitionError("bbox must be west south east north in EPSG:4326")
    session = session or requests.Session()
    local_products = local_products or {}
    unknown = set(local_products) - set(PRODUCTS)
    if unknown:
        raise AcquisitionError(f"unknown LANDFIRE products: {sorted(unknown)}")
    catalog = discover_products(session) if set(local_products) != set(PRODUCTS) else {}
    missing = set(PRODUCTS) - set(local_products) - set(catalog)
    if missing:
        raise AcquisitionError(f"LANDFIRE LF2023 missing products: {', '.join(sorted(missing))}; supply exact product with --local PRODUCT=PATH")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    for product, (units, scale) in PRODUCTS.items():
        try:
            if product in local_products:
                path = Path(local_products[product]).resolve()
                if not path.is_file():
                    raise AcquisitionError(f"local product not found: {path}")
                identified = bool(re.search(rf"LF2023_{product}_CONUS(?:[_.]|$)", path.name, re.I))
                if not identified and path.suffix.lower() in (".tif", ".tiff"):
                    with rasterio.open(path) as source:
                        tags = source.tags()
                        identified = tags.get("PRODUCT") == product and tags.get("VERSION") == "LF2023"
                if not identified:
                    raise AcquisitionError("local file must identify the exact LF2023 product in filename or PRODUCT/VERSION raster tags")
                metadata = {"source_url": path.as_uri(), "sha256": sha256_file(path),
                            "retrieved_utc": utc_now(), "provider_checksums": {}}
            else:
                entry = catalog[product]
                path, metadata = download_cached(entry["url"], Path(cache_dir), session,
                                                  _provider_checksums(entry, session))
            destination = out_dir / f"LF2023_{product}_crop.tif"
            raster_metadata = crop_product(path, product, bbox, destination)
            sources[product] = {"path": str(destination.resolve()), "sha256": sha256_file(destination),
                                "provider": "LANDFIRE", "version": "LF2023", "license": "US federal public-domain data",
                                "retrieved_utc": metadata["retrieved_utc"], "source_url": metadata["source_url"],
                                "source_sha256": metadata["sha256"], "provider_checksums": metadata["provider_checksums"],
                                "valid_from_utc": valid_from.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                                "valid_until_utc": valid_until.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                                "units": units, "scale": scale, "offset": 0.0, **raster_metadata}
        except (OSError, ValueError, requests.RequestException, zipfile.BadZipFile, AcquisitionError) as exc:
            raise AcquisitionError(f"LANDFIRE {product}: {exc}") from exc
    result = {"sources": sources, "bbox_wgs84": list(bbox),
              "note": "Raw LF2023 units retained. Supply data-valid intervals from provider metadata; retrieval time is not data-valid time."}
    _write_json(out_dir / "sources.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bbox", nargs=4, required=True, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"))
    parser.add_argument("--local", action="append", default=[], metavar="PRODUCT=PATH")
    parser.add_argument("--valid-from", required=True, help="Provider-supported data-valid interval start, ISO UTC")
    parser.add_argument("--valid-until", required=True, help="Provider-supported exclusive data-valid interval end, ISO UTC")
    args = parser.parse_args(argv)
    local = {}
    for item in args.local:
        product, separator, path = item.partition("=")
        if not separator or product.upper() not in PRODUCTS or not path or product.upper() in local:
            parser.error("--local requires a unique PRODUCT=PATH for FBFM40, CC, CH, CBH or CBD")
        local[product.upper()] = path
    try:
        fetch_landfire(args.cache, args.bbox, args.out, local,
                       valid_from_utc=args.valid_from, valid_until_utc=args.valid_until)
    except (AcquisitionError, requests.RequestException, OSError, ValueError) as exc:
        parser.exit(1, f"LANDFIRE acquisition failed: {exc}\n")


if __name__ == "__main__":
    main()
