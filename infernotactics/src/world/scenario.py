"""Hashed schema-one scenario directories, with no executable serialization.

``scenario.json`` names ``landscape``, ``weather`` and ``roads`` relative files;
``source_hashes`` maps those names to SHA-256 digests of the serialized bytes.
Original acquisition hashes remain in ``provenance``. Array archives are loaded
with ``allow_pickle=False`` and only numeric/bool arrays pass the contract.
Existing destinations are never overwritten; explicit CLI overwrite policy is
outside this API. Loading verifies artifacts before decoding them.
"""

from io import BytesIO
import hashlib
import json
from pathlib import Path, PureWindowsPath
import shutil

import numpy as np

from .contracts import SCHEMA_VERSION, Scenario

_MANIFEST_FIELDS = frozenset({
    "schema_version", "name", "crs", "transform", "start_utc", "duration_s",
    "weather_step_s", "resources", "provenance", "seed", "landscape",
    "weather", "roads", "source_hashes",
})


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _decode_json(data: bytes, label: str) -> dict:
    try:
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_object,
                            parse_constant=_invalid_constant)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return result


def _artifact_path(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("artifact paths must be nonempty relative paths inside the scenario directory")
    relative = Path(name)
    if relative.is_absolute() or PureWindowsPath(name).drive or ".." in relative.parts:
        raise ValueError(f"artifact path escapes scenario directory: {name!r}")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"missing or invalid scenario artifact: {name}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"artifact path is not a file inside scenario directory: {name}")
    return resolved


def _read_artifact(root: Path, name: object, hashes: dict) -> bytes:
    # Resolve before reading or looking up the digest, rejecting traversal first.
    artifact = _artifact_path(root, name)
    digest = hashes.get(name)
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)):
        raise ValueError(f"missing or invalid SHA-256 source hash for {name}")
    try:
        data = artifact.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read scenario artifact {name}: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != digest.lower():
        raise ValueError(f"SHA-256 mismatch for scenario artifact: {name}")
    return data


def _decode_arrays(data: bytes, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(BytesIO(data), allow_pickle=False) as archive:
            if len(set(archive.files)) != len(archive.files):
                raise ValueError("duplicate archive array name")
            arrays = {key: archive[key] for key in archive.files}
    except Exception as exc:
        raise ValueError(f"invalid nonpickle {label} archive: {exc}") from exc
    return arrays


def load_scenario(path: Path) -> Scenario:
    """Load and validate one directory, rejecting incompatible or corrupt data."""
    root = Path(path).resolve()
    if not root.is_dir():
        raise ValueError(f"scenario directory does not exist: {path}")
    manifest_path = _artifact_path(root, "scenario.json")
    try:
        manifest = _decode_json(manifest_path.read_bytes(), "scenario.json")
    except OSError as exc:
        raise ValueError(f"cannot read scenario.json: {exc}") from exc
    version = manifest.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"incompatible scenario schema_version: {version!r}; expected {SCHEMA_VERSION}")
    missing = _MANIFEST_FIELDS - manifest.keys()
    unknown = manifest.keys() - _MANIFEST_FIELDS
    if missing or unknown:
        raise ValueError(f"invalid scenario manifest fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    hashes = manifest["source_hashes"]
    if not isinstance(hashes, dict):
        raise ValueError("manifest source_hashes must map artifact paths to SHA-256 hashes")
    names = [manifest[key] for key in ("landscape", "weather", "roads")]
    # Resolve all artifact paths before decoding any archive.
    paths = [_artifact_path(root, name) for name in names]
    if len(set(paths)) != 3 or manifest_path in paths:
        raise ValueError("landscape, weather, roads and manifest must be distinct files")
    landscape = _decode_arrays(_read_artifact(root, manifest["landscape"], hashes), "landscape")
    weather = _decode_arrays(_read_artifact(root, manifest["weather"], hashes), "weather")
    roads = _decode_json(_read_artifact(root, manifest["roads"], hashes), "roads.json")
    transform = manifest["transform"]
    if not isinstance(transform, list) or len(transform) != 6:
        raise ValueError("manifest transform must be a six-element affine array")
    scenario = Scenario(
        name=manifest["name"], crs=manifest["crs"], transform=tuple(transform),
        start_utc=manifest["start_utc"], duration_s=manifest["duration_s"],
        landscape=landscape, weather=weather, weather_step_s=manifest["weather_step_s"],
        roads=roads, resources=manifest["resources"], provenance=manifest["provenance"],
        seed=manifest["seed"],
    )
    scenario.validate()
    return scenario


def save_scenario(scenario: Scenario, path: Path) -> None:
    """Validate then create a new scenario directory; never overwrite a path.

    The manifest is written last, so an interrupted write is not a loadable
    successful scenario. Ordinary failures remove only this call's new directory.
    """
    if not isinstance(scenario, Scenario):
        raise ValueError("scenario must be a Scenario")
    destination = Path(path)
    # lexists semantics also reject broken symlinks, not merely directories.
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing scenario path: {path}")
    scenario.validate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    try:
        artifact_names = {"landscape": "landscape.npz", "weather": "weather.npz", "roads": "roads.json"}
        # Named archive fields are numeric-only by Scenario.validate().
        np.savez_compressed(destination / artifact_names["landscape"], **scenario.landscape)
        np.savez_compressed(destination / artifact_names["weather"], **scenario.weather)
        (destination / artifact_names["roads"]).write_bytes(_json_bytes(scenario.roads))
        hashes = {
            name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
            for name in artifact_names.values()
        }
        manifest = {
            "schema_version": SCHEMA_VERSION, "name": scenario.name,
            "crs": scenario.crs, "transform": list(scenario.transform),
            "start_utc": scenario.start_utc, "duration_s": scenario.duration_s,
            "weather_step_s": scenario.weather_step_s, "seed": scenario.seed,
            "resources": scenario.resources, "provenance": scenario.provenance,
            **artifact_names, "source_hashes": hashes,
        }
        (destination / "scenario.json").write_bytes(_json_bytes(manifest))
    except BaseException:
        shutil.rmtree(destination)
        raise
