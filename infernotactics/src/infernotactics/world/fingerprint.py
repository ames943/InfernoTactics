"""Stable fingerprints for generated worlds and their source assets."""

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_fingerprint(grid_path: str | Path, metadata_path: str | Path) -> str:
    payload = {
        "grid_sha256": file_sha256(grid_path),
        "metadata_sha256": file_sha256(metadata_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
