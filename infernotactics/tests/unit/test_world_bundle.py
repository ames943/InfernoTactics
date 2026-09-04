import json

import numpy as np
import pytest

from infernotactics.world import WorldBundle


def _write_world(tmp_path, grid, layer_names=("a", "b")):
    grid_path = tmp_path / "grid.npy"
    metadata_path = tmp_path / "meta.json"
    np.save(grid_path, grid)
    metadata_path.write_text(json.dumps({
        "layer_names": list(layer_names),
        "height": int(grid.shape[-2]),
        "width": int(grid.shape[-1]),
        "crs": "EPSG:5070",
        "transform": [30, 0, 0, 0, -30, 0],
        "cell_size_m": 30,
    }), encoding="utf-8")
    return grid_path, metadata_path


def test_world_bundle_loads_and_fingerprints_valid_assets(tmp_path):
    paths = _write_world(tmp_path, np.zeros((2, 3, 4), dtype=np.float32))
    world = WorldBundle.load(*paths, expected_layers=("a", "b"))

    assert world.grid.shape == (2, 3, 4)
    assert len(world.fingerprint) == 64


def test_world_bundle_rejects_layer_contract_mismatch(tmp_path):
    paths = _write_world(tmp_path, np.zeros((2, 3, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="layer order"):
        WorldBundle.load(*paths, expected_layers=("b", "a"))
