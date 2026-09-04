import numpy as np
import torch

from infernotactics.training.checkpoints import (
    build_training_checkpoint,
    checkpoint_metadata,
    load_checkpoint,
    model_state_from_checkpoint,
    restore_training_state,
)
from infernotactics.training.normalization import RunningMeanStd


def test_full_training_checkpoint_round_trip(tmp_path):
    torch.manual_seed(12)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    normalizer = RunningMeanStd()
    normalizer.update([10.0, 20.0, 30.0])
    rng = np.random.default_rng(99)
    _ = rng.integers(1000)

    checkpoint = build_training_checkpoint(
        model=model,
        optimizer=optimizer,
        normalizer=normalizer,
        episode=17,
        sampling_rng=rng,
        metadata={"world_fingerprint": "world-abc"},
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    checkpoint = load_checkpoint(path)

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
    restored_normalizer = RunningMeanStd()
    restored_rng = np.random.default_rng(1)
    completed = restore_training_state(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        normalizer=restored_normalizer,
        sampling_rng=restored_rng,
    )

    assert completed == 17
    assert checkpoint_metadata(checkpoint)["world_fingerprint"] == "world-abc"
    assert restored_normalizer.state_dict() == normalizer.state_dict()
    assert restored_rng.bit_generator.state == rng.bit_generator.state
    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(expected, actual)


def test_legacy_weight_only_checkpoint_is_supported():
    model = torch.nn.Linear(2, 1)
    legacy = model.state_dict()
    assert model_state_from_checkpoint(legacy) is legacy
    assert checkpoint_metadata(legacy) == {}

    restored = torch.nn.Linear(2, 1)
    completed = restore_training_state(legacy, model=restored)
    assert completed is None
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
