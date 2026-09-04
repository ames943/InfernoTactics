"""Versioned, backward-compatible training checkpoints."""

from pathlib import Path
from typing import Any

import torch


CHECKPOINT_FORMAT_VERSION = 2


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=True)


def model_state_from_checkpoint(checkpoint: Any) -> dict[str, Any]:
    """Extract model weights from v2 bundles or legacy raw state dictionaries."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def checkpoint_metadata(checkpoint: Any) -> dict[str, Any]:
    """Return metadata from a versioned bundle; legacy checkpoints have none."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        metadata = checkpoint.get("metadata", {})
        return metadata if isinstance(metadata, dict) else {}
    return {}


def build_training_checkpoint(
    *,
    model,
    optimizer,
    normalizer,
    episode: int,
    sampling_rng,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "episode": int(episode),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "normalizer_state_dict": normalizer.state_dict(),
        "sampling_rng_state": sampling_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "metadata": dict(metadata or {}),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def restore_training_state(
    checkpoint: Any,
    *,
    model,
    optimizer=None,
    normalizer=None,
    sampling_rng=None,
) -> int | None:
    """Restore all available state and return the completed episode number."""
    model.load_state_dict(model_state_from_checkpoint(checkpoint))
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return None
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if normalizer is not None and "normalizer_state_dict" in checkpoint:
        normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    if sampling_rng is not None and "sampling_rng_state" in checkpoint:
        sampling_rng.bit_generator.state = checkpoint["sampling_rng_state"]
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    episode = checkpoint.get("episode")
    return int(episode) if episode is not None else None
