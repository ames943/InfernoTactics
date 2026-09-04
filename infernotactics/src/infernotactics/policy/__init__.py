"""Runtime policy inference independent of training entry points."""

from .inference import DEFAULT_MAX_DISPATCH_SLOTS, forward_policy
from .targets import (
    NOOP_TARGET_INDEX,
    N_TARGET_TYPES,
    TARGET_FEATURE_DIM,
    TARGET_TYPES,
    decode_action,
    resolve_relative_targets,
)

__all__ = [
    "DEFAULT_MAX_DISPATCH_SLOTS",
    "NOOP_TARGET_INDEX",
    "N_TARGET_TYPES",
    "TARGET_FEATURE_DIM",
    "TARGET_TYPES",
    "decode_action",
    "forward_policy",
    "resolve_relative_targets",
]
