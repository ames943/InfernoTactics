"""Validated domain representation of containment dispatch requests."""

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable


@dataclass(frozen=True, slots=True)
class DispatchAction:
    resource_type: str | None
    target_zone: int | None

    @property
    def is_noop(self) -> bool:
        return self.resource_type is None

    def as_tuple(self) -> tuple[str | None, int | None]:
        return self.resource_type, self.target_zone


def parse_dispatch_actions(
    actions: Iterable[tuple[str | None, int | None] | None],
    *,
    resource_types: tuple[str, ...],
    n_zones: int,
) -> tuple[DispatchAction, ...]:
    """Validate external tuple actions once at the environment boundary."""
    if not isinstance(actions, list):
        raise ValueError("actions must be a list of (resource_type, target_zone) pairs")
    parsed = []
    for action in actions:
        if action is None:
            parsed.append(DispatchAction(None, None))
            continue
        if not isinstance(action, tuple) or len(action) != 2:
            raise ValueError("each action must be a (resource_type, target_zone) tuple or None")
        resource_type, target_zone = action
        if resource_type is None or resource_type == "noop":
            parsed.append(DispatchAction(None, None))
            continue
        if resource_type not in resource_types:
            raise ValueError(f"Unknown resource_type {resource_type!r}; expected one of {resource_types}")
        if not isinstance(target_zone, Integral) or not 0 <= int(target_zone) < n_zones:
            raise ValueError(f"target_zone {target_zone} out of range [0, {n_zones})")
        parsed.append(DispatchAction(resource_type, int(target_zone)))
    return tuple(parsed)
