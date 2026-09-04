import numpy as np
import pytest

from infernotactics.domain.actions import DispatchAction, parse_dispatch_actions


RESOURCE_TYPES = ("water_team", "helicopter")


def test_dispatch_actions_validate_and_normalize_boundary_values():
    actions = parse_dispatch_actions(
        [("water_team", np.int64(3)), None, ("noop", None)],
        resource_types=RESOURCE_TYPES,
        n_zones=4,
    )
    assert actions == (
        DispatchAction("water_team", 3),
        DispatchAction(None, None),
        DispatchAction(None, None),
    )


@pytest.mark.parametrize(
    "actions",
    [
        ("water_team", 0),
        [["water_team", 0]],
        [("unknown", 0)],
        [("water_team", 4)],
    ],
)
def test_invalid_dispatch_actions_fail_at_the_boundary(actions):
    with pytest.raises(ValueError):
        parse_dispatch_actions(actions, resource_types=RESOURCE_TYPES, n_zones=4)
