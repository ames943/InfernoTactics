import pytest

from infernotactics.service import (
    SessionCapacityError,
    SessionManager,
    SessionNotFoundError,
)


def test_sessions_are_independent_and_addressable():
    created = iter(({"value": 1}, {"value": 2}))
    manager = SessionManager(lambda: next(created), max_sessions=2)

    first_id, first = manager.create("default")
    second_id, second = manager.create()

    first["value"] = 10
    assert first_id == "default"
    assert second_id != first_id
    assert manager.get(first_id)["value"] == 10
    assert manager.get(second_id) is second
    assert len(manager) == 2


def test_session_capacity_and_removal_are_explicit():
    manager = SessionManager(dict, max_sessions=1)
    manager.create("default")

    with pytest.raises(SessionCapacityError):
        manager.create()
    removed = manager.remove("default")
    assert removed == {}
    with pytest.raises(SessionNotFoundError):
        manager.get("default")
