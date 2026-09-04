"""Thread-safe ownership and lifecycle for independent simulation sessions."""

from collections.abc import Callable
import threading
from typing import Generic, TypeVar
from uuid import uuid4


SessionT = TypeVar("SessionT")


class SessionNotFoundError(KeyError):
    pass


class SessionCapacityError(RuntimeError):
    pass


class SessionManager(Generic[SessionT]):
    """Own independent stateful sessions behind stable opaque identifiers."""

    def __init__(self, factory: Callable[[], SessionT], *, max_sessions: int = 8):
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self._factory = factory
        self._max_sessions = max_sessions
        self._sessions: dict[str, SessionT] = {}
        self._lock = threading.RLock()

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def create(self, session_id: str | None = None) -> tuple[str, SessionT]:
        identifier = session_id or uuid4().hex
        if not identifier or len(identifier) > 128:
            raise ValueError("session_id must contain 1-128 characters")
        with self._lock:
            if identifier in self._sessions:
                raise ValueError(f"Session already exists: {identifier}")
            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityError(
                    f"Session capacity reached ({self._max_sessions}); delete an inactive session first"
                )
            session = self._factory()
            self._sessions[identifier] = session
            return identifier, session

    def get(self, session_id: str) -> SessionT:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as error:
                raise SessionNotFoundError(session_id) from error

    def remove(self, session_id: str) -> SessionT:
        with self._lock:
            try:
                return self._sessions.pop(session_id)
            except KeyError as error:
                raise SessionNotFoundError(session_id) from error

    def ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._sessions)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
