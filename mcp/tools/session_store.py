"""Concurrency-safe in-memory session store; swap it for Redis in production."""
import asyncio
from datetime import datetime, timedelta, timezone
from .tool_session import ToolSession


class InMemorySessionStore:
    def __init__(self, timeout_minutes: int = 15, max_history: int = 50):
        self._sessions: dict[str, ToolSession] = {}
        self._timeout = timedelta(minutes=timeout_minutes)
        self._max_history = max_history
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> ToolSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and datetime.now(timezone.utc) - session.updated_at > self._timeout:
                self._sessions.pop(session_id, None)
                return None
            return session

    async def save(self, session: ToolSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        session.history = session.history[-self._max_history:]
        async with self._lock:
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
