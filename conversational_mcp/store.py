from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from .models import ToolSession


class InMemorySessionStore:
    """Thread-safe store suitable for a demo or a single-process service."""

    def __init__(self, timeout: timedelta = timedelta(minutes=10)):
        self._sessions: dict[str, ToolSession] = {}
        self._timeout = timeout
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> ToolSession | None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if datetime.now(timezone.utc) - session.updated_at > self._timeout:
                self._sessions.pop(session_id, None)
                return None
            return session

    async def save(self, session: ToolSession) -> None:
        session.updated_at = datetime.now(timezone.utc)
        async with self._lock:
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
