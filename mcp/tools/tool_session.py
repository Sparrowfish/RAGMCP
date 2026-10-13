"""Session state kept independently from any HTTP or CLI transport."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class SessionState(str, Enum):
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ToolSession:
    user_id: str
    tool_name: str
    session_id: str = field(default_factory=lambda: str(uuid4()))
    arguments: dict[str, Any] = field(default_factory=dict)
    state: SessionState = SessionState.COLLECTING
    current_field: str | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
    history: list[dict[str, str]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
