from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Generic, Type, TypeVar
from uuid import uuid4

from pydantic import BaseModel


ModelT = TypeVar("ModelT", bound=BaseModel)


class SessionState(str, Enum):
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class ToolDefinition(Generic[ModelT]):
    """The single conversational definition of an executable tool."""

    name: str
    input_model: Type[ModelT]
    description: str
    examples: list[str] = field(default_factory=list)
    field_questions: dict[str, str] = field(default_factory=dict)
    field_labels: dict[str, str] = field(default_factory=dict)
    requires_confirmation: bool = False
    success_message: str = "Operation completed."
    normalizers: dict[str, Callable[[Any], Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    status: str
    message: str = ""
    data: Any = None
    tool_name: str | None = None
    error_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "status": self.status,
            "message": self.message,
            "data": self.data,
        }
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.error_code:
            result["error_code"] = self.error_code
        return result


@dataclass(slots=True)
class ToolSession:
    user_id: str
    tool_name: str
    session_id: str = field(default_factory=lambda: str(uuid4()))
    arguments: dict[str, Any] = field(default_factory=dict)
    state: SessionState = SessionState.COLLECTING
    current_field: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class EngineResponse:
    """Stable response shape for a CLI, HTTP API, or chat adapter."""

    session_id: str | None
    state: SessionState | None
    message: str
    tool_name: str | None = None
    missing_fields: list[str] = field(default_factory=list)
    result: ToolResult | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state": self.state.value if self.state else None,
            "message": self.message,
            "tool_name": self.tool_name,
            "missing_fields": self.missing_fields,
            "result": self.result.as_dict() if self.result else None,
        }
