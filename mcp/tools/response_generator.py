"""Transport-neutral response shaping for tool workflows."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FlowResponse:
    message: str
    session_id: str | None
    state: str | None
    tool_name: str | None
    missing_fields: list[str]
    result: dict[str, Any] | None = None


def confirmation_message(tool_name: str, arguments: dict[str, Any]) -> str:
    values = ", ".join(f"{key}={value}" for key, value in arguments.items())
    return f"Ready to execute {tool_name} ({values}). Confirm?"
