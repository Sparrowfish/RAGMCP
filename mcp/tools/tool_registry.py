"""Registry that binds schemas and conversational metadata without provider code."""
from dataclasses import dataclass
from typing import Type
from pydantic import BaseModel

from .normalization import normalize_date, normalize_datetime, normalize_email_list, normalize_text
from .tool_metadata import ToolMetadata
from .tools_pydantic import CreateCalendarEventArgs, CreateTaskArgs, SendEmailArgs


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    schema: Type[BaseModel]
    metadata: ToolMetadata


class ToolRegistry:
    def __init__(self, tools: list[RegisteredTool] | None = None):
        self._tools: dict[str, RegisteredTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: RegisteredTool) -> None:
        if tool.metadata.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.metadata.name}")
        self._tools[tool.metadata.name] = tool

    def get(self, name: str) -> RegisteredTool:
        return self._tools[name]

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def prompt_block(self) -> str:
        return "\n\n".join(item.metadata.prompt_block() for item in self.all())


def build_demo_registry() -> ToolRegistry:
    return ToolRegistry([
        RegisteredTool(CreateTaskArgs, ToolMetadata(
            name="create_task", title="Create task", description="Create a task in a task tracker.",
            examples=["create a task", "add a task", "remind me to prepare the report"],
            field_questions={"title": "What should the task be called?"},
            normalizers={"title": normalize_text, "due_date": normalize_date, "priority": normalize_text},
        )),
        RegisteredTool(CreateCalendarEventArgs, ToolMetadata(
            name="create_calendar_event", title="Create calendar event", description="Schedule a calendar event.",
            examples=["schedule a meeting", "add an event tomorrow at 10"],
            negative_examples=["create a task"],
            field_questions={"title": "What is the event title?", "start": "When does it start?"},
            normalizers={"title": normalize_text, "start": normalize_datetime, "attendees": normalize_email_list},
        )),
        RegisteredTool(SendEmailArgs, ToolMetadata(
            name="send_email", title="Send email", description="Send an email to one or more recipients.",
            examples=["send an email", "email the project update to alice@example.com"],
            field_questions={"to": "Who should receive the email?", "subject": "What is the subject?", "body": "What should the email say?"},
            normalizers={"to": normalize_email_list, "subject": normalize_text, "body": normalize_text},
        )),
    ])
