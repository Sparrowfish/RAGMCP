from __future__ import annotations

from pydantic import BaseModel, Field

from conversational_mcp import ConversationEngine, ToolDefinition, ToolRegistry, ToolResult


class CreateTask(BaseModel):
    title: str = Field(min_length=3)
    due_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class CreateEvent(BaseModel):
    title: str = Field(min_length=3)
    start: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$")


registry = ToolRegistry([
    ToolDefinition(
        name="create_task",
        input_model=CreateTask,
        description="Create a task with a due date.",
        examples=["create task", "new task"],
        field_questions={"title": "What is the task title?", "due_date": "What is the due date (YYYY-MM-DD)?"},
    ),
    ToolDefinition(
        name="create_event",
        input_model=CreateEvent,
        description="Schedule a calendar event.",
        examples=["create event", "schedule meeting"],
        field_questions={"title": "What should the event be called?", "start": "When does it start (YYYY-MM-DDTHH:MM)?"},
        requires_confirmation=True,
    ),
])


async def mock_executor(tool_name: str, arguments: dict) -> ToolResult:
    return ToolResult("completed", f"{tool_name} executed.", data=arguments, tool_name=tool_name)


def build_engine() -> ConversationEngine:
    return ConversationEngine(registry, mock_executor)
