from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from conversational_mcp import ConversationEngine, ToolDefinition, ToolRegistry, ToolResult
from src.mcp.integration import registry_from_mcp_tools


class EchoInput(BaseModel):
    text: str


@pytest.mark.asyncio
async def test_engine_can_start_a_tool_selected_by_the_chat_router():
    async def executor(name, arguments):
        return ToolResult("completed", "Done", arguments, name)

    engine = ConversationEngine(
        ToolRegistry([ToolDefinition("echo", EchoInput, "Echo text")]),
        executor,
        extractor=lambda message, tool_name: {"text": message},
    )

    response = await engine.start_for_tool("user-1", "hello", "echo")

    assert response.tool_name == "echo"
    assert response.state.value == "done"
    assert response.result.data == {"text": "hello"}


def test_registry_is_created_from_mcp_tool_schema():
    tool = SimpleNamespace(
        name="create_note",
        description="Create a note",
        inputSchema={
            "type": "object",
            "properties": {"title": {"type": "string"}, "pinned": {"type": "boolean"}},
            "required": ["title"],
        },
    )

    registry = registry_from_mcp_tools([tool])
    definition = registry.get("create_note")

    assert definition.requires_confirmation is True
    assert definition.input_model(title="A note").title == "A note"
    with pytest.raises(Exception):
        definition.input_model()
