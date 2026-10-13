from types import SimpleNamespace

import pytest

from conversational_mcp.tools import ArgumentExtractor, ToolFlowManager
from src.mcp.integration import registry_from_mcp_tools


@pytest.mark.asyncio
async def test_discovered_mcp_tool_uses_the_runtime_flow_manager():
    tool = SimpleNamespace(
        name="echo",
        description="Echo text",
        inputSchema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
    )
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return {"status": "completed", "message": "Done", "data": arguments}

    flow = ToolFlowManager(
        registry_from_mcp_tools([tool]), executor,
        extractor=ArgumentExtractor(lambda _: '{"text":"hello"}'),
    )
    pending = await flow.start("user-1", "echo", "hello")
    assert pending.state == "confirming"

    response = await flow.confirm(pending.session_id, True)
    assert response.state == "done"
    assert calls == [("echo", {"text": "hello"})]


def test_discovered_schema_marks_tools_for_confirmation():
    tool = SimpleNamespace(
        name="create_note", description="Create a note",
        inputSchema={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    )
    definition = registry_from_mcp_tools([tool]).get("create_note")
    assert definition.metadata.requires_confirmation is True
    assert definition.schema(title="A note").title == "A note"
