import pytest

from conversational_mcp.tools import ArgumentExtractor, ToolFlowManager, build_demo_registry


@pytest.mark.asyncio
async def test_flow_collects_fields_confirms_and_executes():
    calls = []

    async def executor(name, arguments):
        calls.append((name, arguments))
        return {"status": "completed", "message": "Created", "data": arguments}

    manager = ToolFlowManager(
        build_demo_registry(), executor,
        extractor=ArgumentExtractor(lambda _: '{"title":"Write public demo"}'),
    )
    first = await manager.start("demo-user", "create_task", "Create a task")
    assert first.state == "confirming"
    assert first.tool_name == "create_task"

    completed = await manager.confirm(first.session_id, True)
    assert completed.state == "done"
    assert calls == [("create_task", {"title": "Write public demo", "due_date": None, "priority": "normal"})]


def test_argument_extractor_uses_key_value_fallback():
    registry = build_demo_registry()
    result = ArgumentExtractor().extract(
        tool_name="create_task", schema=registry.get("create_task").schema,
        message="title=Publish example, priority=high",
    )
    assert result == {"title": "Publish example", "priority": "high"}
