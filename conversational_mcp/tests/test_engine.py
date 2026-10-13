import pytest

from conversational_mcp.engine import ConversationEngine
from conversational_mcp.examples.productivity import build_engine


@pytest.mark.asyncio
async def test_collects_fields_and_executes():
    engine = build_engine()
    first = await engine.start("u1", "create task")
    assert first.missing_fields == ["title", "due_date"]
    second = await engine.continue_session(first.session_id, "title=Write docs, due_date=2026-08-20")
    assert second.state.value == "done"
    assert second.result.data["title"] == "Write docs"


@pytest.mark.asyncio
async def test_confirmation_can_cancel():
    engine = build_engine()
    first = await engine.start("u1", "create event")
    pending = await engine.continue_session(first.session_id, "title=Demo, start=2026-08-20T10:00")
    assert pending.state.value == "confirming"
    cancelled = await engine.confirm(pending.session_id, False)
    assert cancelled.state.value == "cancelled"
