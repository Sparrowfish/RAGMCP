"""Reusable collect → normalize → validate → confirm → execute workflow."""
from collections.abc import Awaitable, Callable
from typing import Any

from .argument_extractor import ArgumentExtractor
from .response_generator import FlowResponse, confirmation_message
from .session_store import InMemorySessionStore
from .tool_registry import ToolRegistry
from .tool_session import SessionState, ToolSession
from .tracing import trace
from .validation import missing_fields, validate_complete, validate_partial

Executor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolFlowManager:
    def __init__(self, registry: ToolRegistry, executor: Executor, *, extractor: ArgumentExtractor | None = None,
                 store: InMemorySessionStore | None = None, max_field_retries: int = 3):
        self.registry, self.executor = registry, executor
        self.extractor = extractor or ArgumentExtractor()
        self.store = store or InMemorySessionStore()
        self.max_field_retries = max_field_retries

    async def start(self, user_id: str, tool_name: str, message: str) -> FlowResponse:
        if tool_name not in {item.metadata.name for item in self.registry.all()}:
            return FlowResponse("Unknown tool.", None, None, tool_name, [])
        session = ToolSession(user_id=user_id, tool_name=tool_name)
        await self.store.save(session)
        return await self.handle(session.session_id, message)

    async def handle(self, session_id: str, message: str) -> FlowResponse:
        session = await self.store.get(session_id)
        if not session:
            return FlowResponse("This tool session has expired.", None, None, None, [])
        if session.state not in {SessionState.COLLECTING}:
            return self._response(session, "This tool session is waiting for confirmation or is already closed.")
        tool = self.registry.get(session.tool_name)
        with trace("tool_arguments", tool_name=session.tool_name, session_id=session_id):
            extracted = self.extractor.extract(tool_name=session.tool_name, schema=tool.schema, message=message,
                                               hints=tool.metadata.extraction_hints)
        session.history.append({"role": "user", "content": message})
        for field, value in extracted.items():
            if field in tool.schema.model_fields:
                session.arguments[field] = value
        session.arguments = {key: tool.metadata.normalizers[key](value) if key in tool.metadata.normalizers else value
                             for key, value in session.arguments.items()}
        partial = validate_partial(tool.schema, session.arguments)
        if not partial["valid"]:
            field = partial["errors"][0]["field"]
            session.retry_counts[field] = session.retry_counts.get(field, 0) + 1
            await self.store.save(session)
            return self._response(session, tool.metadata.field_retry_messages.get(field, f"Invalid value for {field}."))
        required = missing_fields(tool.schema, session.arguments)
        if required:
            session.current_field = required[0]
            await self.store.save(session)
            return self._response(session, tool.metadata.field_questions.get(required[0], f"Please provide {required[0]}."), required)
        try:
            session.arguments = validate_complete(tool.schema, session.arguments)
        except ValueError:
            await self.store.save(session)
            return self._response(session, "One or more values are invalid. Please correct them.")
        if tool.metadata.requires_confirmation:
            session.state = SessionState.CONFIRMING
            await self.store.save(session)
            return self._response(session, confirmation_message(session.tool_name, session.arguments))
        return await self.confirm(session_id, True)

    async def confirm(self, session_id: str, accepted: bool) -> FlowResponse:
        session = await self.store.get(session_id)
        if not session:
            return FlowResponse("This tool session has expired.", None, None, None, [])
        if not accepted:
            session.state = SessionState.CANCELLED
            await self.store.save(session)
            return self._response(session, "Operation cancelled.")
        if session.state != SessionState.CONFIRMING:
            return self._response(session, "There is no operation waiting for confirmation.")
        session.state = SessionState.EXECUTING
        await self.store.save(session)
        try:
            with trace("tool_execute", tool_name=session.tool_name, session_id=session_id):
                result = await self.executor(session.tool_name, session.arguments)
            session.result = result
            session.state = SessionState.FAILED if result.get("status") in {"error", "failed"} else SessionState.DONE
            await self.store.save(session)
            message = result.get("message") or self.registry.get(session.tool_name).metadata.success_message
            return self._response(session, message)
        except Exception as exc:
            session.state, session.result = SessionState.FAILED, {"status": "error", "message": str(exc)}
            await self.store.save(session)
            return self._response(session, str(exc))

    def _response(self, session: ToolSession, message: str, missing: list[str] | None = None) -> FlowResponse:
        return FlowResponse(message, session.session_id, session.state.value, session.tool_name, missing or [], session.result)
