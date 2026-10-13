from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .models import EngineResponse, SessionState, ToolResult, ToolSession
from .registry import ToolRegistry
from .routing import KeyValueExtractor, KeywordRouter, Route
from .store import InMemorySessionStore
from .validation import missing_fields, validate_complete


Executor = Callable[[str, dict[str, Any]], Awaitable[ToolResult | dict[str, Any]]]
Extractor = Callable[[str, str], dict[str, Any]]


class ConversationEngine:
    """Coordinates routing, collection, validation, confirmation, and execution."""

    def __init__(
        self,
        registry: ToolRegistry,
        executor: Executor,
        *,
        router: KeywordRouter | None = None,
        extractor: Extractor | None = None,
        store: InMemorySessionStore | None = None,
    ):
        self.registry = registry
        self.executor = executor
        self.router = router or KeywordRouter(registry)
        self.extractor = extractor or KeyValueExtractor()
        self.store = store or InMemorySessionStore()

    async def start(self, user_id: str, message: str) -> EngineResponse:
        route = self.router.route(message)
        if route.tool_name is None:
            return EngineResponse(None, None, route.clarification or "Please clarify your request.")
        return await self.start_for_tool(user_id, message, route.tool_name)

    async def start_for_tool(self, user_id: str, message: str, tool_name: str) -> EngineResponse:
        """Start a session for a tool selected by an application-level router."""
        try:
            self.registry.get(tool_name)
        except KeyError:
            return EngineResponse(None, None, f"Unknown tool: {tool_name}.")
        session = ToolSession(user_id=user_id, tool_name=tool_name)
        await self.store.save(session)
        return await self._process(session, message)

    async def continue_session(self, session_id: str, message: str) -> EngineResponse:
        session = await self.store.get(session_id)
        if session is None:
            return EngineResponse(None, None, "This conversation has expired. Please start again.")
        if session.state in {SessionState.DONE, SessionState.CANCELLED, SessionState.FAILED}:
            return EngineResponse(session.session_id, session.state, "This conversation is already closed.", session.tool_name)
        return await self._process(session, message)

    async def confirm(self, session_id: str, accepted: bool) -> EngineResponse:
        session = await self.store.get(session_id)
        if session is None:
            return EngineResponse(None, None, "This conversation has expired. Please start again.")
        if not accepted:
            session.state = SessionState.CANCELLED
            await self.store.save(session)
            return EngineResponse(session.session_id, session.state, "Operation cancelled.", session.tool_name)
        if session.state != SessionState.CONFIRMING:
            return EngineResponse(session.session_id, session.state, "There is no operation waiting for confirmation.", session.tool_name)
        return await self._execute(session)

    async def _process(self, session: ToolSession, message: str) -> EngineResponse:
        tool = self.registry.get(session.tool_name)
        session.history.append({"role": "user", "content": message})
        values = self.extractor(message, tool.name)
        for field_name, value in values.items():
            if field_name in tool.normalizers:
                value = tool.normalizers[field_name](value)
            session.arguments[field_name] = value
        required = missing_fields(tool.input_model, session.arguments)
        if required:
            session.current_field = required[0]
            await self.store.save(session)
            field = required[0]
            question = tool.field_questions.get(field, f"Please provide {field}.")
            return EngineResponse(session.session_id, session.state, question, tool.name, required)
        try:
            session.arguments = validate_complete(tool.input_model, session.arguments)
        except ValueError as exc:
            await self.store.save(session)
            return EngineResponse(session.session_id, session.state, f"Some values are invalid: {exc}", tool.name)
        if tool.requires_confirmation:
            session.state = SessionState.CONFIRMING
            await self.store.save(session)
            return EngineResponse(session.session_id, session.state, self._summary(tool.name, session.arguments), tool.name)
        return await self._execute(session)

    async def _execute(self, session: ToolSession) -> EngineResponse:
        tool = self.registry.get(session.tool_name)
        session.state = SessionState.EXECUTING
        await self.store.save(session)
        try:
            raw = await self.executor(tool.name, session.arguments)
            if isinstance(raw, ToolResult):
                result = raw
            else:
                # Executors often return provider-specific fields. Keep the
                # public result contract stable and preserve the payload in
                # ``data`` rather than failing on an unknown response key.
                result = ToolResult(
                    status=raw.get("status", "completed"),
                    message=raw.get("message", ""),
                    data=raw.get("data", raw),
                    tool_name=raw.get("tool_name", tool.name),
                    error_code=raw.get("error_code"),
                )
            if result.status in {"error", "failed"}:
                session.state = SessionState.FAILED
            else:
                session.state = SessionState.DONE
            await self.store.save(session)
            message = result.message or tool.success_message
            return EngineResponse(session.session_id, session.state, message, tool.name, result=result)
        except Exception as exc:  # executor failures belong in the stable result shape
            session.state = SessionState.FAILED
            await self.store.save(session)
            result = ToolResult("error", str(exc), tool_name=tool.name, error_code="EXECUTION_ERROR")
            return EngineResponse(session.session_id, session.state, str(exc), tool.name, result=result)

    def _summary(self, tool_name: str, arguments: dict[str, Any]) -> str:
        pairs = ", ".join(f"{key}={value}" for key, value in arguments.items())
        return f"Ready to execute {tool_name} ({pairs}). Confirm?"
