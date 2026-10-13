from __future__ import annotations

import json
from typing import Any

from .models import ToolResult


class MCPExecutor:
    """Adapter for an already-connected MCP session.

    The dependency on the MCP SDK is intentionally optional; pass any object
    exposing an async ``call_tool(name, arguments)`` method.
    """

    def __init__(self, client: Any):
        self.client = client

    async def __call__(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        response = await self.client.call_tool(tool_name, arguments)
        if isinstance(response, ToolResult):
            return response
        if isinstance(response, dict):
            return ToolResult(
                status=response.get("status", "completed"),
                message=response.get("message", ""),
                data=response.get("data"),
                tool_name=response.get("tool_name", tool_name),
                error_code=response.get("error_code"),
            )
        content = getattr(response, "content", [])
        for item in content:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return ToolResult("completed", text, tool_name=tool_name)
            if isinstance(payload, dict):
                return await self.__call__(tool_name, payload)
        return ToolResult("completed", tool_name=tool_name)
