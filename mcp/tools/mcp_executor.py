"""MCP executor used by the conversational tool-flow manager."""
from __future__ import annotations

import json
from typing import Any


class MCPToolExecutor:
    """Adapt any async MCP client exposing ``call_tool`` to the flow executor contract."""

    def __init__(self, client: Any):
        self.client = client

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.call_tool(tool_name, arguments)
        return self.normalize(response, tool_name)

    @staticmethod
    def normalize(response: Any, tool_name: str) -> dict[str, Any]:
        if isinstance(response, dict):
            payload = dict(response)
        else:
            payload = {"status": "completed", "tool_name": tool_name, "message": "", "data": None}
            for item in getattr(response, "content", []):
                text = getattr(item, "text", None)
                if not text:
                    continue
                try:
                    decoded = json.loads(text)
                except json.JSONDecodeError:
                    payload["message"] = text
                else:
                    if isinstance(decoded, dict):
                        payload.update(decoded)
                    else:
                        payload["data"] = decoded
                break
        payload.setdefault("tool_name", tool_name)
        payload["status"] = "completed" if payload.get("status") in {None, "ok", "success"} else payload["status"]
        return payload
