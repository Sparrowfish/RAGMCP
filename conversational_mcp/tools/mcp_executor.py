"""Normalize MCP responses for ``ToolFlowManager`` without domain assumptions."""
import json
from typing import Any
from .mcp_client import MCPClient


class MCPToolExecutor:
    def __init__(self, client: MCPClient):
        self.client = client

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.call_tool(tool_name, arguments)
        return normalize_mcp_result(response, tool_name)


def normalize_mcp_result(response: Any, tool_name: str) -> dict[str, Any]:
    if isinstance(response, dict):
        payload = dict(response)
    else:
        payload = {"status": "completed", "tool_name": tool_name, "data": None, "message": ""}
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
