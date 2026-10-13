from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any


class MCPStdioClient:
    """Minimal MCP stdio transport with lazy SDK imports.

    Install the optional ``mcp`` extra to use this adapter. Keeping the import
    lazy means the conversation engine remains useful for HTTP and mock tools.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self._session: Any = None

    async def connect(self, server_path: str | Path) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        path = Path(server_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(path)],
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=str(path.parent),
        )
        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self._session.initialize()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        return await self._session.call_tool(name, arguments)

    async def list_tools(self) -> list[Any]:
        if self._session is None:
            raise RuntimeError("MCP client is not connected")
        response = await self._session.list_tools()
        return list(response.tools)

    async def close(self) -> None:
        await self._stack.aclose()
        self._session = None
