"""Compatibility facade around the package's reusable stdio MCP client."""
from ..stdio import MCPStdioClient


class MCPClient(MCPStdioClient):
    def is_connected(self) -> bool:
        return self._session is not None
