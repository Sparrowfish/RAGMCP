# Conversational MCP

`conversational_mcp` is a domain-agnostic orchestration layer for conversational MCP clients. It turns a natural-language interaction into a controlled tool call:

```text
route → collect arguments → normalize → validate → confirm → execute
```

The package does not contain business-domain rules, provider credentials, or a particular LLM integration. Applications register their own Pydantic input models and MCP executor, while the engine manages multi-turn state and side-effect confirmation.

The `tools/` package is a fuller reference architecture for projects that need
metadata, LLM-backed JSON extraction, normalization,
partial/final validation, session storage, structured tracing, and a tool-flow
manager. It contains only safe mock examples: task creation, calendar events,
and email sending.

## Quick start

From the repository root:

```bash
python -m pip install pydantic pytest pytest-asyncio
pytest conversational_mcp/tests -q
python -m conversational_mcp.demo
```

The offline example uses `field=value` extraction so it can run without an API key. Replace `KeyValueExtractor` with an LLM-backed callable when integrating a production router/extractor.

To run the mock MCP server after installing the optional MCP dependency:

```bash
python -m pip install "mcp>=1,<2"
python conversational_mcp/mock_server.py
```

## Extension points

- `ToolRegistry` is the single source of tool definitions.
- `ConversationEngine` accepts custom routing, extraction, normalization, and execution dependencies.
- `MCPExecutor` adapts any async MCP client exposing `call_tool`.
- `MCPStdioClient` provides an optional stdio transport through the `mcp` extra.
- `tools.ToolFlowManager` demonstrates a provider-neutral workflow with explicit confirmation.
- Localization and domain-specific tools belong in application packages, not in the core.

The included productivity example demonstrates `create_task` and `create_event`; it is intentionally provider-neutral and safe to run locally.
