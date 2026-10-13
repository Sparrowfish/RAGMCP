# Conversational MCP

`conversational_mcp` is a domain-neutral, multi-turn workflow engine for MCP
tools. It is designed for applications where an LLM selects a tool, collects
its arguments over several messages, and asks for explicit confirmation before
performing an external action.

```text
tool metadata + Pydantic schema
      → LLM JSON extraction
      → normalization + validation
      → missing-field collection
      → confirmation
      → MCP execution
```

## Components

- `tools/tool_flow_manager.py` — runtime state machine
- `tools/argument_extractor.py` — LLM-agnostic JSON extraction with safe fallback
- `tools/tool_metadata.py` and `tools/tool_registry.py` — tool contracts and discovery metadata
- `tools/tools_pydantic.py` — safe mock input schemas
- `tools/normalization.py` and `tools/validation.py` — reusable input processing
- `tools/tool_session.py` and `tools/session_store.py` — multi-turn state
- `tools/mcp_executor.py` — MCP client adapter and result normalization
- `tools/tracing.py` — structured tracing seam
- `mock_server.py` — a local server exposing task, calendar, and email mocks

No business-system integration, credentials, or organization-specific logic is
included. Replace the mock server with any MCP-compatible stdio server.

## Running the mock server

```bash
python -m pip install "mcp>=1,<2"
python conversational_mcp/mock_server.py
```

The parent chatbot project discovers its tools at startup and creates the
workflow definitions from their MCP schemas.
