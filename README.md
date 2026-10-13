# Local RAG Chatbot with Conversational MCP

A local-first chatbot that lets an LLM choose the right source for each user
message: retrieved knowledge, a conversational MCP tool, or a normal model
response. It combines a 4-bit Hugging Face model, FAISS-based RAG, and a
multi-turn MCP workflow with structured argument collection and confirmation.

> This repository is a standalone technical portfolio project. The MCP tools
> are safe mocks and contain no provider credentials, business data, or
> organization-specific logic.

## Highlights

- Local inference with Mistral, Transformers, bitsandbytes, and 4-bit loading
- Retrieval-augmented generation using Nomic embeddings and FAISS
- LLM-driven decision making: `rag`, `tool`, or ordinary `chat`
- MCP tool discovery at startup from a stdio MCP server
- Multi-turn tool workflows: extract arguments, normalize, validate, request
  confirmation, execute, and return structured results
- Pydantic schemas, session storage, validation, normalization, and tracing
  hooks in the reusable `conversational_mcp` package
- Three safe mock tools: task creation, calendar-event creation, and email
  sending

## Architecture

```text
                         ┌─────────────────────┐
User message ───────────▶│ LLM request router  │
                         └─────────┬───────────┘
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 RAG path       Tool path      Chat path
                    │              │              │
             FAISS retrieval   MCP workflow   Local LLM
                    │              │              │
                    └──────────────┴──────────────┘
                                   ▼
                            API response
```

The router sees the available MCP tools and decides whether the message needs
indexed documents, an external action, or no extra source at all. When it
chooses a tool, the tool flow collects missing fields over multiple messages
and requires explicit confirmation before execution.

## Project layout

```text
src/
  api/                 FastAPI application and chat endpoint
  rag/                 document loading, chunking, embeddings, and FAISS
  model/               local model loading and generation
  mcp/                 bridge between the chatbot and MCP tools

conversational_mcp/
  engine.py            compact, transport-neutral conversation engine
  tools/               reusable workflow components and mock schemas
  mock_server.py       safe MCP server used for local demonstration

tests/                 focused unit tests for MCP integration and tool flow
```

## Quick start

Prerequisites: Python 3.11+ and, for local GPU inference, a compatible CUDA
installation.

```bash
python -m venv venv
venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install "mcp>=1,<2"
```

Set `HF_TOKEN` in `.env` if the selected Hugging Face model requires it.

### Build the knowledge base

Place documents in `data/raw/`, then run:

```bash
python scripts/ingest.py --input data/raw/
```

### Enable the mock MCP server

The API starts the configured MCP server over stdio. The included configuration
already points to the mock server. Enable it in `configs/config.yaml`:

```yaml
mcp:
  enabled: true
  server_path: "./conversational_mcp/mock_server.py"
```

### Start the API

```bash
python -m uvicorn src.api.main:app --reload
```

Check that the model, vector store, and MCP connection are ready:

```bash
curl http://127.0.0.1:8000/health
```

## Chat API

Send a message to the chat endpoint:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\": \"What does the indexed documentation say about installation?\"}"
```

For a tool-oriented request, the response may ask for missing fields or return
a confirmation request:

```json
{
  "message": "Create a task titled Publish the demo",
  "user_id": "demo-user"
}
```

The response includes `tool_session_id`. Continue the same workflow by sending
that ID with the next user message. Approve a ready tool call explicitly:

```json
{
  "message": "yes",
  "tool_session_id": "<session-id>",
  "tool_confirmation": true
}
```

Use `tool_confirmation: false` to cancel.

## Conversational MCP workflow

The reusable workflow is intentionally split into small components:

```text
tool metadata + Pydantic schema
             ↓
LLM JSON argument extraction
             ↓
normalization and partial validation
             ↓
missing-field collection
             ↓
explicit confirmation
             ↓
MCP execution and normalized result
```

The mock server exposes these tools:

| Tool | Purpose |
| --- | --- |
| `create_task` | Create a mock task with optional due date and priority |
| `create_calendar_event` | Create a mock calendar event |
| `send_email` | Queue a mock email without sending a real message |

Replace the mock server with any MCP-compatible stdio server by changing
`mcp.server_path`. The chatbot discovers its tool schemas at startup.

## Tests

```bash
venv\Scripts\python -m pytest tests conversational_mcp/tests -q
```

## License

Add a license appropriate for your public repository before publishing.
