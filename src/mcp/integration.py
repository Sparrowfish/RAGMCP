"""LLM routing and MCP tool registration for the chat API."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from conversational_mcp import ConversationEngine, ToolDefinition, ToolRegistry
from conversational_mcp.mcp import MCPExecutor
from conversational_mcp.routing import Route

RouteKind = Literal["rag", "tool", "chat"]


@dataclass(frozen=True)
class ChatRoute:
    kind: RouteKind
    tool_name: str | None = None


def _json_object(text: str) -> dict[str, Any]:
    """Extract a JSON object even when a local model adds surrounding text."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _generate(*args: Any, **kwargs: Any) -> str:
    # Keep tool schema discovery lightweight (and testable) until inference is needed.
    from src.model.inference import generate
    return generate(*args, **kwargs)


class LLMChatRouter:
    """Ask the local LLM whether retrieval, a tool, or normal chat is needed."""

    def __init__(self, model: Any, tokenizer: Any, registry: ToolRegistry):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry

    def route(self, message: str, *, rag_available: bool) -> ChatRoute:
        tools = self.registry.prompt_block() or "No tools are available."
        prompt = (
            "Classify the user request. Return JSON only, with keys kind and tool_name.\n"
            "kind must be one of: rag (answer from indexed documents), tool (perform an "
            "external action), chat (general conversation). Choose tool only if one listed tool "
            "is needed. Choose rag for questions about the indexed knowledge base.\n\n"
            f"RAG available: {rag_available}\nTools:\n{tools}\n\nUser request: {message}"
        )
        result = _json_object(_generate(
            self.model, self.tokenizer, prompt,
            system_prompt="You are a precise request router. Never answer the user request.",
            max_new_tokens=80, temperature=0.01,
        ))
        kind = result.get("kind")
        tool_name = result.get("tool_name")
        if kind == "tool" and tool_name in {tool.name for tool in self.registry.all()}:
            return ChatRoute("tool", tool_name)
        if kind == "rag" and rag_available:
            return ChatRoute("rag")
        return ChatRoute("chat")


class LLMToolRouter:
    """Conversational MCP router that honours the tool chosen by ``LLMChatRouter``."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name

    def route(self, message: str) -> Route:
        return Route(self.tool_name, 1.0)


class LLMArgumentExtractor:
    """Extract a tool call's arguments using the same local model."""

    def __init__(self, model: Any, tokenizer: Any, registry: ToolRegistry):
        self.model = model
        self.tokenizer = tokenizer
        self.registry = registry

    def __call__(self, message: str, tool_name: str) -> dict[str, Any]:
        tool = self.registry.get(tool_name)
        schema = tool.input_model.model_json_schema()
        prompt = (
            "Extract only values explicitly supplied by the user for this tool. Return a JSON "
            "object only. Do not invent missing values.\n"
            f"Tool: {tool_name}\nInput schema: {json.dumps(schema)}\nUser message: {message}"
        )
        return _json_object(_generate(
            self.model, self.tokenizer, prompt,
            system_prompt="You extract structured tool arguments accurately.",
            max_new_tokens=160, temperature=0.01,
        ))


def _python_type(schema: dict[str, Any]) -> type[Any]:
    """Map common JSON-schema types returned by MCP list_tools to Pydantic fields."""
    type_name = schema.get("type")
    if type_name == "string":
        return str
    if type_name == "integer":
        return int
    if type_name == "number":
        return float
    if type_name == "boolean":
        return bool
    if type_name == "array":
        return list[Any]
    return Any


def registry_from_mcp_tools(mcp_tools: list[Any]) -> ToolRegistry:
    """Turn MCP ``list_tools`` metadata into conversational MCP definitions."""
    definitions = []
    for tool in mcp_tools:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        fields = {
            name: (_python_type(field_schema), Field(... if name in required else None,
                                                       description=field_schema.get("description", "")))
            for name, field_schema in properties.items()
        }
        input_model = create_model(f"MCP_{tool.name}_Input", **fields, __base__=BaseModel)
        definitions.append(ToolDefinition(
            name=tool.name,
            input_model=input_model,
            description=getattr(tool, "description", "") or tool.name,
            field_questions={name: f"Please provide {name}." for name in required},
            # External MCP tools may change data, so require explicit confirmation.
            requires_confirmation=True,
        ))
    return ToolRegistry(definitions)


async def build_mcp_engine(client: Any, model: Any, tokenizer: Any) -> ConversationEngine:
    registry = registry_from_mcp_tools(await client.list_tools())
    extractor = LLMArgumentExtractor(model, tokenizer, registry)
    # The router is replaced per new request after the chat router selects a tool.
    return ConversationEngine(registry, MCPExecutor(client), extractor=extractor)
