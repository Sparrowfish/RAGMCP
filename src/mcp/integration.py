"""Bridge the chatbot's LLM router to the reusable conversational tool workflow."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, create_model

from conversational_mcp.tools import ArgumentExtractor, MCPToolExecutor, ToolFlowManager
from conversational_mcp.tools.tool_metadata import ToolMetadata
from conversational_mcp.tools.tool_registry import RegisteredTool, ToolRegistry

RouteKind = Literal["rag", "tool", "chat"]


@dataclass(frozen=True)
class ChatRoute:
    kind: RouteKind
    tool_name: str | None = None


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _generate(*args: Any, **kwargs: Any) -> str:
    from src.model.inference import generate
    return generate(*args, **kwargs)


class LLMChatRouter:
    """Use the local model to choose RAG, ordinary chat, or one discovered MCP tool."""

    def __init__(self, model: Any, tokenizer: Any, registry: ToolRegistry):
        self.model, self.tokenizer, self.registry = model, tokenizer, registry

    def route(self, message: str, *, rag_available: bool) -> ChatRoute:
        prompt = (
            "Classify the user request. Return JSON only with keys kind and tool_name. "
            "kind must be rag (indexed knowledge), tool (external action), or chat (general conversation). "
            "Choose tool only when one listed tool is needed.\n\n"
            f"RAG available: {rag_available}\nTools:\n{self.registry.prompt_block() or 'No tools are available.'}\n\n"
            f"User request: {message}"
        )
        result = _json_object(_generate(
            self.model, self.tokenizer, prompt,
            system_prompt="You are a precise request router. Never answer the request.",
            max_new_tokens=80, temperature=0.01,
        ))
        known_tools = {tool.metadata.name for tool in self.registry.all()}
        if result.get("kind") == "tool" and result.get("tool_name") in known_tools:
            return ChatRoute("tool", result["tool_name"])
        if result.get("kind") == "rag" and rag_available:
            return ChatRoute("rag")
        return ChatRoute("chat")


def _python_type(schema: dict[str, Any]) -> type[Any]:
    return {"string": str, "integer": int, "number": float, "boolean": bool, "array": list[Any]}.get(schema.get("type"), Any)


def registry_from_mcp_tools(mcp_tools: list[Any]) -> ToolRegistry:
    """Build rich workflow definitions from standard MCP ``list_tools`` metadata."""
    registered = []
    for tool in mcp_tools:
        schema = getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None) or {}
        properties, required = schema.get("properties", {}), set(schema.get("required", []))
        fields = {
            name: (_python_type(field_schema), Field(... if name in required else None, description=field_schema.get("description", "")))
            for name, field_schema in properties.items()
        }
        model = create_model(f"MCP_{tool.name}_Input", **fields, __base__=BaseModel)
        registered.append(RegisteredTool(
            schema=model,
            metadata=ToolMetadata(
                name=tool.name,
                title=tool.name.replace("_", " ").title(),
                description=getattr(tool, "description", "") or tool.name,
                field_questions={name: f"Please provide {name}." for name in required},
                requires_confirmation=True,
            ),
        ))
    return ToolRegistry(registered)


async def build_mcp_engine(client: Any, model: Any, tokenizer: Any) -> ToolFlowManager:
    registry = registry_from_mcp_tools(await client.list_tools())

    def generate_json(prompt: str) -> str:
        return _generate(model, tokenizer, prompt, system_prompt="Extract structured tool arguments accurately.", max_new_tokens=160, temperature=0.01)

    return ToolFlowManager(registry, MCPToolExecutor(client).execute, extractor=ArgumentExtractor(generate_json))
