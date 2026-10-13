from __future__ import annotations

from typing import Any

from .models import ToolDefinition


class ToolRegistry:
    """Registry used by routing, validation, prompting, and execution."""

    def __init__(self, definitions: list[ToolDefinition[Any]] | None = None):
        self._definitions: dict[str, ToolDefinition[Any]] = {}
        for definition in definitions or []:
            self.register(definition)

    def register(self, definition: ToolDefinition[Any]) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def get(self, name: str) -> ToolDefinition[Any]:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def all(self) -> list[ToolDefinition[Any]]:
        return list(self._definitions.values())

    def prompt_block(self) -> str:
        """Create an LLM-neutral description block for a router prompt."""
        blocks = []
        for tool in self.all():
            examples = ", ".join(tool.examples) or "none"
            blocks.append(
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Examples: {examples}"
            )
        return "\n\n".join(blocks)
