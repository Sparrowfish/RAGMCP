from __future__ import annotations

import re
from dataclasses import dataclass

from .registry import ToolRegistry


@dataclass(slots=True)
class Route:
    tool_name: str | None
    confidence: float
    clarification: str | None = None


class KeywordRouter:
    """Dependency-free baseline router; replace with an LLM router in production."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def route(self, message: str) -> Route:
        text = message.casefold()
        scored: list[tuple[int, str]] = []
        for tool in self.registry.all():
            score = sum(1 for example in tool.examples if example.casefold() in text)
            if score:
                scored.append((score, tool.name))
        if not scored:
            return Route(None, 0.0, "I could not identify an operation to perform.")
        scored.sort(reverse=True)
        score, name = scored[0]
        confidence = min(0.95, 0.45 + score * 0.2)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            options = ", ".join(item[1] for item in scored[:3])
            return Route(None, 0.0, f"Which operation do you mean: {options}?")
        return Route(name, confidence)


class KeyValueExtractor:
    """Small offline extractor for examples and tests.

    Applications can inject an LLM-backed callable with the same interface.
    It accepts ``field=value`` pairs and a JSON object.
    """

    _pair = re.compile(r"(?P<key>[A-Za-z_][\w-]*)\s*=\s*(?P<value>[^,;]+)")

    def __call__(self, message: str, tool_name: str) -> dict[str, str]:
        text = message.strip()
        if text.startswith("{") and text.endswith("}"):
            import json

            payload = json.loads(text)
            return payload if isinstance(payload, dict) else {}
        return {match.group("key"): match.group("value").strip() for match in self._pair.finditer(text)}
