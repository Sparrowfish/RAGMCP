"""LLM-agnostic structured extraction with strict JSON parsing and safe fallback."""
from __future__ import annotations
import json
import re
from collections.abc import Callable
from typing import Any
from pydantic import BaseModel


JSONGenerator = Callable[[str], str]
_PAIR = re.compile(r"(?P<key>[A-Za-z_][\w-]*)\s*=\s*(?P<value>[^,;]+)")


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object emitted by an LLM; never execute or repair code."""
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def build_extraction_prompt(tool_name: str, schema: type[BaseModel], message: str, hints: dict[str, str]) -> str:
    hint_text = "\n".join(f"- {field}: {hint}" for field, hint in hints.items()) or "- none"
    return (
        "Extract only values explicitly stated by the user. Return one JSON object and no prose. "
        "Do not infer missing required values.\n"
        f"Tool: {tool_name}\nSchema: {json.dumps(schema.model_json_schema())}\n"
        f"Field hints:\n{hint_text}\nUser message: {message}"
    )


class ArgumentExtractor:
    def __init__(self, generate: JSONGenerator | None = None):
        self.generate = generate

    def extract(self, *, tool_name: str, schema: type[BaseModel], message: str,
                hints: dict[str, str] | None = None) -> dict[str, Any]:
        if self.generate is not None:
            result = extract_json(self.generate(build_extraction_prompt(tool_name, schema, message, hints or {})))
            if result:
                return result
        return {match.group("key"): match.group("value").strip() for match in _PAIR.finditer(message)}
