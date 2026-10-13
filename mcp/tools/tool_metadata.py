"""Portable metadata describing how a conversational tool should behave."""
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class ToolMetadata:
    name: str
    title: str
    description: str
    examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    field_questions: dict[str, str] = field(default_factory=dict)
    field_retry_messages: dict[str, str] = field(default_factory=dict)
    field_labels: dict[str, str] = field(default_factory=dict)
    extraction_hints: dict[str, str] = field(default_factory=dict)
    normalizers: dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    success_message: str = "Operation completed."
    requires_confirmation: bool = True
    priority: int = 0

    def prompt_block(self) -> str:
        examples = "; ".join(self.examples) or "none"
        excluded = "; ".join(self.negative_examples) or "none"
        return (
            f"Tool: {self.name}\nTitle: {self.title}\nDescription: {self.description}\n"
            f"Examples: {examples}\nNot this tool: {excluded}"
        )
