"""Optional embedding text for semantic tool routing."""
from .tool_registry import RegisteredTool


def build_embedding_text(tool: RegisteredTool) -> str:
    metadata = tool.metadata
    return "\n".join([metadata.title, metadata.description, *metadata.examples, *metadata.negative_examples])
