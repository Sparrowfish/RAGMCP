# src/api/dependencies.py
"""
Single shared state for the lifetime of the server.
Model and vector store are loaded once at startup — not per request.
"""
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rag.vectorstore import VectorStore
from conversational_mcp import ConversationEngine
from conversational_mcp.stdio import MCPStdioClient


@dataclass
class AppState:
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    vectorstore: Optional[VectorStore] = None
    mcp_engine: Optional[ConversationEngine] = None
    mcp_client: Optional[MCPStdioClient] = None
    cfg: dict = field(default_factory=dict)

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @property
    def vectorstore_loaded(self) -> bool:
        return self.vectorstore is not None

    @property
    def mcp_loaded(self) -> bool:
        return self.mcp_engine is not None


# Global singleton — imported by routers
state = AppState()
