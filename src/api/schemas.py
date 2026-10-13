# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    system_prompt: Optional[str] = Field(
        default="You are a helpful assistant.",
        description="Override the system prompt"
    )
    use_rag: bool = Field(default=True, description="Whether to retrieve context from vector store")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="Override top_k retrieval")
    max_new_tokens: Optional[int] = Field(default=None, ge=1, le=2048)
    temperature: Optional[float] = Field(default=None, ge=0.01, le=2.0)
    use_tools: bool = Field(default=True, description="Allow the model to use MCP tools")
    user_id: str = Field(default="anonymous", min_length=1, max_length=128)
    tool_session_id: Optional[str] = Field(default=None, description="Continue a pending MCP tool session")
    tool_confirmation: Optional[bool] = Field(default=None, description="Confirm or cancel a pending MCP action")


class SourceDocument(BaseModel):
    content: str
    source: str
    score: float


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceDocument] = []
    rag_used: bool
    tokens_generated: int
    tool_used: bool = False
    tool_name: Optional[str] = None
    tool_session_id: Optional[str] = None
    tool_state: Optional[str] = None
    tool_result: Optional[dict] = None


class IngestRequest(BaseModel):
    input_dir: str = Field(default="data/raw/", description="Directory to ingest from")


class IngestResponse(BaseModel):
    status: str
    vectors_stored: int
    message: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    vectorstore_loaded: bool
    mcp_loaded: bool = False
    vram_used_gb: Optional[float] = None
