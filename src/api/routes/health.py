# src/api/routes/health.py
import torch
from fastapi import APIRouter
from src.api.schemas import HealthResponse
from src.api.dependencies import state

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health() -> HealthResponse:
    vram = None
    if torch.cuda.is_available():
        vram = round(torch.cuda.memory_allocated() / 1e9, 2)

    return HealthResponse(
        status="ok",
        model_loaded=state.model_loaded,
        vectorstore_loaded=state.vectorstore_loaded,
        mcp_loaded=state.mcp_loaded,
        vram_used_gb=vram,
    )
