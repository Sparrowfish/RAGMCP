# src/api/main.py
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.model.loader import load_model, load_tokenizer
from src.rag.vectorstore import VectorStore
from src.api.dependencies import state
from src.api.routes import chat, health, ingest
from conversational_mcp.stdio import MCPStdioClient
from src.mcp.integration import build_mcp_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + vector store once at startup. Clean up on shutdown."""
    cfg = load_config()
    setup_logger(cfg["logging"]["level"], cfg["logging"]["log_dir"])
    state.cfg = cfg

    # --- Load model ---
    logger.info("Server starting up — loading model...")
    state.tokenizer = load_tokenizer(cfg)
    state.model = load_model(cfg)

    # --- Load vector store if it exists ---
    vs = VectorStore(
        embedding_model=cfg["rag"]["embedding_model"],
        vectorstore_path=cfg["rag"]["vectorstore_path"],
    )
    if vs.exists():
        vs.load()
        state.vectorstore = vs
        logger.success("Vector store loaded at startup.")
    else:
        logger.warning("No vector store found. POST /ingest to build one.")

    mcp_cfg = cfg.get("mcp", {})
    if mcp_cfg.get("enabled", False):
        try:
            client = MCPStdioClient()
            await client.connect(mcp_cfg["server_path"])
            state.mcp_engine = await build_mcp_engine(client, state.model, state.tokenizer)
            state.mcp_client = client
            logger.success(f"Connected to MCP server with {len(state.mcp_engine.registry.all())} tools.")
        except Exception as exc:
            logger.exception(f"MCP setup failed; tools are disabled: {exc}")

    yield  # server is running

    # --- Shutdown ---
    logger.info("Shutting down. Freeing VRAM...")
    del state.model
    del state.tokenizer
    if state.mcp_client:
        await state.mcp_client.close()
    torch.cuda.empty_cache()


app = FastAPI(
    title="LLM RAG Chatbot",
    description="Local LLM with Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(ingest.router)
