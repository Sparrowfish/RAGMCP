# src/rag/pipeline.py
from loguru import logger
from src.rag.loader import load_documents
from src.rag.chunker import chunk_documents
from src.rag.vectorstore import VectorStore
from src.utils.config import load_config


def build_rag_pipeline(cfg: dict) -> VectorStore:
    """
    Load vector store from disk if it exists, otherwise raise.
    Use ingest() to build it first.
    """
    vs = VectorStore(
        embedding_model=cfg["rag"]["embedding_model"],
        vectorstore_path=cfg["rag"]["vectorstore_path"],
    )
    vs.load()
    return vs


def ingest(input_dir: str, cfg: dict) -> VectorStore:
    """
    Full ingestion pipeline:
    load files → chunk → embed → save to disk
    """
    documents = load_documents(input_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=cfg["rag"]["chunk_size"],
        chunk_overlap=cfg["rag"]["chunk_overlap"],
    )

    vs = VectorStore(
        embedding_model=cfg["rag"]["embedding_model"],
        vectorstore_path=cfg["rag"]["vectorstore_path"],
    )
    vs.build(chunks)
    vs.save()
    return vs


def retrieve_context(query: str, vs: VectorStore, top_k: int = 3) -> str:
    """
    Retrieve top_k chunks and format them as a single context string
    ready to be injected into the LLM prompt.
    """
    results = vs.search(query, top_k=top_k)

    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] (source: {r['source']}, score: {r['score']:.3f})\n{r['content']}")

    return "\n\n---\n\n".join(parts)