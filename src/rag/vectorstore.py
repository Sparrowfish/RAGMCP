# src/rag/vectorstore.py
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger
from src.rag.chunker import Chunk


class VectorStore:
    """
    FAISS-backed vector store with a parallel metadata list.
    FAISS stores the vectors; we store the text + metadata ourselves.
    """

    def __init__(self, embedding_model: str, vectorstore_path: str):
        self.embedding_model_name = embedding_model
        self.vectorstore_path = Path(vectorstore_path)
        self.vectorstore_path.mkdir(parents=True, exist_ok=True)

        self._index_path = self.vectorstore_path / "index.faiss"
        self._meta_path  = self.vectorstore_path / "metadata.pkl"

        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.dim = self.embedder.get_sentence_embedding_dimension()

        self.index: faiss.IndexFlatIP | None = None  # Inner Product = cosine on normalized vecs
        self.metadata: list[dict] = []               # parallel list to FAISS index rows

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed all chunks and build a fresh FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided to build vector store.")

        texts = [chunk.content for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # required for cosine similarity via inner product
            convert_to_numpy=True,
        ).astype("float32")

        # IndexFlatIP: exact nearest neighbour, inner product
        # On normalized vectors this equals cosine similarity
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)

        # Store chunk text + metadata so we can return it at retrieval time
        self.metadata = [
            {"content": chunk.content, "source": chunk.source, **chunk.metadata}
            for chunk in chunks
        ]

        logger.success(f"Vector store built: {self.index.ntotal} vectors | dim={self.dim}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build() first.")

        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        logger.success(f"Vector store saved to {self.vectorstore_path}")

    def load(self) -> None:
        """Load existing FAISS index and metadata from disk."""
        if not self._index_path.exists():
            raise FileNotFoundError(f"No vector store found at {self.vectorstore_path}. Run ingestion first.")

        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        logger.success(f"Vector store loaded: {self.index.ntotal} vectors")

    def exists(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Embed a query and return the top_k most similar chunks.
        Returns list of dicts with 'content', 'source', 'score'.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load() or build() first.")

        query_vec = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for empty slots
                continue
            results.append({
                **self.metadata[idx],
                "score": float(score),
            })

        return results