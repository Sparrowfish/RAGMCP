# src/rag/chunker.py
from dataclasses import dataclass, field
from loguru import logger
from src.rag.loader import Document


@dataclass
class Chunk:
    """A text chunk ready for embedding."""
    content: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text by characters with overlap.
    Tries to split on paragraph boundaries first, falls back to hard cut.
    """
    # Prefer splitting on double newlines (paragraph boundaries)
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Try to find a clean split point near the end of this window
        split_at = end
        for sep in separators:
            idx = text.rfind(sep, start + chunk_size // 2, end)
            if idx != -1:
                split_at = idx + len(sep)
                break

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size - overlap
        start = split_at - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """
    Chunk all documents into fixed-size overlapping windows.
    Returns flat list of Chunk objects.
    """
    all_chunks = []

    for doc in documents:
        pieces = _split_text(doc.content, chunk_size, chunk_overlap)

        for i, piece in enumerate(pieces):
            all_chunks.append(Chunk(
                content=piece,
                source=doc.source,
                chunk_index=i,
                metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(pieces)},
            ))

        logger.debug(f"{doc.metadata.get('filename', doc.source)}: {len(pieces)} chunks")

    logger.success(f"Total chunks created: {len(all_chunks)}")
    return all_chunks