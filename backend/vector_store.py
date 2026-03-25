"""
Vector store — pure numpy in-memory cosine search with file persistence.
No external DB dependency. Works everywhere (local, Docker, Render, Railway).
"""
import json
import logging
from pathlib import Path

import numpy as np

from backend.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """In-memory vector store using numpy brute-force cosine search."""

    def __init__(self, dimension: int = 3072, **kwargs):
        self.dimension = dimension
        self._vectors: np.ndarray | None = None  # (N, dim)
        self._metadata: list[dict] = []  # parallel list of chunk metadata
        logger.info(f"VectorStore initialized (in-memory, dim={dimension})")

    def insert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """
        Insert chunks with their embeddings.

        Args:
            chunks: List of Chunk objects.
            embeddings: numpy array of shape (len(chunks), dim).

        Returns:
            Number of inserted records.
        """
        meta = [
            {
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "page_hint": c.page_hint,
            }
            for c in chunks
        ]

        if self._vectors is None:
            self._vectors = embeddings.copy()
            self._metadata = meta
        else:
            self._vectors = np.vstack([self._vectors, embeddings])
            self._metadata.extend(meta)

        logger.info(f"Inserted {len(chunks)} chunks (total: {len(self._metadata)})")
        return len(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        """
        Cosine similarity search.

        Args:
            query_vector: Query embedding (1D array).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, doc_id, chunk_id, page_hint, score.
        """
        if self._vectors is None or len(self._metadata) == 0:
            return []

        # Normalize for cosine similarity
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
        normalized = self._vectors / norms

        # Compute cosine similarity
        scores = normalized @ query_norm
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append({
                **self._metadata[idx],
                "score": float(scores[idx]),
            })
        return results

    def count(self) -> int:
        """Get total number of stored chunks."""
        return len(self._metadata)

    def drop(self):
        """Clear all data."""
        self._vectors = None
        self._metadata = []
        logger.info("VectorStore cleared")

    def save(self, path: str | Path):
        """Serialize index to disk (vectors.npy + metadata.json)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors is not None and len(self._metadata) > 0:
            np.save(str(path / "vectors.npy"), self._vectors)
            with open(path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False)
            logger.info(f"VectorStore saved: {len(self._metadata)} chunks → {path}")
        else:
            logger.warning("VectorStore is empty, nothing to save")

    def load(self, path: str | Path) -> bool:
        """Load index from disk. Returns True if loaded successfully."""
        path = Path(path)
        vec_path = path / "vectors.npy"
        meta_path = path / "metadata.json"
        if not vec_path.exists() or not meta_path.exists():
            return False
        self._vectors = np.load(str(vec_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)
        logger.info(f"VectorStore loaded: {len(self._metadata)} chunks from {path}")
        return True
