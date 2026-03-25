"""
Vector store abstraction — wraps Milvus Lite for local vector search.
Decoupled from retrieval logic so it can be swapped for other DBs.
"""
import logging
from pathlib import Path

import numpy as np
from pymilvus import MilvusClient

from backend import config
from backend.chunker import Chunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Simple Milvus Lite wrapper for storing and searching chunk embeddings."""

    def __init__(
        self,
        db_path: str | None = None,
        collection: str | None = None,
        dimension: int = 3072,
    ):
        self.db_path = db_path or config.vectordb.db_path
        self.collection = collection or config.vectordb.collection
        self.dimension = dimension

        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.client = MilvusClient(self.db_path)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.has_collection(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                dimension=self.dimension,
                metric_type="COSINE",
            )
            logger.info(f"Created collection '{self.collection}' (dim={self.dimension})")

    def insert(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """
        Insert chunks with their embeddings into the vector store.

        Args:
            chunks: List of Chunk objects.
            embeddings: numpy array of shape (len(chunks), dim).

        Returns:
            Number of inserted records.
        """
        data = []
        for i, chunk in enumerate(chunks):
            data.append({
                "id": hash(chunk.uid) & 0x7FFFFFFFFFFFFFFF,  # positive int64
                "vector": embeddings[i].tolist(),
                "text": chunk.text,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "page_hint": chunk.page_hint,
            })

        self.client.insert(collection_name=self.collection, data=data)
        logger.info(f"Inserted {len(data)} chunks into '{self.collection}'")
        return len(data)

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> list[dict]:
        """
        Search for similar chunks.

        Args:
            query_vector: Query embedding (1D array).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: text, doc_id, chunk_id, page_hint, score.
        """
        results = self.client.search(
            collection_name=self.collection,
            data=[query_vector.tolist()],
            limit=top_k,
            output_fields=["text", "doc_id", "chunk_id", "page_hint"],
        )

        hits = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            hits.append({
                "text": entity.get("text", ""),
                "doc_id": entity.get("doc_id", ""),
                "chunk_id": entity.get("chunk_id", 0),
                "page_hint": entity.get("page_hint", ""),
                "score": hit.get("distance", 0.0),
            })
        return hits

    def count(self) -> int:
        """Get total number of chunks in the collection."""
        stats = self.client.get_collection_stats(self.collection)
        return stats.get("row_count", 0)

    def drop(self):
        """Drop the collection (for rebuilding index)."""
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)
            logger.info(f"Dropped collection '{self.collection}'")
        self._ensure_collection()
