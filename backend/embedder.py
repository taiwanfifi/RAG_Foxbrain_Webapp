"""
Embedding client — wraps OpenAI embedding API.
Decoupled so it can be swapped for local models later.
"""
import logging
import numpy as np
from openai import AsyncOpenAI

from backend import config

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=config.openai_cfg.api_key)
    return _client


async def embed_texts(texts: list[str], model: str | None = None) -> np.ndarray:
    """
    Embed a list of texts using OpenAI API.

    Args:
        texts: List of strings to embed.
        model: Override embedding model (defaults to config).

    Returns:
        numpy array of shape (len(texts), dim).
    """
    if not texts:
        return np.array([])

    model = model or config.openai_cfg.embedding_model
    client = _get_client()

    # OpenAI supports batch embedding
    resp = await client.embeddings.create(input=texts, model=model)
    vectors = [item.embedding for item in resp.data]

    logger.debug(f"Embedded {len(texts)} texts, dim={len(vectors[0])}")
    return np.array(vectors, dtype=np.float32)


async def embed_single(text: str, model: str | None = None) -> np.ndarray:
    """Embed a single text, returns 1D array."""
    result = await embed_texts([text], model=model)
    return result[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix for a set of vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = vectors / norms
    return normalized @ normalized.T
