"""
Centralized configuration — all settings loaded from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent


class FoxBrainConfig:
    base_url: str = os.getenv("FOXBRAIN_BASE_URL", "http://4.151.237.144:8000/v1")
    api_key: str = os.getenv("FOXBRAIN_API_KEY", "token-abc123")
    model: str = os.getenv("FOXBRAIN_MODEL", "20251203_remove_repeat")
    temperature: float = float(os.getenv("GENERATION_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("GENERATION_MAX_TOKENS", "1024"))


class OpenAIConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    judge_model: str = os.getenv("THELMA_JUDGE_MODEL", "gpt-4o-mini")


class RAGConfig:
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
    top_k_retrieve: int = int(os.getenv("RAG_TOP_K_RETRIEVE", "20"))
    top_k_rerank: int = int(os.getenv("RAG_TOP_K_RERANK", "8"))
    bm25_weight: float = float(os.getenv("RAG_BM25_WEIGHT", "0.4"))

    @property
    def vector_weight(self) -> float:
        return 1.0 - self.bm25_weight


class VectorDBConfig:
    db_path: str = os.getenv("VECTOR_DB_PATH", str(BASE_DIR / "data" / "milvus.db"))
    collection: str = os.getenv("VECTOR_DB_COLLECTION", "rag_collection")


class ThelmaConfig:
    max_concurrency: int = int(os.getenv("THELMA_MAX_CONCURRENCY", "3"))


class ServerConfig:
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))


# Singleton instances
foxbrain = FoxBrainConfig()
openai_cfg = OpenAIConfig()
rag = RAGConfig()
vectordb = VectorDBConfig()
thelma = ThelmaConfig()
server = ServerConfig()
