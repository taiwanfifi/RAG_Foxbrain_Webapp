"""
Hybrid Retriever — BM25 + Vector search with reranking.
Each stage is independently callable for testing/debugging.
"""
import re
import math
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from openai import AsyncOpenAI

import config
import prompts
import embedder
from chunker import Chunk
from vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Single retrieved chunk with scores."""
    text: str
    doc_id: str
    page_hint: str
    bm25_score: float = 0.0
    vector_score: float = 0.0
    ensemble_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RetrievalPipelineOutput:
    """Full output of the retrieval pipeline, with IO for each stage."""
    # Stage 1: Query understanding
    original_query: str = ""
    rewritten_query: str = ""
    sub_queries: list[str] = field(default_factory=list)
    stage1_time_ms: float = 0

    # Stage 2: Hybrid retrieval
    bm25_hits: int = 0
    vector_hits: int = 0
    candidates_after_dedup: int = 0
    stage2_time_ms: float = 0

    # Stage 3: Rerank
    reranked: list[RetrievalResult] = field(default_factory=list)
    stage3_time_ms: float = 0

    @property
    def top_k_text(self) -> str:
        """Formatted context string for generation."""
        parts = []
        for i, r in enumerate(self.reranked):
            src = f"{r.doc_id} {r.page_hint}".strip()
            parts.append(f"[來源 {i + 1}] ({src})\n{r.text}")
        return "\n\n".join(parts)


# ============================================================
# BM25 (in-memory, no external dependency)
# ============================================================

class BM25Index:
    """Simple BM25 implementation for Chinese + English text."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []       # original chunk data
        self.doc_freqs: list[Counter] = []
        self.idf: dict[str, float] = {}
        self.avg_dl: float = 0
        self.doc_count: int = 0

    def build(self, chunks: list[Chunk]):
        """Build BM25 index from chunks."""
        self.docs = [{"text": c.text, "doc_id": c.doc_id, "page_hint": c.page_hint} for c in chunks]
        self.doc_freqs = [Counter(_tokenize(c.text)) for c in chunks]
        self.doc_count = len(chunks)

        if self.doc_count == 0:
            return

        self.avg_dl = sum(sum(df.values()) for df in self.doc_freqs) / self.doc_count

        # Compute IDF
        term_doc_count: Counter = Counter()
        for df in self.doc_freqs:
            for term in df:
                term_doc_count[term] += 1

        for term, count in term_doc_count.items():
            self.idf[term] = math.log((self.doc_count - count + 0.5) / (count + 0.5) + 1)

        logger.info(f"BM25 index built: {self.doc_count} docs, {len(self.idf)} terms")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search BM25 index, returns list of {text, doc_id, page_hint, score}."""
        query_terms = _tokenize(query)
        scores = []

        for i, df in enumerate(self.doc_freqs):
            doc_len = sum(df.values())
            score = 0.0
            for term in query_terms:
                if term not in df:
                    continue
                tf = df[term]
                idf = self.idf.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
                score += idf * numerator / denominator
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score <= 0:
                break
            results.append({**self.docs[idx], "score": score})
        return results


def _tokenize(text: str) -> list[str]:
    """
    Tokenizer for BM25: English words kept whole, Chinese uses character bigrams.
    No NLP dependencies — pure regex + sliding window.

    "公文系統 EIP login" → ['公文', '文系', '系統', 'eip', 'login']
    """
    text = text.lower()
    # Split into runs of CJK characters vs English/digit words
    parts = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+", text)
    tokens = []
    for part in parts:
        if re.match(r"[\u4e00-\u9fff]", part):
            # Chinese: character bigrams (sliding window)
            if len(part) == 1:
                tokens.append(part)
            else:
                for i in range(len(part) - 1):
                    tokens.append(part[i:i+2])
        else:
            # English/numbers: keep as whole word
            tokens.append(part)
    return tokens


# ============================================================
# Retrieval Pipeline (each stage is a separate method)
# ============================================================

class HybridRetriever:
    """
    Full retrieval pipeline with independently testable stages:
      1. query_understanding() — rewrite + sub-query decomposition
      2. hybrid_search() — BM25 + vector search
      3. rerank() — embedding cosine similarity reranking
    """

    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self._llm_client: AsyncOpenAI | None = None

    def _get_llm(self) -> AsyncOpenAI:
        if self._llm_client is None:
            self._llm_client = AsyncOpenAI(api_key=config.openai_cfg.api_key)
        return self._llm_client

    # --- Stage 1: Query Understanding ---

    async def query_understanding(self, query: str) -> tuple[str, list[str]]:
        """
        Rewrite query + decompose into sub-queries.

        Returns:
            (rewritten_query, [sub_query_1, sub_query_2, ...])
        """
        client = self._get_llm()

        # Parallel: rewrite + decompose
        import asyncio
        rewrite_task = client.chat.completions.create(
            model=config.openai_cfg.judge_model,
            messages=[{"role": "user", "content": prompts.RAG_REWRITE.format(question=query)}],
            temperature=0,
            max_tokens=200,
        )
        subquery_task = client.chat.completions.create(
            model=config.openai_cfg.judge_model,
            messages=[{"role": "user", "content": prompts.RAG_SUBQUERY.format(question=query)}],
            temperature=0,
            max_tokens=300,
        )

        rewrite_resp, subquery_resp = await asyncio.gather(rewrite_task, subquery_task)

        rewritten = rewrite_resp.choices[0].message.content.strip()
        sub_queries = [
            line.strip()
            for line in subquery_resp.choices[0].message.content.strip().split("\n")
            if line.strip()
        ]

        return rewritten, sub_queries

    # --- Stage 2: Hybrid Search ---

    async def hybrid_search(
        self,
        rewritten_query: str,
        sub_queries: list[str],
        top_k_retrieve: int | None = None,
        bm25_weight: float | None = None,
    ) -> list[RetrievalResult]:
        """
        BM25 + Vector hybrid search with deduplication.

        Returns:
            List of RetrievalResult with ensemble scores.
        """
        top_k = top_k_retrieve or config.rag.top_k_retrieve
        w_bm25 = bm25_weight or config.rag.bm25_weight
        w_vec = 1.0 - w_bm25

        all_queries = [rewritten_query] + sub_queries

        # BM25 search (all queries)
        bm25_results: dict[str, RetrievalResult] = {}
        for q in all_queries:
            for hit in self.bm25.search(q, top_k=top_k):
                key = f"{hit['doc_id']}::{hit['text'][:50]}"
                if key not in bm25_results:
                    bm25_results[key] = RetrievalResult(
                        text=hit["text"],
                        doc_id=hit["doc_id"],
                        page_hint=hit.get("page_hint", ""),
                        bm25_score=hit["score"],
                    )
                else:
                    bm25_results[key].bm25_score = max(bm25_results[key].bm25_score, hit["score"])

        # Vector search (all queries)
        vector_results: dict[str, RetrievalResult] = {}
        for q in all_queries:
            q_vec = await embedder.embed_single(q)
            for hit in self.vector_store.search(q_vec, top_k=top_k):
                key = f"{hit['doc_id']}::{hit['text'][:50]}"
                if key not in vector_results:
                    vector_results[key] = RetrievalResult(
                        text=hit["text"],
                        doc_id=hit["doc_id"],
                        page_hint=hit.get("page_hint", ""),
                        vector_score=hit["score"],
                    )
                else:
                    vector_results[key].vector_score = max(vector_results[key].vector_score, hit["score"])

        # Merge and compute ensemble score
        all_results: dict[str, RetrievalResult] = {}
        for key, r in bm25_results.items():
            all_results[key] = r
        for key, r in vector_results.items():
            if key in all_results:
                all_results[key].vector_score = r.vector_score
            else:
                all_results[key] = r

        # Normalize scores to [0, 1] and compute ensemble
        bm25_max = max((r.bm25_score for r in all_results.values()), default=1) or 1
        vec_max = max((r.vector_score for r in all_results.values()), default=1) or 1
        for r in all_results.values():
            norm_bm25 = r.bm25_score / bm25_max
            norm_vec = r.vector_score / vec_max
            r.ensemble_score = w_bm25 * norm_bm25 + w_vec * norm_vec

        # Sort by ensemble score
        merged = sorted(all_results.values(), key=lambda x: x.ensemble_score, reverse=True)
        return merged

    # --- Stage 3: Rerank ---

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k_rerank: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Rerank candidates using embedding cosine similarity.

        Returns:
            Top-K reranked results.
        """
        top_k = top_k_rerank or config.rag.top_k_rerank
        if not candidates:
            return []

        # Embed query + all candidate texts
        texts = [query] + [c.text for c in candidates]
        all_embeddings = await embedder.embed_texts(texts)
        query_emb = all_embeddings[0]
        candidate_embs = all_embeddings[1:]

        # Compute cosine similarity
        for i, candidate in enumerate(candidates):
            candidate.rerank_score = embedder.cosine_similarity(query_emb, candidate_embs[i])

        # Sort by rerank score
        candidates.sort(key=lambda x: x.rerank_score, reverse=True)

        # Diversity selection: round-robin across documents by score
        # Ensures each document gets fair representation in top-K
        from collections import defaultdict
        by_doc: dict[str, list[RetrievalResult]] = defaultdict(list)
        for c in candidates:
            by_doc[c.doc_id].append(c)

        # If only 1 document, just return top-K
        if len(by_doc) <= 1:
            return candidates[:top_k]

        # Round-robin: take best from each doc, repeat
        selected: list[RetrievalResult] = []
        doc_iters = {doc: iter(chunks) for doc, chunks in by_doc.items()}
        while len(selected) < top_k and doc_iters:
            exhausted = []
            for doc, it in doc_iters.items():
                if len(selected) >= top_k:
                    break
                item = next(it, None)
                if item is None:
                    exhausted.append(doc)
                else:
                    selected.append(item)
            for doc in exhausted:
                del doc_iters[doc]

        return selected

    # --- Full Pipeline ---

    async def retrieve(self, query: str) -> RetrievalPipelineOutput:
        """
        Run the full retrieval pipeline. Returns detailed output for each stage.
        """
        import time
        output = RetrievalPipelineOutput(original_query=query)

        # Stage 1
        t0 = time.time()
        output.rewritten_query, output.sub_queries = await self.query_understanding(query)
        output.stage1_time_ms = (time.time() - t0) * 1000

        # Stage 2
        t0 = time.time()
        candidates = await self.hybrid_search(output.rewritten_query, output.sub_queries)
        output.bm25_hits = sum(1 for c in candidates if c.bm25_score > 0)
        output.vector_hits = sum(1 for c in candidates if c.vector_score > 0)
        output.candidates_after_dedup = len(candidates)
        output.stage2_time_ms = (time.time() - t0) * 1000

        # Stage 3
        t0 = time.time()
        output.reranked = await self.rerank(query, candidates)
        output.stage3_time_ms = (time.time() - t0) * 1000

        return output
