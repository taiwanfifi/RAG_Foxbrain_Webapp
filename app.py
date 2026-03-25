"""
FoxBrain RAG Demo — FastAPI Application
SSE streaming for pipeline stages, REST for data management.
"""
import json
import time
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config
from pdf_parser import parse_pdf, scan_folder, ParsedDocument
from chunker import chunk_text, Chunk
from embedder import embed_texts
from vector_store import VectorStore
from retriever import HybridRetriever, BM25Index
from generator import Generator
from thelma_engine import ThelmaEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# Global state (initialized on startup)
# ============================================================
vector_store: VectorStore | None = None
bm25_index: BM25Index | None = None
retriever: HybridRetriever | None = None
generator: Generator | None = None
thelma: ThelmaEngine | None = None

# Track uploaded documents
documents: dict[str, dict] = {}  # filename -> {parsed_doc metadata, chunks_count, status}
all_chunks: list[Chunk] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, bm25_index, retriever, generator, thelma
    logger.info("Starting FoxBrain RAG Demo...")

    vector_store = VectorStore()
    bm25_index = BM25Index()
    retriever = HybridRetriever(vector_store, bm25_index)
    generator = Generator()
    thelma = ThelmaEngine()

    # Auto-scan documents/ folder on startup
    _scan_existing_documents()

    logger.info("All engines initialized.")
    yield
    logger.info("Shutting down.")


def _scan_existing_documents():
    """Register any PDFs already in the documents/ folder on startup."""
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_files:
        logger.info("No existing PDFs in documents/ folder.")
        return

    for pdf_path in pdf_files:
        try:
            parsed = parse_pdf(pdf_path)
            documents[parsed.filename] = {
                "file_size_bytes": parsed.file_size_bytes,
                "page_count": parsed.page_count,
                "text_length": len(parsed.text),
                "chunks_count": 0,
                "status": "uploaded",  # needs Process to index
            }
        except Exception as e:
            logger.error(f"Failed to scan {pdf_path.name}: {e}")

    logger.info(f"Found {len(documents)} PDFs in documents/ folder (click Process to index).")


app = FastAPI(title="FoxBrain RAG Demo", version="0.1.0", lifespan=lifespan)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

DOCUMENTS_DIR = Path(__file__).parent / "documents"
DOCUMENTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Request / Response Models
# ============================================================

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    query: str
    enable_thelma: bool = True
    chat_history: list[ChatMessage] = []  # previous turns for multi-round


class FolderRequest(BaseModel):
    folder_path: str


class ConfigUpdate(BaseModel):
    foxbrain_base_url: str | None = None
    foxbrain_model: str | None = None
    foxbrain_api_key: str | None = None
    generation_temperature: float | None = None
    rag_chunk_size: int | None = None
    rag_top_k_rerank: int | None = None
    rag_bm25_weight: float | None = None


# ============================================================
# Frontend
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>FoxBrain RAG Demo</h1><p>Place index.html in static/</p>")


# ============================================================
# Data Management Endpoints
# ============================================================

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get current knowledge base status."""
    total_size = sum(d.get("file_size_bytes", 0) for d in documents.values())
    return {
        "documents": [
            {
                "filename": fname,
                "file_size_mb": f"{d.get('file_size_bytes', 0) / 1024 / 1024:.1f} MB",
                "page_count": d.get("page_count", 0),
                "chunks_count": d.get("chunks_count", 0),
                "status": d.get("status", "pending"),
                "summary": d.get("summary", ""),
            }
            for fname, d in documents.items()
        ],
        "total_documents": len(documents),
        "total_chunks": len(all_chunks),
        "total_size_mb": f"{total_size / 1024 / 1024:.1f} MB",
        "index_status": "ready" if all_chunks else "empty",
    }


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    save_path = DOCUMENTS_DIR / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        parsed = parse_pdf(save_path)
        documents[parsed.filename] = {
            "file_size_bytes": parsed.file_size_bytes,
            "page_count": parsed.page_count,
            "text_length": len(parsed.text),
            "chunks_count": 0,
            "status": "uploaded",
        }
        return {
            "filename": parsed.filename,
            "page_count": parsed.page_count,
            "file_size_mb": parsed.file_size_mb,
            "status": "uploaded",
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to parse PDF: {e}")


@app.post("/api/scan-folder")
async def scan_folder_endpoint(req: FolderRequest):
    """Scan a folder for PDFs and register them."""
    try:
        docs = scan_folder(req.folder_path)
        for parsed in docs:
            documents[parsed.filename] = {
                "file_size_bytes": parsed.file_size_bytes,
                "page_count": parsed.page_count,
                "text_length": len(parsed.text),
                "chunks_count": 0,
                "status": "uploaded",
            }
        return {"scanned": len(docs), "filenames": [d.filename for d in docs]}
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {e}")


@app.post("/api/process")
async def process_and_build_index():
    """Process all uploaded PDFs: chunk, embed, and build index."""
    global all_chunks

    if not documents:
        raise HTTPException(400, "No documents uploaded yet.")

    # Re-parse and chunk all documents
    new_chunks: list[Chunk] = []
    docs_to_summarize: list[tuple[str, str]] = []  # (filename, first_2000_chars)
    for fname, meta in documents.items():
        if meta["status"] == "indexed":
            continue
        pdf_path = DOCUMENTS_DIR / fname
        if not pdf_path.exists():
            continue
        parsed = parse_pdf(pdf_path)
        chunks = chunk_text(
            parsed.text,
            doc_id=fname,
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        new_chunks.extend(chunks)
        documents[fname]["chunks_count"] = len(chunks)
        documents[fname]["status"] = "indexed"
        docs_to_summarize.append((fname, parsed.text[:2000]))

    if not new_chunks:
        return {"message": "No new documents to process.", "total_chunks": len(all_chunks)}

    # Embed new chunks only
    texts = [c.text for c in new_chunks]
    embeddings = await embed_texts(texts)

    # Append to vector store (incremental, no drop)
    vector_store.insert(new_chunks, embeddings)
    all_chunks.extend(new_chunks)

    # Rebuild BM25 with all chunks (fast, in-memory)
    bm25_index.build(all_chunks)

    # Generate document summaries (async, parallel)
    await _generate_doc_summaries(docs_to_summarize)

    return {
        "processed": len(new_chunks),
        "total_chunks": len(all_chunks),
        "status": "ready",
    }


async def _generate_doc_summaries(docs: list[tuple[str, str]]):
    """Generate a 2-3 sentence summary for each document."""
    if not docs:
        return
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=config.openai_cfg.api_key)

    async def _summarize(fname: str, text_preview: str):
        try:
            resp = await client.chat.completions.create(
                model=config.openai_cfg.judge_model,
                messages=[{"role": "user", "content": f"用 2-3 句話摘要這份文件的主題和內容。只輸出摘要。\n\n文件名: {fname}\n內容開頭:\n{text_preview}"}],
                temperature=0, max_tokens=150,
            )
            summary = resp.choices[0].message.content.strip()
            documents[fname]["summary"] = summary
            logger.info(f"Summary for {fname}: {summary[:80]}...")
        except Exception as e:
            logger.warning(f"Failed to summarize {fname}: {e}")
            documents[fname]["summary"] = ""

    await asyncio.gather(*[_summarize(f, t) for f, t in docs])


def _build_kb_overview() -> str:
    """Build a knowledge base overview string for injection into generation prompt."""
    if not documents:
        return ""
    lines = ["目前知識庫中包含以下文件："]
    for i, (fname, meta) in enumerate(documents.items(), 1):
        summary = meta.get("summary", "")
        chunks = meta.get("chunks_count", 0)
        pages = meta.get("page_count", 0)
        line = f"{i}. {fname} ({pages}頁, {chunks}切塊)"
        if summary:
            line += f" — {summary}"
        lines.append(line)
    return "\n".join(lines)


@app.delete("/api/knowledge-base")
async def clear_knowledge_base():
    """Clear all documents and index."""
    global all_chunks
    documents.clear()
    all_chunks.clear()
    vector_store.drop()
    bm25_index.build([])
    return {"status": "cleared"}


@app.delete("/api/document/{filename}")
async def delete_document(filename: str):
    """Delete a single document and remove its chunks from the index."""
    global all_chunks
    if filename not in documents:
        raise HTTPException(404, f"Document not found: {filename}")

    # Remove from documents registry
    del documents[filename]

    # Remove its chunks from all_chunks
    all_chunks = [c for c in all_chunks if c.doc_id != filename]

    # Rebuild vector store without this document's chunks
    vector_store.drop()
    if all_chunks:
        embeddings = await embed_texts([c.text for c in all_chunks])
        vector_store.insert(all_chunks, embeddings)

    # Rebuild BM25
    bm25_index.build(all_chunks)

    # Delete PDF file
    pdf_path = DOCUMENTS_DIR / filename
    if pdf_path.exists():
        pdf_path.unlink()

    return {
        "deleted": filename,
        "remaining_documents": len(documents),
        "remaining_chunks": len(all_chunks),
    }


# ============================================================
# Main Pipeline: Ask (SSE streaming)
# ============================================================

@app.post("/api/ask")
async def ask(req: AskRequest):
    """
    Run the full RAG pipeline with SSE streaming.
    Each stage emits an event with its input/output.
    """
    if not all_chunks:
        raise HTTPException(400, "Knowledge base is empty. Upload and process documents first.")

    async def event_stream():
        total_t0 = time.time()

        def _emit(stage: str, data: dict):
            payload = json.dumps(data, ensure_ascii=False)
            return f"event: stage\ndata: {json.dumps({'stage': stage, **data}, ensure_ascii=False)}\n\n"

        # Build context-aware query (resolve references like "它", "第二個" etc.)
        effective_query = req.query
        if req.chat_history:
            # Use last 3 turns to resolve references
            recent = req.chat_history[-6:]  # last 3 Q+A pairs
            history_context = "\n".join([f"{m.role}: {m.content[:200]}" for m in recent])
            resolve_prompt = f"""根據以下對話歷史，將最新的問題改寫為一個獨立、完整的問題（不依賴上下文就能理解）。
只輸出改寫後的問題，不要解釋。

對話歷史：
{history_context}

最新問題：{req.query}"""
            try:
                from openai import AsyncOpenAI
                _client = AsyncOpenAI(api_key=config.openai_cfg.api_key)
                resolve_resp = await _client.chat.completions.create(
                    model=config.openai_cfg.judge_model,
                    messages=[{"role": "user", "content": resolve_prompt}],
                    temperature=0, max_tokens=200,
                )
                effective_query = resolve_resp.choices[0].message.content.strip()
            except Exception:
                effective_query = req.query

        # Stage 1: Query Understanding
        t0 = time.time()
        rewritten, sub_queries = await retriever.query_understanding(effective_query)
        stage1_ms = (time.time() - t0) * 1000
        yield _emit("query_understanding", {
            "input": {"query": req.query, "resolved_query": effective_query if effective_query != req.query else None},
            "output": {"rewritten": rewritten, "sub_queries": sub_queries},
            "time_ms": round(stage1_ms),
        })

        # Stage 2: Hybrid Retrieval
        t0 = time.time()
        candidates = await retriever.hybrid_search(rewritten, sub_queries)
        stage2_ms = (time.time() - t0) * 1000
        yield _emit("retrieval", {
            "input": {"rewritten": rewritten, "sub_queries": sub_queries},
            "output": {
                "bm25_hits": sum(1 for c in candidates if c.bm25_score > 0),
                "vector_hits": sum(1 for c in candidates if c.vector_score > 0),
                "total_candidates": len(candidates),
            },
            "time_ms": round(stage2_ms),
        })

        # Stage 3: Rerank
        t0 = time.time()
        reranked = await retriever.rerank(req.query, candidates)
        stage3_ms = (time.time() - t0) * 1000
        yield _emit("rerank", {
            "input": {"candidates": len(candidates)},
            "output": {
                "top_k": len(reranked),
                "top_scores": [{"score": round(r.rerank_score, 3), "src": f"{r.doc_id} {r.page_hint}".strip()} for r in reranked],
            },
            "time_ms": round(stage3_ms),
        })

        # Build context
        source_chunks = [
            {"doc_id": r.doc_id, "page_hint": r.page_hint, "text": r.text}
            for r in reranked
        ]
        context_parts = []
        for i, sc in enumerate(source_chunks):
            src = f"{sc['doc_id']} {sc['page_hint']}".strip()
            context_parts.append(f"[來源 {i+1}] ({src})\n{sc['text']}")
        context_str = "\n\n".join(context_parts)

        # Prepend KB overview so LLM knows what documents exist
        kb_overview = _build_kb_overview()
        if kb_overview:
            context_str = kb_overview + "\n\n---\n\n" + context_str

        # Stage 4: Generation
        t0 = time.time()
        history_dicts = [{"role": m.role, "content": m.content} for m in req.chat_history] if req.chat_history else None
        gen_output = await generator.generate(req.query, context_str, source_chunks, chat_history=history_dicts)
        stage4_ms = (time.time() - t0) * 1000
        yield _emit("generation", {
            "input": {"model": gen_output.model_used, "context_chunks": len(reranked)},
            "output": {
                "answer": gen_output.answer,
                "citations": [
                    {
                        "source_id": c.source_id,
                        "doc_id": c.doc_id,
                        "page_hint": c.page_hint,
                        "chunk_text": c.chunk_text,
                        "reason": c.reason,
                    }
                    for c in gen_output.citations
                ],
                "tokens_in": gen_output.tokens_in,
                "tokens_out": gen_output.tokens_out,
            },
            "time_ms": round(stage4_ms),
        })

        # Stage 5: THELMA Evaluation (optional)
        if req.enable_thelma:
            t0 = time.time()
            eval_output = await thelma.evaluate(req.query, gen_output.answer, context_str)
            stage5_ms = (time.time() - t0) * 1000
            yield _emit("thelma", {
                "input": {"query": req.query},
                "output": eval_output.to_dict(),
                "time_ms": round(stage5_ms),
            })

        total_ms = (time.time() - total_t0) * 1000
        yield f"event: done\ndata: {json.dumps({'total_time_ms': round(total_ms)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ============================================================
# Config Endpoints (runtime updates)
# ============================================================

@app.get("/api/config")
async def get_config():
    """Get current configuration (secrets masked)."""
    return {
        "foxbrain": {
            "base_url": config.foxbrain.base_url,
            "model": config.foxbrain.model,
            "temperature": config.foxbrain.temperature,
            "max_tokens": config.foxbrain.max_tokens,
        },
        "rag": {
            "chunk_size": config.rag.chunk_size,
            "chunk_overlap": config.rag.chunk_overlap,
            "top_k_retrieve": config.rag.top_k_retrieve,
            "top_k_rerank": config.rag.top_k_rerank,
            "bm25_weight": config.rag.bm25_weight,
        },
        "embedding_model": config.openai_cfg.embedding_model,
        "judge_model": config.openai_cfg.judge_model,
    }


@app.put("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration at runtime."""
    if update.foxbrain_base_url is not None:
        config.foxbrain.base_url = update.foxbrain_base_url
    if update.foxbrain_model is not None:
        config.foxbrain.model = update.foxbrain_model
    if update.foxbrain_api_key is not None:
        config.foxbrain.api_key = update.foxbrain_api_key
    if update.generation_temperature is not None:
        config.foxbrain.temperature = update.generation_temperature
    if update.rag_chunk_size is not None:
        config.rag.chunk_size = update.rag_chunk_size
    if update.rag_top_k_rerank is not None:
        config.rag.top_k_rerank = update.rag_top_k_rerank
    if update.rag_bm25_weight is not None:
        config.rag.bm25_weight = update.rag_bm25_weight
    return {"status": "updated"}


# ============================================================
# Health check
# ============================================================

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "foxbrain_url": config.foxbrain.base_url,
        "documents": len(documents),
        "chunks": len(all_chunks),
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
