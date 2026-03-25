# FoxBrain RAG Demo

Enterprise RAG system with FoxBrain LLM, hybrid retrieval, source citations, and THELMA 6-dimension quality evaluation.

## Quick Start

### 1. Setup

```bash
cd webapp
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run (Local)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8080
```

### 3. Run (Docker)

```bash
docker-compose up --build
# Open http://localhost:8080
```

## Configuration (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `FOXBRAIN_BASE_URL` | FoxBrain API endpoint | Yes |
| `FOXBRAIN_API_KEY` | FoxBrain API key | Yes |
| `FOXBRAIN_MODEL` | FoxBrain model ID | Yes |
| `OPENAI_API_KEY` | OpenAI key (for embedding + THELMA judge) | Yes |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | No (default: text-embedding-3-large) |
| `THELMA_JUDGE_MODEL` | THELMA judge model | No (default: gpt-4o-mini) |
| `RAG_CHUNK_SIZE` | Characters per chunk | No (default: 500) |
| `RAG_BM25_WEIGHT` | BM25 vs Vector weight | No (default: 0.4) |

See `.env.example` for all options.

## Architecture

```
PDF Upload → PyMuPDF Extract → Chunk (500 chars, 100 overlap)
                                  ↓
                    Embed (text-embedding-3-large) → Milvus Lite + BM25

User Query → Rewrite → Sub-query Decompose → Hybrid Search (BM25 + Vector)
                                                    ↓
                                              Rerank (cosine) → Top 8
                                                    ↓
                                         FoxBrain Generation + Citations
                                                    ↓
                                         THELMA 6-Metric Evaluation
```

## Project Structure

```
webapp/
├── app.py              # FastAPI server (SSE streaming)
├── config.py           # .env config loader
├── pdf_parser.py       # PDF → text (PyMuPDF)
├── chunker.py          # Text → chunks (recursive splitter)
├── embedder.py         # OpenAI embedding client
├── vector_store.py     # Milvus Lite wrapper
├── retriever.py        # BM25 + Vector hybrid + rerank
├── generator.py        # FoxBrain generation + citation extraction
├── thelma_engine.py    # THELMA 6-metric async evaluation
├── prompts.py          # All prompt templates
├── static/index.html   # Frontend (single file, zh/en toggle)
├── .env.example        # Config template
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build
└── docker-compose.yml  # One-command deploy
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend |
| GET | `/api/knowledge-base` | Document list & stats |
| POST | `/api/upload` | Upload PDF |
| POST | `/api/scan-folder` | Scan folder for PDFs |
| POST | `/api/process` | Chunk + embed + build index |
| POST | `/api/ask` | Full pipeline (SSE stream) |
| GET | `/api/config` | Current config |
| PUT | `/api/config` | Update config at runtime |
| GET | `/api/health` | Health check |

## THELMA Metrics

| Metric | Name | Measures |
|--------|------|----------|
| SP | Source Precision | Are retrieved sources relevant? |
| SQC | Source Query Coverage | Do sources cover all sub-questions? |
| RP | Response Precision | Is the response free of irrelevant content? |
| RQC | Response Query Coverage | Does the response cover all sub-questions? |
| GR | Groundedness | Is the response faithful to sources? (Low = hallucination) |
| SD | Self-Distinctness | Is the response free of redundancy? |

## How to Use — Query Guide

Upload any PDF (papers, manuals, reports, regulations...) and start asking. The system supports three types of queries:

### 1. Overview / Meta Questions

Ask about what's in the knowledge base — the system auto-generates document summaries on upload.

```
現在知識庫裡有哪些文件？每篇大概在講什麼？
哪篇論文跟 RAG 評估有關？
有沒有跟公文系統操作相關的文件？
這幾份文件之間有什麼關聯？
```

### 2. Deep-dive / Specific Questions

Ask detailed questions about content within any uploaded document.

```
THELMA 的六個評估指標分別是什麼？各自衡量什麼？
AI Scientist v2 的 open-ended research loop 跟 v1 差在哪？
無法登入公文系統，該怎麼辦？
論文中的 Groundedness 指標具體怎麼計算的？
```

### 3. Multi-turn Follow-up

Use pronouns and references — the system resolves context from conversation history.

```
Turn 1: THELMA 有哪些 metrics?
Turn 2: 它的第三個是什麼意思？可以詳細解釋嗎？
Turn 3: 那這個指標如果分數很低代表什麼問題？
```

### 4. Cross-document Comparison

Ask questions that span multiple uploaded documents.

```
THELMA 的評估方法跟 AI Scientist 的 review 機制有什麼差異？
公文系統的操作流程跟論文中提到的 RAG pipeline 有什麼相似之處？
```

### Tips

- Upload first, then click **Process & Build Index** — the system needs to chunk and embed before querying
- Toggle **THELMA** checkbox to enable/disable quality evaluation per query (saves time when exploring)
- The pipeline flow visualization (top panel) shows exactly what happened at each stage — click nodes to inspect I/O
- Each answer includes **source citations** with page numbers and reasons — click `[來源 N]` to highlight
- Use the **ZH/EN** toggle for interface language switching

## Features

- **Hybrid Retrieval**: BM25 (keyword) + Vector (semantic) with configurable weights and cross-document diversity
- **Source Citations**: `[來源 N]` with extracted citation reasons
- **THELMA Evaluation**: 6-dimension quality scoring + AI-generated summary
- **Document Summaries**: Auto-generated on upload for meta-question support
- **Multi-turn Conversation**: Context-aware query resolution across turns
- **Pipeline Inspector**: Real-time visualization of each pipeline stage with I/O
- **i18n**: Chinese/English toggle (no reload)
- **Incremental Index**: Add/remove documents without rebuilding everything
- **Easy Deploy**: Single `docker-compose up` or `python app.py`
