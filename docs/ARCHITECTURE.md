# FoxBrain RAG Demo — Architecture Plan

## 定位

一個**可快速部署**的 Web Demo，讓所長 / 對外展示時能直觀看到：
1. 上傳 PDF → 自動建立知識庫
2. 提問 → RAG pipeline 逐步運作（每一步都有 I/O sample）
3. FoxBrain 生成答案 + **來源引用 & 引用理由**
4. THELMA 自動評估 → 6 維雷達圖 + 診斷碼

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | **FastAPI** (Python) | async 原生、跟現有 thelma3.1 同語言、部署簡單 |
| Frontend | **單一 HTML + Vanilla JS/CSS** | 零 build step、一個檔案就能 demo |
| PDF 解析 | **PyMuPDF (fitz)** | 速度快、支援中文、layout 保留好 |
| 向量 DB | **Milvus Lite** (SQLite-backed) | 單檔、不用另起服務 |
| LLM | **FoxBrain API** (OpenAI-compatible) | 公司模型，demo 主角 |
| Judge/Embed | **OpenAI gpt-4o-mini + text-embedding-3-large** | THELMA 評估用 |
| 部署 | **Docker 一鍵** 或 `pip install + python app.py` | 對方拿到就能跑 |

## Pipeline 設計（6 階段，前端每段都顯示 I/O）

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 0: Knowledge Base                                         │
│ Input:  PDF file (upload)                                       │
│ Process: PyMuPDF extract → RecursiveCharacterTextSplitter       │
│          → Embed (text-embedding-3-large) → Milvus + BM25      │
│ Output: {chunks_count, sample_chunk, index_status}              │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: Query Understanding                                    │
│ Input:  "無法登入公文系統，怎麼辦？"                               │
│ Process: LLM Rewrite + Sub-query Decomposition                  │
│ Output: {rewritten: "公文系統登入失敗解決方法",                    │
│          sub_queries: ["登入失敗原因?", "重設密碼步驟?"]}          │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2: Hybrid Retrieval                                       │
│ Input:  rewritten query + sub_queries                           │
│ Process: BM25 (k=20, 40%) + Vector (k=20, 60%) → Ensemble      │
│ Output: {candidates: [{text, score, source}...], count: 28}     │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: Rerank                                                 │
│ Input:  28 candidate chunks                                     │
│ Process: Embedding cosine similarity reranking                  │
│ Output: {top_8: [{text, rerank_score, page_num}...]}            │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: Generation (FoxBrain)                                  │
│ Input:  query + top_8 context                                   │
│ Process: FoxBrain API (20251203_remove_repeat)                  │
│ Output: {answer: "...", citations: [{chunk_id, reason}...]}     │
│         ← 每段引用標注 [來源 1] 並附引用理由                      │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ Stage 5: THELMA Evaluation                                      │
│ Input:  query + answer + retrieved_context                      │
│ Process: Decompose → Match → Score                              │
│ Output: {SP, SQC, RP, RQC, GR, SD, diagnosis, details}         │
└─────────────────────────────────────────────────────────────────┘
```

## 前端 Layout 規劃

```
┌──────────────┬───────────────────────────────┬──────────────────┐
│              │                               │                  │
│  知識庫管理   │     主對話區                    │  Pipeline 視圖   │
│              │                               │                  │
│ ☐ 上傳 PDF   │  User: 無法登入怎麼辦？         │  ▶ Query理解     │
│ ☐ 已建立索引  │                               │    └ I/O sample  │
│   - doc1.pdf │  Bot: 根據文件，登入失敗        │  ▶ 混合檢索      │
│   - doc2.pdf │       可能原因如下...           │    └ I/O sample  │
│              │       [來源1] 因為...           │  ▶ 重排序        │
│ ☐ 索引狀態   │       [來源2] 因為...           │    └ I/O sample  │
│   chunks: 42 │                               │  ▶ FoxBrain 生成 │
│   status: ✓  │  ┌─────────────────────────┐   │    └ I/O sample  │
│              │  │ THELMA 評估結果          │   │  ▶ THELMA 評估  │
│              │  │ 🔵 SP: 0.88  RP: 0.75  │   │    └ 6 metrics   │
│              │  │ 🔵 SQC:0.67  RQC:0.67  │   │                  │
│              │  │ 🔵 GR: 0.80  SD: 0.85  │   │                  │
│              │  │ 診斷: 正常              │   │                  │
│              │  └─────────────────────────┘   │                  │
│              │                               │                  │
│              │  [________________] [送出]     │                  │
└──────────────┴───────────────────────────────┴──────────────────┘
```

## 後端 API Endpoints

```
POST /api/upload          ← 上傳 PDF，回傳 {chunks, sample}
GET  /api/knowledge-base  ← 知識庫狀態
POST /api/ask             ← 主流程：RAG + 生成 + 評估
                            回傳 SSE stream，逐步推送每個 stage 的結果
DELETE /api/knowledge-base ← 清除知識庫
```

`/api/ask` 用 **Server-Sent Events (SSE)** 逐步推送：
```
event: stage
data: {"stage": "query_understanding", "input": "...", "output": {...}, "time_ms": 320}

event: stage
data: {"stage": "retrieval", "input": {...}, "output": {...}, "time_ms": 1200}

event: stage
data: {"stage": "rerank", ...}

event: stage
data: {"stage": "generation", "output": {"answer": "...", "citations": [...]}}

event: stage
data: {"stage": "evaluation", "output": {"SP": 0.88, ...}}

event: done
data: {"total_time_ms": 8500}
```

## 引用設計

答案中的引用格式：
```
根據文件，登入失敗可能有以下原因：

1. EIP 帳號輸入錯誤 [來源 1]
2. 密碼已過期需重設 [來源 2]
3. 系統維護中暫時無法使用 [來源 1]

---
[來源 1] doc1.pdf p.3 — "EIP帳號登入說明..."
  引用理由: 直接回答了登入失敗的原因與帳號相關問題

[來源 2] doc1.pdf p.7 — "密碼重設流程..."
  引用理由: 提供了密碼過期的處理步驟
```

前端點擊 `[來源 1]` 可展開/高亮原文段落。

## 部署方式

### Option A: 最簡（對方電腦直接跑）
```bash
pip install -r requirements.txt
python app.py
# → http://localhost:8000
```

### Option B: Docker（推薦，環境隔離）
```bash
docker-compose up
# → http://localhost:8000
```

### 檔案結構
```
webapp/
├── app.py                 # FastAPI 主程式
├── rag_engine.py          # RAG pipeline（從 thelma3.1 重構）
├── thelma_engine.py       # THELMA 評估（從 thelma3.1 重構）
├── pdf_parser.py          # PDF → text
├── prompts.py             # 所有 prompt templates
├── config.py              # 設定集中管理
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── static/
│   └── index.html         # 前端（單檔）
├── ARCHITECTURE.md         # 本文件
└── FoxBrain_v1.5_UserGuide.md
```

## 開發順序

1. **Phase 0**: Mock prototype（HTML 假資料）→ UI/UX 討論 ← 現在
2. **Phase 1**: PDF 上傳 + 解析 + 知識庫建立
3. **Phase 2**: RAG pipeline 接通 FoxBrain
4. **Phase 3**: 引用標注 + 來源展示
5. **Phase 4**: THELMA 評估整合
6. **Phase 5**: Docker 打包 + 部署文件
