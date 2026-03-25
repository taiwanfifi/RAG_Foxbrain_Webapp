"""
All prompt templates — centralized for easy review and modification.
"""

# ============================================================
# RAG Pipeline Prompts
# ============================================================

RAG_REWRITE = """將以下問題改寫為更清晰、適合搜尋引擎的查詢語句。只輸出改寫後的句子，不要加任何解釋。
問題: {question}"""

RAG_SUBQUERY = """將以下問題分解為 1-3 個關鍵子問題，以便從不同角度檢索。每行一個子問題，不要編號。
問題: {question}"""

RAG_GENERATION = """基於以下參考檔案回答問題。

規範：
1. 嚴格基於檔案內容作答，不要加入檔案中沒有提到的額外推論、建議或例子。
2. 若檔案中完全沒有相關資訊，請明確說明「根據提供的檔案，無法找到相關資訊」。
3. 回答需完整且條理分明，適度使用條列式說明步驟或重點。
4. 在引用特定來源時，使用 [來源 N] 標記，N 為來源編號。

參考檔案：
{context}

問題：{question}
回答："""

RAG_CITATION_REASON = """以下是一段問答中使用的來源片段。請用一句話說明為什麼這段內容被引用來回答該問題。

問題：{question}
來源片段：{chunk}
回答中引用此來源的部分：{usage}

請只輸出引用理由，一句話，不超過 30 字。"""

# ============================================================
# THELMA Evaluation Prompts
# ============================================================

EVAL_CLAIM_EXTRACTOR = """請從以下回答中提取所有獨立的事實性聲明（factual claims）。
每個聲明應是一個完整、可獨立驗證的陳述。每行一個聲明。

回答：{text}

請在 <output></output> 標籤中輸出：
<output>
聲明1
聲明2
...
</output>"""

EVAL_QUESTION_DECOMPOSER = """請將以下問題分解為 2-3 個核心資訊需求面向。
要求：
- 每個子問題代表「回答這個問題需要涵蓋的一個核心面向」
- 子問題應該是文件或回答「應該包含」的內容，而非使用者需要自己排查的事
- 每行一個，不要超過 3 個

問題：{text}

請在 <output></output> 標籤中輸出：
<output>
子問題1
子問題2
...
</output>"""

EVAL_RELEVANCE = """判斷以下事實對於回答該問題是否「必要」(essential) 或「多餘」(extraneous)。

問題：{query}
事實：{unit}

只輸出一個詞：essential 或 extraneous"""

EVAL_GROUNDEDNESS = """判斷以下聲明是否被來源文件所支持。

聲明：{claim}
來源文件：{source}

若聲明的內容可以從來源中找到依據，輸出 1；否則輸出 0。
只輸出一個數字：1 或 0"""

EVAL_COVERAGE = """判斷以下內容是否至少部分回答或涵蓋了該問題的資訊需求。
即使只提供了部分相關資訊，也算涵蓋。

問題：{question}
內容：{content}

只輸出 yes 或 no"""

EVAL_SUMMARY = """以下是一個 RAG 系統對某個問題的回答品質評估結果（THELMA 六維指標）。
請用 2-3 句中文總結這次回答的品質，指出優點和需要改進之處。語氣專業但易懂。

問題：{query}
指標：
- Source Precision (SP={sp}): 檢索到的來源中，相關來源的比例
- Source Query Coverage (SQC={sqc}): 來源對問題各面向的覆蓋率
- Response Precision (RP={rp}): 回答中與問題相關內容的比例
- Response Query Coverage (RQC={rqc}): 回答對問題各面向的覆蓋率
- Groundedness (GR={gr}): 回答忠實於來源的程度（低=幻覺）
- Self-Distinctness (SD={sd}): 回答各句的差異性（低=冗餘重複）
診斷碼：{diagnosis}

請輸出總結："""
