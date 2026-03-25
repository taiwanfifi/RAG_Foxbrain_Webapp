"""
THELMA Evaluation Engine — Reference-free RAG QA evaluation.
6 metrics: SP, SQC, RP, RQC, GR, SD + AI summary.
Fully async with concurrency control.
"""
import re
import time
import logging
import asyncio
from dataclasses import dataclass, field

import numpy as np
from openai import AsyncOpenAI

import config
import prompts
import embedder

logger = logging.getLogger(__name__)


@dataclass
class MetricDetail:
    """Detail for a single metric."""
    score: float
    numerator: int = 0
    denominator: int = 0
    description_zh: str = ""
    description_en: str = ""


@dataclass
class ThelmaOutput:
    """Full THELMA evaluation result."""
    sp: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    sqc: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    rp: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    rqc: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    gr: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    sd: MetricDetail = field(default_factory=lambda: MetricDetail(0))
    diagnosis: str = ""
    diagnosis_zh: str = ""
    ai_summary: str = ""
    total_llm_calls: int = 0
    time_ms: float = 0

    def to_dict(self) -> dict:
        return {
            "sp": {"score": self.sp.score, "detail": f"{self.sp.numerator}/{self.sp.denominator}",
                   "desc_zh": self.sp.description_zh, "desc_en": self.sp.description_en},
            "sqc": {"score": self.sqc.score, "detail": f"{self.sqc.numerator}/{self.sqc.denominator}",
                    "desc_zh": self.sqc.description_zh, "desc_en": self.sqc.description_en},
            "rp": {"score": self.rp.score, "detail": f"{self.rp.numerator}/{self.rp.denominator}",
                   "desc_zh": self.rp.description_zh, "desc_en": self.rp.description_en},
            "rqc": {"score": self.rqc.score, "detail": f"{self.rqc.numerator}/{self.rqc.denominator}",
                    "desc_zh": self.rqc.description_zh, "desc_en": self.rqc.description_en},
            "gr": {"score": self.gr.score, "detail": f"{self.gr.numerator}/{self.gr.denominator}",
                   "desc_zh": self.gr.description_zh, "desc_en": self.gr.description_en},
            "sd": {"score": self.sd.score, "detail": f"avg_sim={1-self.sd.score:.2f}",
                   "desc_zh": self.sd.description_zh, "desc_en": self.sd.description_en},
            "diagnosis": self.diagnosis,
            "diagnosis_zh": self.diagnosis_zh,
            "ai_summary": self.ai_summary,
            "total_llm_calls": self.total_llm_calls,
            "time_ms": self.time_ms,
        }


class ThelmaEngine:
    """
    Async THELMA evaluator with semaphore-based concurrency control.
    Each metric computation is independently callable for testing.
    """

    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._sem = asyncio.Semaphore(config.thelma.max_concurrency)
        self._llm_call_count = 0

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=config.openai_cfg.api_key)
        return self._client

    # --- LLM call with rate limiting and retry ---

    async def _call_judge(self, prompt: str, max_tokens: int = 50) -> str:
        """Call the judge LLM with semaphore control and retry."""
        async with self._sem:
            for attempt in range(3):
                try:
                    client = self._get_client()
                    resp = await client.chat.completions.create(
                        model=config.openai_cfg.judge_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    self._llm_call_count += 1
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(1)
                    else:
                        logger.error(f"Judge call failed after 3 attempts: {e}")
                        return ""

    # --- Decomposition ---

    async def decompose_claims(self, response: str) -> list[str]:
        """Extract factual claims from a response."""
        result = await self._call_judge(
            prompts.EVAL_CLAIM_EXTRACTOR.format(text=response),
            max_tokens=500,
        )
        return self._parse_output_tags(result)

    async def decompose_questions(self, query: str) -> list[str]:
        """Split query into sub-questions."""
        result = await self._call_judge(
            prompts.EVAL_QUESTION_DECOMPOSER.format(text=query),
            max_tokens=300,
        )
        return self._parse_output_tags(result)

    def decompose_sentences(self, text: str) -> list[str]:
        """Split text into sentences (regex, no LLM)."""
        sentences = re.split(r"[.!?。！？\n]+", text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def decompose_source(self, source: str) -> list[str]:
        """Split source into paragraphs."""
        paragraphs = re.split(r"\n\n+", source)
        return [p.strip() for p in paragraphs if len(p.strip()) > 10]

    # --- Matching (binary judgments) ---

    async def judge_relevance(self, query: str, unit: str) -> float:
        """Is this unit essential to answering the query? Returns 1.0 or 0.0."""
        result = await self._call_judge(
            prompts.EVAL_RELEVANCE.format(query=query, unit=unit)
        )
        return 1.0 if "essential" in result.lower() else 0.0

    async def judge_groundedness(self, claim: str, source: str) -> float:
        """Is this claim supported by the source? Returns 1.0 or 0.0."""
        result = await self._call_judge(
            prompts.EVAL_GROUNDEDNESS.format(claim=claim, source=source)
        )
        return 1.0 if "1" in result or "supported" in result.lower() else 0.0

    async def judge_coverage(self, question: str, content: str) -> float:
        """Does this content answer the question? Returns 1.0 or 0.0."""
        result = await self._call_judge(
            prompts.EVAL_COVERAGE.format(question=question, content=content)
        )
        return 1.0 if "yes" in result.lower() else 0.0

    # --- Self-Distinctness (embedding-based, no LLM) ---

    async def calc_self_distinctness(self, sentences: list[str]) -> float:
        """Calculate 1 - avg pairwise cosine similarity."""
        if len(sentences) < 2:
            return 1.0

        embeddings = await embedder.embed_texts(sentences)
        sim_matrix = embedder.cosine_similarity_matrix(embeddings)

        # Upper triangle (excluding diagonal)
        n = len(sentences)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += sim_matrix[i][j]
                count += 1

        avg_sim = total / count if count > 0 else 0.0
        return max(0.0, 1.0 - avg_sim)

    # --- Diagnosis ---

    def diagnose(self, sp, sqc, rp, rqc, gr, sd) -> tuple[str, str]:
        """Generate diagnosis code from metrics."""
        if sd < 0.5 and rp > 0.7:
            return "redundancy", "冗長重複"
        if sqc < 0.5:
            return "retrieval_gap", "檢索缺失"
        if gr < 0.5:
            return "hallucination", "幻覺"
        return "normal", "正常"

    # --- AI Summary ---

    async def generate_summary(
        self, query: str, sp, sqc, rp, rqc, gr, sd, diagnosis_zh: str
    ) -> str:
        """Generate a natural language summary of the evaluation."""
        result = await self._call_judge(
            prompts.EVAL_SUMMARY.format(
                query=query, sp=sp, sqc=sqc, rp=rp, rqc=rqc, gr=gr, sd=sd,
                diagnosis=diagnosis_zh,
            ),
            max_tokens=200,
        )
        return result

    # --- Full Evaluation ---

    async def evaluate(self, query: str, response: str, source: str) -> ThelmaOutput:
        """
        Run full THELMA evaluation. Returns all 6 metrics + diagnosis + summary.
        """
        t0 = time.time()
        self._llm_call_count = 0

        # Step 1: Decompose (parallel — LLM calls async, regex sync via executor)
        loop = asyncio.get_running_loop()
        sub_questions, claims, sentences, source_paragraphs = await asyncio.gather(
            self.decompose_questions(query),
            self.decompose_claims(response),
            loop.run_in_executor(None, self.decompose_sentences, response),
            loop.run_in_executor(None, self.decompose_source, source),
        )

        # Ensure non-empty
        if not sub_questions:
            sub_questions = [query]
        if not claims:
            claims = self.decompose_sentences(response)
        if not source_paragraphs:
            source_paragraphs = [source]

        # Step 2: Compute metrics (parallel)
        # For coverage checks, truncate source/response to avoid overwhelming the judge
        source_truncated = source[:3000] if len(source) > 3000 else source
        response_truncated = response[:2000] if len(response) > 2000 else response

        # SP: source paragraph relevance to query
        sp_tasks = [self.judge_relevance(query, p) for p in source_paragraphs]
        # SQC: sub-questions covered by source (use truncated to help judge focus)
        sqc_tasks = [self.judge_coverage(q, source_truncated) for q in sub_questions]
        # RP: claims relevance to query
        rp_tasks = [self.judge_relevance(query, c) for c in claims]
        # RQC: sub-questions answered by response
        rqc_tasks = [self.judge_coverage(q, response_truncated) for q in sub_questions]
        # GR: claims grounded in source
        gr_tasks = [self.judge_groundedness(c, source_truncated) for c in claims]

        all_results = await asyncio.gather(
            asyncio.gather(*sp_tasks),
            asyncio.gather(*sqc_tasks),
            asyncio.gather(*rp_tasks),
            asyncio.gather(*rqc_tasks),
            asyncio.gather(*gr_tasks),
            self.calc_self_distinctness(sentences),
        )

        sp_scores, sqc_scores, rp_scores, rqc_scores, gr_scores, sd_val = all_results

        # Compute averages
        def _avg(scores):
            return sum(scores) / len(scores) if scores else 0.0

        sp_val = _avg(sp_scores)
        sqc_val = _avg(sqc_scores)
        rp_val = _avg(rp_scores)
        rqc_val = _avg(rqc_scores)
        gr_val = _avg(gr_scores)

        # Step 3: Diagnosis
        diag_en, diag_zh = self.diagnose(sp_val, sqc_val, rp_val, rqc_val, gr_val, sd_val)

        # Step 4: AI Summary (async, can fail gracefully)
        try:
            ai_summary = await self.generate_summary(
                query, sp_val, sqc_val, rp_val, rqc_val, gr_val, sd_val, diag_zh
            )
        except Exception:
            ai_summary = ""

        elapsed = (time.time() - t0) * 1000

        return ThelmaOutput(
            sp=MetricDetail(
                score=round(sp_val, 2),
                numerator=int(sum(sp_scores)), denominator=len(source_paragraphs),
                description_zh="檢索到的來源中，與問題相關的比例",
                description_en="Ratio of retrieved sources relevant to the query",
            ),
            sqc=MetricDetail(
                score=round(sqc_val, 2),
                numerator=int(sum(sqc_scores)), denominator=len(sub_questions),
                description_zh="來源對問題各面向的覆蓋率",
                description_en="How well sources cover all aspects of the query",
            ),
            rp=MetricDetail(
                score=round(rp_val, 2),
                numerator=int(sum(rp_scores)), denominator=len(claims),
                description_zh="回答中與問題相關的內容比例",
                description_en="Ratio of response content relevant to the query",
            ),
            rqc=MetricDetail(
                score=round(rqc_val, 2),
                numerator=int(sum(rqc_scores)), denominator=len(sub_questions),
                description_zh="回答對問題各面向的覆蓋率",
                description_en="How well the response covers all aspects of the query",
            ),
            gr=MetricDetail(
                score=round(gr_val, 2),
                numerator=int(sum(gr_scores)), denominator=len(claims),
                description_zh="回答忠實於來源的程度（低=幻覺）",
                description_en="How faithfully the response follows sources (low = hallucination)",
            ),
            sd=MetricDetail(
                score=round(sd_val, 2),
                description_zh="回答各句的差異性（低=冗餘重複）",
                description_en="Diversity of sentences in response (low = redundant)",
            ),
            diagnosis=diag_en,
            diagnosis_zh=diag_zh,
            ai_summary=ai_summary,
            total_llm_calls=self._llm_call_count,
            time_ms=elapsed,
        )

    # --- Helpers ---

    def _parse_output_tags(self, text: str) -> list[str]:
        """Extract lines from <output></output> tags."""
        match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
        content = match.group(1) if match else text
        lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        # Remove numbering prefixes
        cleaned = []
        for line in lines:
            cleaned.append(re.sub(r"^\d+[\.\)]\s*", "", line))
        return cleaned
