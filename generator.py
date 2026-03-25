"""
Generation module — calls FoxBrain (or OpenAI fallback) with citation extraction.
Decoupled from retrieval so it can be tested independently.
"""
import re
import time
import asyncio
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

import config
import prompts

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    source_id: int          # [來源 N]
    doc_id: str
    page_hint: str
    chunk_text: str         # the source chunk
    reason: str = ""        # why this was cited


@dataclass
class GenerationOutput:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    model_used: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    time_ms: float = 0


class Generator:
    """
    Generates answers using FoxBrain API with OpenAI fallback.
    Extracts inline citations [來源 N] and generates citation reasons.
    """

    def __init__(self):
        self._foxbrain: AsyncOpenAI | None = None
        self._openai: AsyncOpenAI | None = None

    def _get_foxbrain(self) -> AsyncOpenAI:
        if self._foxbrain is None:
            self._foxbrain = AsyncOpenAI(
                api_key=config.foxbrain.api_key,
                base_url=config.foxbrain.base_url,
            )
        return self._foxbrain

    def _get_openai(self) -> AsyncOpenAI:
        if self._openai is None:
            self._openai = AsyncOpenAI(api_key=config.openai_cfg.api_key)
        return self._openai

    async def generate(
        self,
        query: str,
        context: str,
        source_chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> GenerationOutput:
        """
        Generate an answer with citations.

        Args:
            query: User's question.
            context: Formatted context string (with [來源 N] prefixes).
            source_chunks: List of {doc_id, page_hint, text} for citation tracking.
            chat_history: Previous conversation turns [{role, content}, ...].

        Returns:
            GenerationOutput with answer, citations, and timing.
        """
        prompt = prompts.RAG_GENERATION.format(context=context, question=query)
        messages = [
            {"role": "system", "content": "你是一個專業的企業知識助理。請基於提供的參考檔案精準回答問題。"},
        ]
        # Inject chat history (last 3 turns max to save tokens)
        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"][:500]})
        messages.append({"role": "user", "content": prompt})

        t0 = time.time()
        answer, model_used, tokens_in, tokens_out = await self._call_llm(messages)
        elapsed = (time.time() - t0) * 1000

        # Extract citations from answer
        citations = self._extract_citations(answer, source_chunks)

        # Generate citation reasons (async, non-blocking)
        await self._generate_citation_reasons(query, answer, citations)

        return GenerationOutput(
            answer=answer,
            citations=citations,
            model_used=model_used,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            time_ms=elapsed,
        )

    async def _call_llm(self, messages: list[dict]) -> tuple[str, str, int, int]:
        """Try FoxBrain first (5s timeout), fallback to OpenAI."""
        # Try FoxBrain with short timeout
        try:
            client = self._get_foxbrain()
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=config.foxbrain.model,
                    messages=messages,
                    temperature=config.foxbrain.temperature,
                    max_tokens=config.foxbrain.max_tokens,
                ),
                timeout=10.0,
            )
            answer = resp.choices[0].message.content.strip()
            return (
                answer,
                config.foxbrain.model,
                resp.usage.prompt_tokens if resp.usage else 0,
                resp.usage.completion_tokens if resp.usage else 0,
            )
        except Exception as e:
            logger.warning(f"FoxBrain failed, falling back to OpenAI: {e}")

        # Fallback to OpenAI
        client = self._get_openai()
        resp = await client.chat.completions.create(
            model=config.openai_cfg.judge_model,
            messages=messages,
            temperature=config.foxbrain.temperature,
            max_tokens=config.foxbrain.max_tokens,
        )
        answer = resp.choices[0].message.content.strip()
        return (
            answer,
            config.openai_cfg.judge_model,
            resp.usage.prompt_tokens if resp.usage else 0,
            resp.usage.completion_tokens if resp.usage else 0,
        )

    def _extract_citations(self, answer: str, source_chunks: list[dict]) -> list[Citation]:
        """Find [來源 N] markers in the answer and map to source chunks."""
        cited_ids = set(int(m) for m in re.findall(r"\[來源\s*(\d+)\]", answer))
        citations = []
        for cid in sorted(cited_ids):
            idx = cid - 1  # 1-indexed
            if 0 <= idx < len(source_chunks):
                chunk = source_chunks[idx]
                citations.append(Citation(
                    source_id=cid,
                    doc_id=chunk.get("doc_id", ""),
                    page_hint=chunk.get("page_hint", ""),
                    chunk_text=chunk.get("text", "")[:200],
                ))
        return citations

    async def _generate_citation_reasons(
        self,
        query: str,
        answer: str,
        citations: list[Citation],
    ):
        """Generate a one-line reason for each citation (async)."""
        if not citations:
            return

        client = self._get_openai()
        import asyncio

        async def _gen_reason(cite: Citation):
            try:
                # Find where this source is used in the answer
                pattern = rf"\[來源\s*{cite.source_id}\]"
                match = re.search(pattern, answer)
                usage_context = ""
                if match:
                    start = max(0, match.start() - 60)
                    end = min(len(answer), match.end() + 20)
                    usage_context = answer[start:end]

                resp = await client.chat.completions.create(
                    model=config.openai_cfg.judge_model,
                    messages=[{
                        "role": "user",
                        "content": prompts.RAG_CITATION_REASON.format(
                            question=query,
                            chunk=cite.chunk_text,
                            usage=usage_context,
                        ),
                    }],
                    temperature=0,
                    max_tokens=60,
                )
                cite.reason = resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Failed to generate citation reason: {e}")
                cite.reason = ""

        await asyncio.gather(*[_gen_reason(c) for c in citations])
