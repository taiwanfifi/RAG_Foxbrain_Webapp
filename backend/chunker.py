"""
Text chunking — splits documents into overlapping chunks for retrieval.
Pure Python, no external dependency needed.
"""
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    doc_id: str          # source filename
    chunk_id: int        # sequential index within doc
    text: str
    page_hint: str = ""  # e.g. "p.3" extracted from [Page 3] markers
    metadata: dict = field(default_factory=dict)

    @property
    def uid(self) -> str:
        return f"{self.doc_id}::{self.chunk_id}"


# Separators ordered by priority (try paragraph first, then sentence, then char)
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " "]


def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """
    Recursive character text splitter (same logic as LangChain's).

    Args:
        text: Full document text.
        doc_id: Identifier for the source document.
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    raw_chunks = _recursive_split(text, _SEPARATORS, chunk_size)

    # Apply overlap
    chunks = []
    for i, raw in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev = raw_chunks[i - 1]
            overlap = prev[-chunk_overlap:]
            raw = overlap + raw

        page_hint = _extract_page_hint(raw)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=i,
            text=raw.strip(),
            page_hint=page_hint,
        ))

    logger.info(f"Chunked {doc_id}: {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split text recursively using separator hierarchy."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that exists in the text
    sep = ""
    for s in separators:
        if s in text:
            sep = s
            break

    if not sep:
        # Fallback: hard split by chunk_size
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > chunk_size:
                # Recurse with next separator
                remaining_seps = separators[separators.index(sep) + 1:]
                if remaining_seps:
                    chunks.extend(_recursive_split(part, remaining_seps, chunk_size))
                else:
                    chunks.extend([part[i:i + chunk_size] for i in range(0, len(part), chunk_size)])
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks


def _extract_page_hint(text: str) -> str:
    """Extract page number from [Page N] markers if present."""
    match = re.search(r"\[Page\s+(\d+)\]", text)
    return f"p.{match.group(1)}" if match else ""
