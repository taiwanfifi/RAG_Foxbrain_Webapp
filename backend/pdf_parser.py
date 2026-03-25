"""
PDF text extraction — decoupled from the rest of the pipeline.
Uses PyMuPDF (fitz) for fast, accurate Chinese text extraction.
"""
import logging
from pathlib import Path
from dataclasses import dataclass

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    filename: str
    text: str
    page_count: int
    file_size_bytes: int

    @property
    def file_size_mb(self) -> str:
        return f"{self.file_size_bytes / 1024 / 1024:.1f} MB"


def parse_pdf(file_path: str | Path) -> ParsedDocument:
    """
    Extract text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ParsedDocument with extracted text and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))
    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            cleaned = _clean_pdf_text(text.strip())
            if cleaned:
                pages_text.append(f"[Page {page_num + 1}]\n{cleaned}")

    full_text = "\n\n".join(pages_text)
    result = ParsedDocument(
        filename=path.name,
        text=full_text,
        page_count=len(doc),
        file_size_bytes=path.stat().st_size,
    )
    doc.close()

    logger.info(f"Parsed {path.name}: {result.page_count} pages, {len(full_text)} chars")
    return result


def _clean_pdf_text(text: str) -> str:
    """Remove common PDF artifacts: headers, footers, page numbers, excessive whitespace."""
    import re
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone page numbers (e.g., "44", "第7頁")
        if re.match(r'^[\d]+$', stripped):
            continue
        if re.match(r'^第\d+頁', stripped):
            continue
        # Skip very short lines that are likely headers/footers
        if len(stripped) < 3 and not re.search(r'[\u4e00-\u9fff]', stripped):
            continue
        cleaned.append(line)
    result = '\n'.join(cleaned)
    # Collapse excessive whitespace
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


def scan_folder(folder_path: str | Path) -> list[ParsedDocument]:
    """
    Scan a folder for PDFs and parse all of them.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    docs = []
    for pdf_file in sorted(folder.glob("*.pdf")):
        try:
            docs.append(parse_pdf(pdf_file))
        except Exception as e:
            logger.error(f"Failed to parse {pdf_file.name}: {e}")
    return docs
