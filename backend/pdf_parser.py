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
            pages_text.append(f"[Page {page_num + 1}]\n{text.strip()}")

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
