from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.logger import get_logger


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
logger = get_logger(__name__)


@dataclass(slots=True)
class LoadedDocument:
    name: str
    path: str
    file_type: str
    text: str


def load_document(path: str | Path) -> LoadedDocument:
    document_path = Path(path).expanduser().resolve()
    logger.debug("Loading document from path=%s", document_path)
    if not document_path.exists():
        raise FileNotFoundError(f"Document not found: {document_path}")
    if document_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {document_path.suffix}")

    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        text = _load_pdf(document_path)
    else:
        text = document_path.read_text(encoding="utf-8")

    if not text.strip():
        raise ValueError(f"No text extracted from document: {document_path}")

    logger.debug(
        "Loaded document name=%s type=%s chars=%s",
        document_path.name,
        suffix,
        len(text),
    )
    return LoadedDocument(
        name=document_path.name,
        path=str(document_path),
        file_type=suffix,
        text=text,
    )


def _load_pdf(path: Path) -> str:
    logger.debug("Extracting text from PDF path=%s", path)
    reader = PdfReader(str(path))
    parts: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            parts.append(f"[Page {index}]\n{page_text.strip()}")
        logger.debug("Processed PDF page=%s extracted_chars=%s", index, len(page_text))
    return "\n\n".join(parts)
