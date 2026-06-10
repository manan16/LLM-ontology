from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1

from app.logger import get_logger
from ingestion.loaders import LoadedDocument


SECTION_HEADING_PATTERNS = [
    re.compile(r"^\s{0,3}#{1,6}\s+(.+)$"),
    re.compile(r"^\s*(Article|Section|Chapter|Part|Annex)\s+([A-Za-z0-9.\-]+)\b.*$", re.IGNORECASE),
    re.compile(r"^\s*(\d+(\.\d+){0,3})\s+.+$"),
]
logger = get_logger(__name__)


@dataclass(slots=True)
class SectionBlock:
    section_id: str
    title: str
    text: str
    start_char: int
    end_char: int


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    regulation_name: str
    section_id: str
    section_title: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int


def split_into_sections(text: str) -> list[SectionBlock]:
    logger.debug("Splitting document text into sections chars=%s", len(text))
    lines = text.splitlines()
    sections: list[SectionBlock] = []

    current_title: str | None = None
    current_lines: list[str] = []
    current_start = 0
    cursor = 0
    section_counter = 0

    for line in lines:
        stripped = line.strip()
        is_heading = stripped and any(pattern.match(stripped) for pattern in SECTION_HEADING_PATTERNS)
        if is_heading:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append(
                        SectionBlock(
                            section_id=f"sec-{section_counter:04d}",
                            title=current_title or "Preamble",
                            text=body,
                            start_char=current_start,
                            end_char=cursor,
                        )
                    )
                    section_counter += 1
            current_title = stripped.lstrip("#").strip()
            current_lines = [line]
            current_start = cursor
        else:
            if current_title is None:
                current_title = "Preamble"
            current_lines.append(line)
        cursor += len(line) + 1

    body = "\n".join(current_lines).strip()
    if body:
        sections.append(
            SectionBlock(
                section_id=f"sec-{section_counter:04d}",
                title=current_title or "Preamble",
                text=body,
                start_char=current_start,
                end_char=max(cursor, len(text)),
            )
        )

    logger.debug("Identified %s sections", len(sections))
    return sections or [
        SectionBlock(
            section_id="sec-0000",
            title="Document",
            text=text,
            start_char=0,
            end_char=len(text),
        )
    ]


def build_chunks(document: LoadedDocument, max_chars: int, overlap_chars: int) -> list[DocumentChunk]:
    if overlap_chars >= max_chars:
        raise ValueError("chunk overlap must be smaller than max chunk size")

    logger.debug(
        "Building chunks for document=%s max_chars=%s overlap_chars=%s",
        document.name,
        max_chars,
        overlap_chars,
    )
    sections = split_into_sections(document.text)
    chunks: list[DocumentChunk] = []

    for section in sections:
        section_text = section.text
        offset = 0
        chunk_index = 0
        while offset < len(section_text):
            end = min(len(section_text), offset + max_chars)
            candidate = section_text[offset:end]

            if end < len(section_text):
                split_at = candidate.rfind("\n\n")
                if split_at == -1:
                    split_at = candidate.rfind("\n")
                if split_at != -1 and split_at > max_chars // 3:
                    end = offset + split_at
                    candidate = section_text[offset:end]

            candidate = candidate.strip()
            if not candidate:
                break

            chunk_id = _build_chunk_id(document.name, section.section_id, chunk_index, candidate)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    regulation_name=document.name,
                    section_id=section.section_id,
                    section_title=section.title,
                    chunk_index=chunk_index,
                    text=candidate,
                    start_char=section.start_char + offset,
                    end_char=section.start_char + end,
                )
            )
            logger.debug(
                "Created chunk id=%s section=%s index=%s chars=%s start=%s end=%s",
                chunk_id,
                section.title,
                chunk_index,
                len(candidate),
                section.start_char + offset,
                section.start_char + end,
            )
            chunk_index += 1
            if end >= len(section_text):
                break
            offset = max(0, end - overlap_chars)

    logger.debug("Built %s chunks for document=%s", len(chunks), document.name)
    return chunks


def split_chunk_for_retry(chunk: DocumentChunk, max_chars: int) -> list[DocumentChunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]

    split_at = _find_retry_split_index(chunk.text)
    left_text = chunk.text[:split_at].strip()
    right_text = chunk.text[split_at:].strip()

    if not left_text or not right_text:
        midpoint = len(chunk.text) // 2
        left_text = chunk.text[:midpoint].strip()
        right_text = chunk.text[midpoint:].strip()

    if not left_text or not right_text:
        return [chunk]

    left_offset = chunk.text.find(left_text)
    right_offset = chunk.text.rfind(right_text)

    children = [
        DocumentChunk(
            chunk_id=_build_chunk_id(chunk.regulation_name, chunk.section_id, chunk.chunk_index * 10 + 1, left_text),
            regulation_name=chunk.regulation_name,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            chunk_index=chunk.chunk_index * 10 + 1,
            text=left_text,
            start_char=chunk.start_char + max(left_offset, 0),
            end_char=chunk.start_char + max(left_offset, 0) + len(left_text),
        ),
        DocumentChunk(
            chunk_id=_build_chunk_id(chunk.regulation_name, chunk.section_id, chunk.chunk_index * 10 + 2, right_text),
            regulation_name=chunk.regulation_name,
            section_id=chunk.section_id,
            section_title=chunk.section_title,
            chunk_index=chunk.chunk_index * 10 + 2,
            text=right_text,
            start_char=chunk.start_char + max(right_offset, 0),
            end_char=chunk.start_char + max(right_offset, 0) + len(right_text),
        ),
    ]
    logger.debug(
        "Split timed-out chunk id=%s chars=%s into [%s, %s]",
        chunk.chunk_id,
        len(chunk.text),
        len(left_text),
        len(right_text),
    )
    return children


def _build_chunk_id(regulation_name: str, section_id: str, chunk_index: int, text: str) -> str:
    digest = sha1(text.encode("utf-8")).hexdigest()[:10]
    safe_name = regulation_name.lower().replace(" ", "-")
    return f"{safe_name}:{section_id}:{chunk_index}:{digest}"


def _find_retry_split_index(text: str) -> int:
    midpoint = len(text) // 2
    candidates = [
        text.rfind("\n\n", max(0, midpoint - 400), min(len(text), midpoint + 400)),
        text.rfind("\n", max(0, midpoint - 300), min(len(text), midpoint + 300)),
        text.rfind(". ", max(0, midpoint - 250), min(len(text), midpoint + 250)),
        text.rfind("; ", max(0, midpoint - 250), min(len(text), midpoint + 250)),
    ]
    for candidate in candidates:
        if candidate > 0:
            return candidate + 1
    return midpoint
