from __future__ import annotations

from collections import OrderedDict
from typing import Any


STATEMENT_TYPE_LABELS = {
    "Statement",
    "Obligation",
    "Requirement",
    "Permission",
    "Prohibition",
    "Exception",
    "Definition",
    "Risk",
    "Control",
}

PERMISSION_EXCEPTION_EVIDENCE_TERMS = (
    "may disclose",
    "may use",
    "may use or disclose",
    "permitted disclosure",
    "permitted use",
    "without authorization",
    "authorization not required",
    "not required",
    "opportunity to agree or object is not required",
    "except",
)

TPO_GROUP_NAME = "Treatment, Payment, and Health Care Operations"

TPO_GROUP_TERMS = (
    "treatment, payment, or operations",
    "treatment, payment, and health care operations",
    "treatment payment health care operations",
    "health care operations",
    "organized health care arrangement",
    "ohca",
    "treatment",
    "payment",
)

TPO_EXCLUSION_TERMS = (
    "research-related treatment",
    "authorization for research",
    "emergency",
    "directory",
    "employment records",
    "business associate",
    "to the individual",
    "personal representative",
    "parent",
    "guardian",
    "law enforcement",
    "public health",
    "disaster relief",
    "abuse",
    "neglect",
    "domestic violence",
    "judicial",
    "subpoena",
    "donation",
    "whistleblower",
)

MAX_EVIDENCE_SNIPPETS_PER_ITEM = 3


def build_context(rows: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Convert compact retriever rows into evidence-focused LLM context."""
    if not rows or max_chars <= 0:
        return ""

    items: list[str] = []
    seen_evidence: set[str] = set()
    grouped_rows = _group_rows(rows)

    for row in grouped_rows:
        statement_name = _clean(row.get("statement_name"))
        seed_name = _clean(row.get("seed_name"))
        evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))

        if evidence and evidence in seen_evidence:
            continue

        block = _format_item(len(items) + 1, row, evidence)
        next_context = "\n\n".join([*items, block])
        if len(next_context) > max_chars:
            break

        items.append(block)
        if evidence:
            seen_evidence.add(evidence)
        elif seed_name:
            seen_evidence.add(f"no-evidence:{seed_name}:{_clean(row.get('related_name'))}")

    return "\n\n".join(items)


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in sorted(rows, key=_row_rank):
        key = _context_group_key(row)
        if not key:
            key = "|".join(
                [
                    _clean(row.get("seed_name")),
                    _clean(row.get("relationship")),
                    _clean(row.get("related_name")),
                    _clean(row.get("evidence_text")),
                ]
            )
        if key not in grouped:
            grouped[key] = _prepare_grouped_row(row)
            continue
        grouped[key] = _merge_grouped_row(grouped[key], row)
    return list(grouped.values())


def _context_group_key(row: dict[str, Any]) -> str:
    if _is_tpo_row(row):
        return f"group:{TPO_GROUP_NAME}"
    statement_name = _clean(row.get("statement_name"))
    if statement_name:
        return f"statement:{statement_name}"
    return ""


def _prepare_grouped_row(row: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    if _is_tpo_row(row):
        prepared["statement_name"] = TPO_GROUP_NAME
        prepared["grouped_statements"] = [_clean(row.get("statement_name")) or TPO_GROUP_NAME]
    evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
    prepared["evidence_snippets"] = [evidence] if evidence else []
    return prepared


def _merge_grouped_row(base: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if _is_tpo_row(base) or _is_tpo_row(row):
        merged["statement_name"] = TPO_GROUP_NAME
        merged["grouped_statements"] = _combine_lists(
            merged.get("grouped_statements"),
            [_clean(row.get("statement_name"))],
        )
        evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
        merged["evidence_snippets"] = _append_limited_unique(
            merged.get("evidence_snippets"),
            evidence,
            limit=MAX_EVIDENCE_SNIPPETS_PER_ITEM,
        )
    for key in (
        "seed_name",
        "relationship",
        "related_name",
        "citation",
        "section_title",
        "page_number",
        "article_number",
        "clause_number",
        "source_document",
    ):
        merged[key] = _combine_values(merged.get(key), row.get(key))
    if not _is_tpo_row(merged):
        merged["evidence_text"] = _combine_limited_values(
            merged.get("evidence_text"),
            row.get("evidence_text"),
            limit=MAX_EVIDENCE_SNIPPETS_PER_ITEM,
        )
        merged["source_chunk_text"] = _combine_limited_values(
            merged.get("source_chunk_text"),
            row.get("source_chunk_text"),
            limit=MAX_EVIDENCE_SNIPPETS_PER_ITEM,
        )
    for key in ("seed_labels", "statement_labels", "related_labels"):
        merged[key] = _combine_lists(merged.get(key), row.get(key))
    merged["score"] = max(float(merged.get("score") or 0), float(row.get("score") or 0))
    return merged


def _format_item(index: int, row: dict[str, Any], evidence: str) -> str:
    labels = _statement_type(row) or _join_labels(row.get("statement_labels")) or _join_labels(row.get("seed_labels"))
    statement = TPO_GROUP_NAME if _is_tpo_row(row) else _clean(row.get("statement_name")) or _clean(row.get("related_name")) or "Unknown"
    actor = _clean(row.get("seed_name"))
    if actor == statement:
        actor = _clean(row.get("related_name"))

    lines = [
        f"[{index}]",
        f"Statement: {statement}",
    ]
    _append(lines, "Type", labels)
    _append(lines, "Entity/Actor", actor)
    grouped_statements = _clean_grouped_statements(row)
    if grouped_statements:
        lines.append("Grouped statements:")
        lines.extend(f"- {statement_name}" for statement_name in grouped_statements)
    _append(lines, "Relationship", _clean(row.get("relationship")))
    _append(lines, "Evidence", _format_evidence(row, evidence))
    _append(lines, "Citation", _clean(row.get("citation")))
    _append(lines, "Section", _clean(row.get("section_title")))
    _append(lines, "Page", _clean(row.get("page_number")))
    _append(lines, "Article", _clean(row.get("article_number")))
    _append(lines, "Clause", _clean(row.get("clause_number")))
    _append(lines, "Source Document", _clean(row.get("source_document")))
    return "\n".join(lines)


def _append(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _row_rank(row: dict[str, Any]) -> tuple[int, int, int, float, str]:
    has_evidence = 0 if (_clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))) else 1
    has_statement = 0 if _clean(row.get("statement_name")) else 1
    permission_exception_rank = 0 if _is_permission_exception_evidence(row) else 1
    score = -float(row.get("score") or 0)
    return (
        has_evidence,
        permission_exception_rank,
        has_statement,
        score,
        _clean(row.get("statement_name")) or _clean(row.get("seed_name")),
    )


def _is_permission_exception_evidence(row: dict[str, Any]) -> bool:
    labels = {str(label) for label in _as_list(row.get("statement_labels"))}
    text = (
        f"{_clean(row.get('statement_name'))} "
        f"{_clean(row.get('evidence_text'))} "
        f"{_clean(row.get('source_chunk_text'))}"
    ).lower()
    return bool(labels.intersection({"Permission", "Exception"})) and any(
        term in text for term in PERMISSION_EXCEPTION_EVIDENCE_TERMS
    )


def _is_tpo_row(row: dict[str, Any]) -> bool:
    text = (
        f"{_clean(row.get('statement_name'))} "
        f"{_clean(row.get('evidence_text'))}"
    ).lower()
    return any(term in text for term in TPO_GROUP_TERMS) and not any(term in text for term in TPO_EXCLUSION_TERMS)


def _statement_type(row: dict[str, Any]) -> str:
    labels = []
    for label in _as_list(row.get("statement_labels")):
        if str(label) in STATEMENT_TYPE_LABELS:
            labels.append(str(label))
    return ", ".join(labels)


def _join_labels(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return _clean(value)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _combine_values(left: Any, right: Any) -> str:
    values = [*_as_list(left), *_as_list(right)]
    cleaned = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return "; ".join(cleaned)


def _combine_limited_values(left: Any, right: Any, limit: int) -> str:
    values = []
    for value in [*_as_list(left), *_as_list(right)]:
        values.extend(_split_joined_value(value))
    cleaned = []
    seen: set[str] = set()
    for value in values:
        text = _clean(value)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return "; ".join(cleaned)


def _combine_lists(left: Any, right: Any) -> list[Any]:
    values = [*_as_list(left), *_as_list(right)]
    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key and key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _append_limited_unique(left: Any, value: Any, limit: int) -> list[str]:
    values = [_clean(item) for item in _as_list(left)]
    text = _clean(value)
    if text and text not in values and len(values) < limit:
        values.append(text)
    return values[:limit]


def _clean_grouped_statements(row: dict[str, Any]) -> list[str]:
    statements = [_clean(item) for item in _as_list(row.get("grouped_statements")) if _clean(item)]
    unique = []
    seen: set[str] = set()
    for statement in statements:
        if statement not in seen and statement != TPO_GROUP_NAME:
            seen.add(statement)
            unique.append(statement)
    return unique if len(unique) > 1 else []


def _format_evidence(row: dict[str, Any], fallback: str) -> str:
    snippets = [_clean(item) for item in _as_list(row.get("evidence_snippets")) if _clean(item)]
    if not snippets:
        snippets = [_clean(item) for item in _split_joined_value(fallback) if _clean(item)]
    return "; ".join(snippets[:MAX_EVIDENCE_SNIPPETS_PER_ITEM])


def _split_joined_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value]
    text = _clean(value)
    return [item.strip() for item in text.split("; ") if item.strip()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
