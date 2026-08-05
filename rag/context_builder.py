from __future__ import annotations

from collections import OrderedDict
import re
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
    "required by law",
    "research-related treatment",
    "research-related treatment authorization",
    "authorization for research",
    "authorization required",
    "with authorization",
    "emergency",
    "emergency directory",
    "directory",
    "employment records",
    "business associate",
    "to the individual",
    "personal representative",
    "parent",
    "guardian",
    "caretaker",
    "law enforcement",
    "public health",
    "disaster relief",
    "abuse",
    "neglect",
    "domestic violence",
    "judicial",
    "subpoena",
    "donation",
    "organ",
    "whistleblower",
)

EMPLOYMENT_RECORD_TERMS = (
    "employment record",
    "employment records",
    "held by a covered entity in its role as employer",
    "covered entity as employer",
    "as employer",
)

MAX_EVIDENCE_SNIPPETS_PER_ITEM = 4

AI_OBLIGATION_GROUPS = OrderedDict(
    [
        ("Risk management system", ("risk management", "continuous iterative process", "lifecycle", "risk mitigation", "residual risk", "known and foreseeable risks", "document and explain the choices")),
        ("Technical documentation and record keeping", ("technical documentation", "record keeping", "record-keeping", "logs", "traceability", "instructions for use")),
        ("Human oversight", ("human oversight", "natural persons can oversee", "human operator", "operational constraints", "competence training and authority")),
        ("Cybersecurity, robustness, and resilience", ("cybersecurity", "cyber resilience", "security controls", "data poisoning", "adversarial attacks", "robustness", "accuracy")),
        ("Conformity assessment", ("conformity assessment", "prior to market placement", "provider responsibility for conformity assessment")),
        ("Quality management and post-market monitoring", ("quality management", "post-market monitoring", "post market monitoring", "serious incident", "corrective action")),
    ]
)


UNCONFIRMED_METADATA_BLOCK = (
    "[Metadata]\n"
    "Evidence type: semantic_only\n"
    "Confidence: unconfirmed\n"
    "Note: The passages below were retrieved by vector similarity and are NOT "
    "confirmed against the compliance knowledge graph. Present them as possibly "
    "related context, not as a definitive compliance determination."
)


def build_context(rows: list[dict[str, Any]], max_chars: int = 8000, unconfirmed: bool = False) -> str:
    """Convert compact retriever rows into evidence-focused LLM context.

    When ``unconfirmed`` is True the evidence came from semantic-only retrieval
    (no graph confirmation); a distinct ``[Metadata]`` header is prepended so the
    answer generator — and any human reader of the context — can tell this state
    apart from a graph-confirmed answer.
    """
    if not rows or max_chars <= 0:
        return UNCONFIRMED_METADATA_BLOCK if unconfirmed else ""

    items: list[str] = []
    seen_evidence: set[str] = set()
    grouped_rows = _group_rows(_filter_context_rows(rows))

    for row in grouped_rows:
        if _is_employment_record_row(row):
            continue
        statement_name = _clean(row.get("statement_name"))
        seed_name = _clean(row.get("seed_name"))
        evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))

        if evidence and evidence in seen_evidence:
            continue

        block = _format_item(len(items) + 1, row, evidence)
        next_context = "\n\n".join([*items, block])
        if len(next_context) > max_chars:
            if not items:
                compact_block = _format_item(len(items) + 1, row, evidence, compact=True)
                if len(compact_block) <= max_chars:
                    items.append(compact_block)
            break

        items.append(block)
        if evidence:
            seen_evidence.add(evidence)
        elif seed_name:
            seen_evidence.add(f"no-evidence:{seed_name}:{_clean(row.get('related_name'))}")

    body = "\n\n".join(items)
    if unconfirmed:
        return f"{UNCONFIRMED_METADATA_BLOCK}\n\n{body}" if body else UNCONFIRMED_METADATA_BLOCK
    return body


def context_row_ids(rows: list[dict[str, Any]], max_chars: int = 8000) -> list[str]:
    """Return row IDs in the same order build_context will render them."""
    if not rows or max_chars <= 0:
        return []

    items: list[str] = []
    rendered_ids: list[str] = []
    seen_evidence: set[str] = set()
    grouped_rows = _group_rows(_filter_context_rows(rows))

    for row in grouped_rows:
        if _is_employment_record_row(row):
            continue
        evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
        seed_name = _clean(row.get("seed_name"))
        if evidence and evidence in seen_evidence:
            continue
        block = _format_item(len(items) + 1, row, evidence)
        next_context = "\n\n".join([*items, block])
        if len(next_context) > max_chars:
            if not items:
                compact_block = _format_item(len(items) + 1, row, evidence, compact=True)
                if len(compact_block) <= max_chars:
                    rendered_ids.append(_context_debug_id(row))
            break
        items.append(block)
        rendered_ids.append(_context_debug_id(row))
        if evidence:
            seen_evidence.add(evidence)
        elif seed_name:
            seen_evidence.add(f"no-evidence:{seed_name}:{_clean(row.get('related_name'))}")

    return rendered_ids


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    preserve_retriever_order = any(row.get("_retriever_ranked") for row in rows)
    ordered_rows = rows if preserve_retriever_order else sorted(rows, key=_row_rank) if any(_is_permission_exception_evidence(row) for row in rows) else rows
    for source_index, row in enumerate(ordered_rows, start=1):
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
            grouped[key] = _prepare_grouped_row(row, source_index)
            continue
        grouped[key] = _merge_grouped_row(grouped[key], row, source_index)
    return list(grouped.values())


def _context_debug_id(row: dict[str, Any]) -> str:
    parts = [
        _canonical_source_document(row.get("source_document")) or "unknown",
        _clean(row.get("statement_key")) or _clean(row.get("statement_name")) or _clean(row.get("seed_name")) or "unknown",
        _clean(row.get("citation")) or _clean(row.get("article_number")),
    ]
    return "|".join(part.strip().replace("\n", " ") for part in parts)


def _canonical_source_document(value: Any) -> str:
    text = _clean(value).lower().replace("\\", "/").rsplit("/", 1)[-1].replace("-", "_")
    if text in {"gdpr", "gdpr.pdf"} or "general_data_protection_regulation" in text:
        return "gdpr.pdf"
    if text in {"hipaa", "hipaa.pdf"}:
        return "hipaa.pdf"
    if text in {"eu_ai_act", "eu_ai_act.pdf", "eu ai act", "eu ai act.pdf"} or "2024_1689" in text:
        return "eu_ai_act.pdf"
    return text


def _filter_context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_evidence_rows = any(_has_context_evidence(row) for row in rows)
    if not has_evidence_rows:
        return rows
    return [row for row in rows if not _is_missing_entity_placeholder(row)]


def _context_group_key(row: dict[str, Any]) -> str:
    semantic_group = get_semantic_group(row)
    if semantic_group:
        return f"group:{semantic_group}"
    statement_name = _clean(row.get("statement_name"))
    if statement_name:
        return f"statement:{statement_name}"
    return ""


def get_semantic_group(row: dict[str, Any]) -> str | None:
    """Return a semantic answer/context group for rows that should be merged."""
    if _is_tpo_row(row):
        return TPO_GROUP_NAME
    if _is_eu_ai_obligation_row(row):
        text = _row_text(row)
        for group_name, terms in AI_OBLIGATION_GROUPS.items():
            if any(_normalized_term(term) in text for term in terms):
                return group_name
    return None


def is_suppressed_context_row(row: dict[str, Any]) -> bool:
    return _is_employment_record_row(row)


def _prepare_grouped_row(row: dict[str, Any], source_index: int) -> dict[str, Any]:
    prepared = dict(row)
    semantic_group = get_semantic_group(row)
    if semantic_group:
        prepared["semantic_group"] = semantic_group
        prepared["statement_name"] = semantic_group
        prepared["grouped_statements"] = [_clean(row.get("statement_name")) or semantic_group]
    evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
    prepared["evidence_snippets"] = [evidence] if evidence else []
    prepared["evidence_refs"] = [source_index] if evidence else []
    return prepared


def _merge_grouped_row(base: dict[str, Any], row: dict[str, Any], source_index: int) -> dict[str, Any]:
    merged = dict(base)
    semantic_group = _clean(base.get("semantic_group")) or get_semantic_group(row)
    if semantic_group:
        evidence_limit = 3 if semantic_group in AI_OBLIGATION_GROUPS else MAX_EVIDENCE_SNIPPETS_PER_ITEM
        merged["semantic_group"] = semantic_group
        merged["statement_name"] = semantic_group
        merged["grouped_statements"] = _combine_lists(
            merged.get("grouped_statements"),
            [_clean(row.get("statement_name"))],
        )[:MAX_EVIDENCE_SNIPPETS_PER_ITEM]
        evidence = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
        merged["evidence_snippets"] = _append_limited_unique(
            merged.get("evidence_snippets"),
            evidence,
            limit=evidence_limit,
        )
        if evidence:
            merged["evidence_refs"] = _append_limited_unique(
                merged.get("evidence_refs"),
                source_index,
                limit=evidence_limit,
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
    if not _clean(merged.get("semantic_group")):
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


def _format_item(index: int, row: dict[str, Any], evidence: str, compact: bool = False) -> str:
    labels = _statement_type(row) or _join_labels(row.get("statement_labels")) or _join_labels(row.get("seed_labels"))
    statement = _clean(row.get("semantic_group")) or _clean(row.get("statement_name")) or _clean(row.get("related_name")) or "Unknown"
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
    if grouped_statements and not compact:
        lines.append("Grouped statements:")
        lines.extend(f"- {statement_name}" for statement_name in grouped_statements)
    if not compact:
        _append(lines, "Relationship", _clean(row.get("relationship")))
    _append(lines, "Evidence", _format_evidence(row, evidence))
    _append(lines, "Evidence References", _format_evidence_references(row, index, statement))
    if not compact:
        _append(lines, "Citation", _clean_citation(row.get("citation")))
        _append(lines, "Section", _clean(row.get("section_title")))
        _append(lines, "Page", _clean(row.get("page_number")))
        _append(lines, "Article", _clean(row.get("article_number")))
        _append(lines, "Clause", _clean(row.get("clause_number")))
        _append(lines, "Source Document", _clean(row.get("source_document")))
    return "\n".join(lines)


def _append(lines: list[str], label: str, value: str) -> None:
    if value:
        lines.append(f"{label}: {value}")


def _row_rank(row: dict[str, Any]) -> tuple[int, int, int, int, int, int, float, str]:
    has_evidence = 0 if (_clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))) else 1
    has_statement = 0 if _clean(row.get("statement_name")) else 1
    permission_exception_rank = 0 if _is_permission_exception_evidence(row) else 1
    broad_rank = 1 if row.get("broad_ai_statement") or "broad_preamble" in _as_list(row.get("penalties")) else 0
    concrete_rank = 0 if row.get("concrete_ai_obligation") or "concrete_obligation" in _as_list(row.get("boosts")) else 1
    semantic_group = get_semantic_group(row)
    group_rank = list(AI_OBLIGATION_GROUPS).index(semantic_group) if semantic_group in AI_OBLIGATION_GROUPS else len(AI_OBLIGATION_GROUPS)
    score = -float(row.get("score") or 0)
    return (
        has_evidence,
        permission_exception_rank,
        broad_rank,
        concrete_rank,
        group_rank,
        has_statement,
        score,
        _clean(row.get("statement_name")) or _clean(row.get("seed_name")),
    )


def _is_eu_ai_obligation_row(row: dict[str, Any]) -> bool:
    source = _clean(row.get("source_document")).lower()
    text = _row_text(row)
    has_ai_source = "eu_ai" in source or "eu ai" in source or "ai_act" in source or "2024/1689" in source
    has_ai_text = "ai system" in text or "high risk ai" in text or "this regulation" in text
    has_concrete_term = any(_normalized_term(term) in text for terms in AI_OBLIGATION_GROUPS.values() for term in terms)
    return has_concrete_term and (has_ai_source or has_ai_text or bool(row.get("concrete_ai_obligation")))


def _row_text(row: dict[str, Any]) -> str:
    return _normalized_term(
        " ".join(
            [
                _clean(row.get("statement_name")),
                _clean(row.get("related_name")),
                _clean(row.get("evidence_text")),
                _clean(row.get("source_chunk_text")),
            ]
        )
    )


def _normalized_term(value: str) -> str:
    text = value.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    return any(term in text for term in TPO_GROUP_TERMS) and not any(
        _contains_exclusion_term(text, term) for term in TPO_EXCLUSION_TERMS
    )


def _is_employment_record_row(row: dict[str, Any]) -> bool:
    text = (
        f"{_clean(row.get('statement_name'))} "
        f"{_clean(row.get('related_name'))} "
        f"{_clean(row.get('evidence_text'))} "
        f"{_clean(row.get('source_chunk_text'))}"
    ).lower()
    return any(term in text for term in EMPLOYMENT_RECORD_TERMS)


def _has_context_evidence(row: dict[str, Any]) -> bool:
    return bool(_clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text")))


def _is_missing_entity_placeholder(row: dict[str, Any]) -> bool:
    labels = set(str(label) for label in _as_list(row.get("seed_labels")))
    labels.update(str(label) for label in _as_list(row.get("related_labels")))
    labels.update(str(label) for label in _as_list(row.get("statement_labels")))
    statement_name = _clean(row.get("statement_name"))
    return not _has_context_evidence(row) and ("Entity" in labels or not statement_name or statement_name == "Unknown")


def _contains_exclusion_term(text: str, term: str) -> bool:
    if term == "organ":
        return re.search(r"\borgan\b", text) is not None
    return term in text


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


def _clean_citation(value: Any) -> str:
    values = [_clean(item) for item in _split_joined_value(value)]
    cleaned = []
    seen: set[str] = set()
    for item in values:
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    if len(cleaned) > 4:
        return ""
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
    return unique[:MAX_EVIDENCE_SNIPPETS_PER_ITEM]


def _format_evidence(row: dict[str, Any], fallback: str) -> str:
    snippets = [_clean(item) for item in _as_list(row.get("evidence_snippets")) if _clean(item)]
    if not snippets:
        snippets = [_clean(item) for item in _split_joined_value(fallback) if _clean(item)]
    return "; ".join(snippets[:MAX_EVIDENCE_SNIPPETS_PER_ITEM])


def _format_evidence_references(row: dict[str, Any], context_index: int, statement: str) -> str:
    refs = []
    for ref in _as_list(row.get("evidence_refs")):
        try:
            refs.append(f"[{int(ref)}]")
        except (TypeError, ValueError):
            continue
    if not refs:
        refs = [f"[{context_index}]"]
    source_document = _clean(row.get("source_document"))
    prefix = ", ".join(refs[:MAX_EVIDENCE_SNIPPETS_PER_ITEM])
    details = [prefix, statement]
    if source_document:
        details.append(source_document)
    return ", ".join(details)


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
