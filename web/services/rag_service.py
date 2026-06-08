from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

from rag import AnswerGenerator, GraphRetriever, build_context
from rag.context_builder import context_row_ids
from rag.query_planner import QueryPlan


LOGGER = logging.getLogger(__name__)


def answer_question(
    question: str,
    limit: int = 30,
    show_context: bool = False,
    debug: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    """Run the existing KG-RAG pipeline and return web/CLI friendly data."""
    cleaned_question = (question or "").strip()
    if not cleaned_question:
        raise ValueError("Please enter a compliance question.")

    safe_limit = _safe_limit(limit)
    started_at = perf_counter()
    timings: dict[str, int] = {}

    routing_started = perf_counter()
    _log_stage("routing_started", question=cleaned_question)
    retriever = GraphRetriever()
    timings["routing_ms"] = _elapsed_ms(routing_started)

    retrieval_started = perf_counter()
    graph_query_started = perf_counter()
    _log_stage("retrieval_started", question=cleaned_question, limit=safe_limit)
    _log_stage("graph_query_started", question=cleaned_question, limit=safe_limit)
    rows = retriever.retrieve(cleaned_question, limit=safe_limit)
    timings["graph_query_ms"] = _elapsed_ms(graph_query_started)
    timings["retrieval_ms"] = _elapsed_ms(retrieval_started)

    ranking_started = perf_counter()
    _log_stage("ranking_started", rows_retrieved=len(rows))
    context = build_context(rows)
    evidence = _extract_evidence(rows)
    timings["ranking_ms"] = _elapsed_ms(ranking_started)

    generation_started = perf_counter()
    _log_stage("generation_started", evidence_items=len(evidence))
    answer = AnswerGenerator(model=model).generate(cleaned_question, context)
    timings["generation_ms"] = _elapsed_ms(generation_started)
    elapsed_ms = _elapsed_ms(started_at)
    timings["total_ms"] = elapsed_ms
    _log_stage("pipeline_completed", **timings)

    return {
        "question": cleaned_question,
        "answer": answer,
        "evidence": evidence,
        "context": context if show_context else "",
        "debug": _build_debug(retriever, rows) if debug else {},
        "metrics": _build_metrics(rows, evidence, safe_limit, elapsed_ms, model, timings),
    }


def _safe_limit(limit: int) -> int:
    try:
        parsed = int(limit)
    except (TypeError, ValueError):
        return 30
    return min(max(parsed, 1), 100)


def _extract_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence_items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        evidence_text = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
        statement = _clean(row.get("statement_name")) or _clean(row.get("related_name")) or "Unknown"
        source_document = _clean(row.get("source_document"))
        if not evidence_text:
            continue
        key = (statement, source_document, evidence_text)
        if key in seen:
            continue
        seen.add(key)
        evidence_items.append(
            {
                "id": len(evidence_items) + 1,
                "statement": statement,
                "source_document": source_document,
                "evidence_text": evidence_text,
                "citation": _clean(row.get("citation")),
                "score": row.get("score"),
                "source_group": _source_group(row),
            }
        )
    return evidence_items


def _build_metrics(
    rows: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    limit: int,
    elapsed_ms: int,
    model: str | None,
    timings: dict[str, int] | None = None,
) -> dict[str, Any]:
    source_coverage = _source_coverage(rows, evidence)
    stage_timings = timings or {}
    return {
        "top_k": limit,
        "rows_retrieved": len(rows),
        "evidence_items": len(evidence),
        "elapsed_ms": elapsed_ms,
        "routing_ms": stage_timings.get("routing_ms", 0),
        "retrieval_ms": stage_timings.get("retrieval_ms", 0),
        "graph_query_ms": stage_timings.get("graph_query_ms", 0),
        "ranking_ms": stage_timings.get("ranking_ms", 0),
        "generation_ms": stage_timings.get("generation_ms", 0),
        "total_ms": stage_timings.get("total_ms", elapsed_ms),
        "model": model or "qwen3:8b",
        "source_coverage": source_coverage,
        "breakdown": {
            "hipaa": source_coverage.get("hipaa", 0),
            "eu_ai_act": source_coverage.get("eu_ai_act", 0),
            "gdpr": source_coverage.get("gdpr", 0),
            "source_chunks": sum(1 for row in rows if "SourceChunk" in row.get("related_labels", [])),
            "entity_nodes": sum(
                1
                for row in rows
                if "Entity" in row.get("seed_labels", []) or "Entity" in row.get("related_labels", [])
            ),
        },
    }


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _log_stage(stage: str, **fields: Any) -> None:
    LOGGER.info("kep_pipeline_stage", extra={"stage": stage, **fields})


def _build_debug(retriever: GraphRetriever, rows: list[dict[str, Any]]) -> dict[str, Any]:
    plan = retriever.last_query_plan
    selection = dict(getattr(retriever, "last_selection_debug", {}) or {})
    selection["context_row_ids_rendered"] = context_row_ids(rows)
    return {
        "query_plan": _serialize_plan(plan),
        "top_rows": [_serialize_row(row) for row in rows[:20]],
        "expansion": retriever.last_expansion_debug,
        "selection": selection,
    }


def _serialize_plan(plan: QueryPlan | None) -> dict[str, Any]:
    if plan is None:
        return {}
    return {
        "category": plan.category,
        "intent": plan.intent,
        "detected_phrases": plan.phrases,
        "detected_actors": plan.detected_actors,
        "detected_objects": plan.detected_objects,
        "detected_domains": plan.detected_domains,
        "concept_groups": plan.concept_groups,
        "ambiguous_terms": plan.ambiguous_terms,
        "expansion_terms": plan.expansion_terms,
        "sub_questions": plan.sub_questions,
        "explicit_regulations": getattr(plan, "explicit_regulations", []),
        "inferred_regulations": getattr(plan, "inferred_regulations", []),
        "target_source_documents": getattr(plan, "target_source_documents", []),
        "explicit_single_regulation": getattr(plan, "explicit_single_regulation", False),
        "cross_regulation": getattr(plan, "cross_regulation", False),
        "is_mental_health_query": getattr(plan, "is_mental_health_query", False),
        "mental_health_intent_labels": getattr(plan, "mental_health_intent_labels", []),
        "mental_health_expansion_terms": getattr(plan, "mental_health_expansion_terms", []),
    }


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": row.get("score"),
        "route": row.get("query_route"),
        "statement": row.get("statement_name") or row.get("seed_name"),
        "relationship": row.get("relationship"),
        "source_group": row.get("source_group"),
        "source_document": row.get("source_document"),
        "matched_concept_groups": row.get("matched_concept_groups") or [],
        "matched_terms": row.get("matched_terms") or [],
        "matched_topic_groups": row.get("matched_topic_groups") or [],
        "boosts": row.get("boosts") or [],
        "penalties": row.get("penalties") or [],
    }


def _source_coverage(rows: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, int]:
    coverage = {"hipaa": 0, "eu_ai_act": 0, "gdpr": 0, "unknown": 0}
    source_items: list[dict[str, Any]] = [*rows, *evidence]
    for item in source_items:
        coverage[_source_group(item)] += 1
    return coverage


def _source_group(row: dict[str, Any]) -> str:
    explicit = _clean(row.get("source_group")).lower()
    source_document = _clean(row.get("source_document")).lower()
    text = f"{source_document} {_clean(row.get('statement'))} {_clean(row.get('statement_name'))}".lower()
    if explicit in {"hipaa", "eu_ai_act", "gdpr"}:
        return explicit
    if "gdpr" in text or "general data protection regulation" in text:
        return "gdpr"
    if "hipaa" in text or "phi" in text or "covered entity" in text:
        return "hipaa"
    if "eu_ai" in text or "eu ai" in text or "ai_act" in text or "2024/1689" in text:
        return "eu_ai_act"
    return "unknown"


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()
