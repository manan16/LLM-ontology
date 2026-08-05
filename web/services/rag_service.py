from __future__ import annotations

import atexit
import logging
from threading import Lock
from time import perf_counter
from typing import Any

from app.config import get_settings
from rag import (
    AnswerGenerator,
    DeterminationGenerator,
    GraphRetriever,
    HybridRetriever,
    SemanticRetriever,
    build_context,
    insufficient_determination,
    unconfirmed_determination,
)
from rag.context_builder import context_row_ids
from rag.query_planner import QueryPlan


LOGGER = logging.getLogger(__name__)
_retriever: GraphRetriever | None = None
_retriever_lock = Lock()
_semantic_retriever: SemanticRetriever | None = None
_semantic_lock = Lock()
NO_EVIDENCE_MESSAGE = "No sufficient regulatory evidence was found in the knowledge graph for this question."
UNCONFIRMED_MESSAGE = (
    "These passages are semantically related to your question but could not be confirmed "
    "against the compliance knowledge graph. Treat them as possibly relevant context, not "
    "as a compliance determination."
)

# Response-level classifications so the UI and callers can tell the three states
# apart: a graph-confirmed answer, a semantic-only (unconfirmed) answer, and the
# genuine no-evidence case.
CONFIDENCE_CONFIRMED = "confirmed"
CONFIDENCE_UNCONFIRMED = "unconfirmed"
CONFIDENCE_NONE = "none"
EVIDENCE_GRAPH = "graph_confirmed"
EVIDENCE_SEMANTIC_ONLY = "semantic_only"
EVIDENCE_NONE = "none"


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
    retriever = _get_retriever()
    timings["routing_ms"] = _elapsed_ms(routing_started)

    retrieval_started = perf_counter()
    _log_stage("retrieval_started", question=cleaned_question, limit=safe_limit)
    _log_stage("graph_query_started", question=cleaned_question, limit=safe_limit)
    hybrid = _build_hybrid_retriever(retriever)
    retrieval = hybrid.retrieve(cleaned_question, limit=safe_limit)
    rows = retrieval.rows
    graph_expansion = retrieval.graph_expansion
    timings["graph_query_ms"] = retrieval.graph_ms
    timings["semantic_query_ms"] = retrieval.semantic_ms
    timings["retrieval_ms"] = _elapsed_ms(retrieval_started)

    ranking_started = perf_counter()
    _log_stage("ranking_started", rows_retrieved=len(rows))
    evidence = _extract_evidence(rows)
    graph_confirmed = _has_graph_confirmed_evidence(rows)
    timings["ranking_ms"] = _elapsed_ms(ranking_started)

    # --- Case 1: nothing at all (neither graph nor semantic) -> no evidence ----
    if not evidence:
        context = build_context(rows)
        elapsed_ms = _elapsed_ms(started_at)
        timings["generation_ms"] = 0
        timings["total_ms"] = elapsed_ms
        _log_stage("generation_skipped", reason="no_evidence")
        _log_stage("pipeline_completed", **timings)
        return {
            "question": cleaned_question,
            "answer": NO_EVIDENCE_MESSAGE,
            "evidence": [],
            "graph": _build_no_evidence_graph(cleaned_question),
            "context": context if show_context else "",
            "debug": _build_debug(retriever, rows) if debug else {},
            "determination": insufficient_determination(NO_EVIDENCE_MESSAGE),
            "confidence": CONFIDENCE_NONE,
            "evidence_type": EVIDENCE_NONE,
            "graph_expansion": {},
            "metrics": {
                **_build_metrics(rows, evidence, safe_limit, elapsed_ms, model, timings),
                "no_evidence": True,
            },
            "response_source": "no_evidence",
        }

    # --- Case 2: semantic-only -> answer WITHOUT a confident determination -----
    # Graph expansion surfaced no confirmed entities/obligations, so we
    # must not stamp a verdict from vector-similarity chunks alone. We still keep
    # the semantic hits and present them as possibly-related context.
    if not graph_confirmed:
        graph = _build_graph_payload(cleaned_question, rows, evidence)
        context = build_context(rows, unconfirmed=True)

        generation_started = perf_counter()
        _log_stage("generation_started", evidence_items=len(evidence), evidence_type=EVIDENCE_SEMANTIC_ONLY)
        answer = AnswerGenerator(model=model).generate(cleaned_question, context)
        timings["generation_ms"] = _elapsed_ms(generation_started)
        # Determination LLM is deliberately skipped for semantic-only evidence.
        _log_stage("determination_skipped", reason="semantic_only_unconfirmed")

        elapsed_ms = _elapsed_ms(started_at)
        timings["total_ms"] = elapsed_ms
        _log_stage("pipeline_completed", **timings)
        return {
            "question": cleaned_question,
            "answer": answer,
            "evidence": evidence,
            "graph": graph,
            "context": context if show_context else "",
            "debug": _build_debug(retriever, rows) if debug else {},
            "determination": unconfirmed_determination(UNCONFIRMED_MESSAGE),
            "confidence": CONFIDENCE_UNCONFIRMED,
            "evidence_type": EVIDENCE_SEMANTIC_ONLY,
            "graph_expansion": graph_expansion,
            "metrics": {
                **_build_metrics(rows, evidence, safe_limit, elapsed_ms, model, timings),
                "unconfirmed": True,
                "evidence_type": EVIDENCE_SEMANTIC_ONLY,
            },
            "response_source": "semantic_only",
        }

    # --- Case 3: graph-confirmed -> full answer + determination verdict --------
    # Rank (not just concatenate) the merged graph + semantic evidence via the
    # weighted hybrid score. This is the only place ranking is applied; state
    # determination above is unaffected.
    rows = hybrid.rank_rows(rows)
    context = build_context(rows)
    evidence = _extract_evidence(rows)
    graph = _build_graph_payload(cleaned_question, rows, evidence)

    generation_started = perf_counter()
    _log_stage("generation_started", evidence_items=len(evidence))
    answer = AnswerGenerator(model=model).generate(cleaned_question, context)
    timings["generation_ms"] = _elapsed_ms(generation_started)

    determination_started = perf_counter()
    _log_stage("determination_started", evidence_items=len(evidence))
    determination = DeterminationGenerator(model=model).generate(cleaned_question, context, answer)
    timings["determination_ms"] = _elapsed_ms(determination_started)
    _log_stage("determination_completed", verdict=determination.get("verdict"))

    elapsed_ms = _elapsed_ms(started_at)
    timings["total_ms"] = elapsed_ms
    _log_stage("pipeline_completed", **timings)

    return {
        "question": cleaned_question,
        "answer": answer,
        "evidence": evidence,
        "graph": graph,
        "context": context if show_context else "",
        "debug": _build_debug(retriever, rows) if debug else {},
        "determination": determination,
        "confidence": CONFIDENCE_CONFIRMED,
        "evidence_type": EVIDENCE_GRAPH,
        "graph_expansion": graph_expansion,
        "metrics": _build_metrics(rows, evidence, safe_limit, elapsed_ms, model, timings),
        "response_source": "live",
    }


def _get_retriever() -> GraphRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                LOGGER.info("Creating shared GraphRetriever for web RAG service")
                _retriever = GraphRetriever()
    else:
        LOGGER.debug("Reusing shared GraphRetriever for web RAG service")
    return _retriever


def _get_semantic_retriever() -> SemanticRetriever | None:
    """Return the shared SemanticRetriever, or None if it cannot be initialized."""
    global _semantic_retriever
    if not get_settings().semantic_retrieval_enabled:
        return None
    if _semantic_retriever is None:
        with _semantic_lock:
            if _semantic_retriever is None:
                try:
                    LOGGER.info("Creating shared SemanticRetriever for web RAG service")
                    _semantic_retriever = SemanticRetriever()
                except Exception:
                    LOGGER.exception("Semantic retriever unavailable; falling back to graph-only retrieval")
                    return None
    return _semantic_retriever


def _build_hybrid_retriever(retriever: GraphRetriever) -> HybridRetriever:
    """Construct a per-request HybridRetriever around the shared graph retriever.

    The semantic side is resolved through ``_get_semantic_retriever`` at call
    time (via the ``semantic_provider`` callable), so tests that monkeypatch it —
    and the graceful "semantic unavailable -> graph-only" fallback — keep working.
    """
    return HybridRetriever(
        graph_retriever=retriever,
        semantic_provider=_get_semantic_retriever,
        settings=get_settings(),
    )


def _has_graph_confirmed_evidence(rows: list[dict[str, Any]]) -> bool:
    """True if at least one *non-semantic* (graph-retrieved) row carries evidence.

    Semantic rows are marked with ``semantic=True`` by SemanticRetriever. Only
    graph rows count as confirmation against the compliance graph — this is the
    signal that gates whether a confident determination may be produced.
    """
    for row in rows:
        if row.get("semantic"):
            continue
        if _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text")):
            return True
    return False


def close_retriever() -> None:
    global _retriever, _semantic_retriever
    if _semantic_retriever is not None:
        semantic_client = getattr(_semantic_retriever, "neo4j_client", None)
        semantic_close = getattr(semantic_client, "close", None)
        if callable(semantic_close):
            semantic_close()
        _semantic_retriever = None
    if _retriever is None:
        return
    neo4j_client = getattr(_retriever, "neo4j_client", None)
    close = getattr(neo4j_client, "close", None)
    if callable(close):
        close()
    _retriever = None


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
        item = {
            "id": len(evidence_items) + 1,
            "statement": statement,
            "source_document": source_document,
            "evidence_text": evidence_text,
            "citation": _clean(row.get("citation")),
            "score": row.get("score"),
            "source_group": _source_group(row),
            # Per-item provenance so the UI can mark semantic-only cards as
            # "possibly related" rather than graph-confirmed.
            "evidence_type": EVIDENCE_SEMANTIC_ONLY if row.get("semantic") else EVIDENCE_GRAPH,
        }
        # Weighted hybrid scores are present once the confirmed-state ranking has
        # run (HybridRetriever.rank_rows); absent in the other states.
        for score_field in ("semantic_score", "graph_relevance_score", "hybrid_score"):
            if row.get(score_field) is not None:
                item[score_field] = row.get(score_field)
        evidence_items.append(item)
    return evidence_items


def _build_graph_payload(question: str, rows: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a UI graph from retrieved rows without inventing KG topology."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, str]] = set()

    def add_node(node_id: str, label: str, node_type: str, **extra: Any) -> str:
        if not node_id:
            return ""
        existing = nodes.get(node_id)
        payload = {
            "id": node_id,
            "label": _graph_label(label) or node_id,
            "type": node_type,
            **{key: value for key, value in extra.items() if value not in (None, "", [])},
        }
        if existing:
            existing.update({key: value for key, value in payload.items() if value not in (None, "", [])})
        else:
            nodes[node_id] = payload
        return node_id

    def add_edge(source: str, target: str, label: str, relationship_type: str, **extra: Any) -> None:
        if not source or not target or source == target:
            return
        key = (source, target, label, relationship_type)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "label": label,
                "relationship_type": relationship_type,
                **{field: value for field, value in extra.items() if value not in (None, "", [])},
            }
        )

    question_id = add_node("question", question, "question")
    evidence_by_key = _evidence_ids_by_key(evidence)

    for index, row in enumerate(rows[:20], start=1):
        source_document = _clean(row.get("source_document")) or "unknown_source"
        source_group = _source_group(row)
        statement = _clean(row.get("statement_name")) or _clean(row.get("related_name")) or _clean(row.get("seed_name")) or f"Retrieved row {index}"
        evidence_text = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
        citation = _clean(row.get("citation")) or _clean(row.get("article_number")) or _clean(row.get("section_title"))
        statement_key = _clean(row.get("statement_key")) or statement
        evidence_id = evidence_by_key.get((statement, source_document, evidence_text))

        regulation_id = add_node(
            f"reg:{source_document}",
            source_document,
            "regulation",
            source_document=source_document,
            source_group=source_group,
        )
        statement_id = add_node(
            f"stmt:{_slug(statement_key)}",
            statement,
            "statement",
            source_document=source_document,
            citation=citation,
            route=row.get("query_route"),
            score=row.get("score"),
        )
        add_edge(question_id, statement_id, "ranked_for_question", "UI_RETRIEVAL", evidence_id=evidence_id)
        add_edge(statement_id, regulation_id, "supports_answer", "UI_RETRIEVAL", evidence_id=evidence_id)

        if evidence_text:
            evidence_node_id = add_node(
                f"evidence:{evidence_id or index}",
                citation or f"Evidence {evidence_id or index}",
                "evidence",
                source_document=source_document,
                citation=citation,
                evidence_id=evidence_id,
            )
            add_edge(statement_id, evidence_node_id, "retrieved_as_evidence", "UI_RETRIEVAL", evidence_id=evidence_id)
            add_edge(evidence_node_id, regulation_id, "source_document", "UI_RETRIEVAL", evidence_id=evidence_id)

        relationship = _clean(row.get("relationship"))
        concept_name = _clean(row.get("seed_name")) or _clean(row.get("related_name"))
        if concept_name and concept_name != statement:
            concept_labels = row.get("seed_labels") or row.get("related_labels") or []
            concept_type = _graph_node_type(concept_labels, row.get("node_type"))
            concept_id = add_node(
                f"entity:{_slug(concept_name)}",
                concept_name,
                concept_type,
                source_document=source_document,
            )
            if relationship:
                add_edge(concept_id, statement_id, relationship, relationship, evidence_id=evidence_id)
            else:
                add_edge(concept_id, statement_id, "RELATED_CONCEPT", "RELATED_CONCEPT", evidence_id=evidence_id)

        for group in _as_list(row.get("matched_concept_groups"))[:4]:
            concept_label = _clean(group).split(":", 1)[-1]
            if not concept_label:
                continue
            concept_id = add_node(f"concept:{_slug(concept_label)}", concept_label, "entity")
            add_edge(question_id, concept_id, "RELATED_CONCEPT", "UI_RETRIEVAL")
            add_edge(concept_id, statement_id, "ranked_for_question", "UI_RETRIEVAL", evidence_id=evidence_id)

    return {
        "nodes": list(nodes.values())[:40],
        "edges": edges[:80],
        "meta": {
            "source": "live_retrieval_rows",
            "ui_edge_types": ["ranked_for_question", "retrieved_as_evidence", "supports_answer"],
            "real_relationship_types": sorted(
                {
                    edge["relationship_type"]
                    for edge in edges
                    if edge.get("relationship_type") not in {"UI_RETRIEVAL"}
                }
            ),
        },
    }


def _build_no_evidence_graph(question: str) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": "question",
                "label": _graph_label(question) or "Question",
                "type": "question",
            }
        ],
        "edges": [],
        "meta": {
            "source": "no_evidence",
            "message": NO_EVIDENCE_MESSAGE,
        },
    }


def _evidence_ids_by_key(evidence: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
    return {
        (_clean(item.get("statement")), _clean(item.get("source_document")), _clean(item.get("evidence_text"))): str(item.get("id"))
        for item in evidence
    }


def _graph_label(value: Any) -> str:
    text = _clean(value)
    return text[:80]


def _graph_node_type(labels: Any, fallback: Any = None) -> str:
    label_set = {str(label).lower() for label in _as_list(labels)}
    fallback_text = _clean(fallback).lower()
    if "ontologyclass" in label_set or "ontology" in fallback_text:
        return "ontology"
    if "statement" in label_set:
        return "statement"
    if any(label in label_set for label in {"requirement", "obligation", "control"}):
        return "control"
    if "risk" in label_set or "risk" in fallback_text:
        return "risk"
    return "entity"


def _slug(value: Any) -> str:
    text = _clean(value).lower()
    cleaned = "".join(char if char.isalnum() else "-" for char in text)
    return "-".join(part for part in cleaned.split("-") if part)[:80] or "node"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


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
        "semantic_query_ms": stage_timings.get("semantic_query_ms", 0),
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


atexit.register(close_retriever)
