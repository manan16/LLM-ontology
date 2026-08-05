from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from graph.neo4j_client import Neo4jClient
from rag.query_planner import QueryPlan, build_query_plan, extract_query_terms
from rag.retriever.dedup import _dedupe_ids, _dedupe_values
from rag.retriever.scoring import DEFAULT_MIN_PRIMARY_ROWS, GDPR_RIGHTS_GROUPS, _canonical_source_document, _count_primary_source_rows, _coverage_sources_missing, _coverage_sources_present, _debug_row, _dedupe_rows, _entity_seed_priority, _has_inferred_single_target, _is_concrete_ai_obligation_query, _is_evidence_expansion_seed, _is_gdpr_data_subject_rights_query, _mental_health_source_terms, _query_requests_ai, _row_debug_id, _seed_expansion_terms, _select_final_rows, _selected_gdpr_rights_groups, _source_document_counts
from rag.retriever.routes import GRAPH_RETRIEVAL_CATEGORIES, MIN_ROUTE_LIMIT, ROUTE_LIMIT_MULTIPLIER, _ACTOR_TO_STATEMENT_QUERY, _AI_RISK_STATEMENT_QUERY, _CONCRETE_AI_OBLIGATION_QUERY, _ENTITY_EVIDENCE_EXPANSION_QUERY, _ENTITY_SECTION_EVIDENCE_EXPANSION_QUERY, _EXACT_PHRASE_ENTITY_QUERY, _FALLBACK_STATEMENT_QUERY, _PERMISSION_EXCEPTION_STATEMENT_QUERY, _STATEMENT_QUERY, _STATEMENT_TO_EVIDENCE_QUERY, _TARGET_SOURCE_STATEMENT_QUERY, _merge_plans, _query_parameters


class GraphRetriever:
    def __init__(self, neo4j_client: Neo4jClient | None = None, settings: Settings | None = None) -> None:
        self.neo4j_client = neo4j_client or Neo4jClient(settings or get_settings())
        self.last_query_plan: QueryPlan | None = None
        self.last_debug_rows: list[dict[str, Any]] = []
        self.last_expansion_debug: dict[str, list[dict[str, Any]]] = {
            "seeds": [],
            "expanded": [],
            "excluded": [],
            "selected": [],
            "diversity_groups": [],
        }
        self.last_selection_debug: dict[str, Any] = {}

    def retrieve(
        self,
        question: str,
        limit: int = 30,
        conversation_history: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        plan = build_query_plan(question, conversation_history)
        self.last_query_plan = plan
        self.last_debug_rows = []
        self.last_expansion_debug = {"seeds": [], "expanded": [], "excluded": [], "selected": [], "diversity_groups": []}
        self.last_selection_debug = {}

        if plan.category not in GRAPH_RETRIEVAL_CATEGORIES:
            return []

        retrieval_questions = plan.sub_questions or [plan.normalized_question]
        rows: list[dict[str, Any]] = []
        query_limit = max(limit * ROUTE_LIMIT_MULTIPLIER, MIN_ROUTE_LIMIT)
        for retrieval_question in retrieval_questions:
            sub_plan = build_query_plan(retrieval_question, conversation_history)
            merged_plan = _merge_plans(plan, sub_plan)
            if not merged_plan.terms and not merged_plan.phrases and not merged_plan.expansion_terms:
                continue
            rows.extend(self._run_graph_routes(merged_plan, query_limit))

        ranked_candidates = _dedupe_rows(rows, plan)
        expanded_rows = self._expand_missing_evidence_seeds(ranked_candidates, plan, query_limit)
        if expanded_rows:
            ranked_candidates = _dedupe_rows([*expanded_rows, *ranked_candidates], plan)

        ranked_rows, excluded_rows = _select_final_rows(ranked_candidates, plan, limit)
        for row in ranked_rows:
            row["_retriever_ranked"] = True
        broad_excluded = [row for row in excluded_rows if "broad_row_excluded_concrete_available" in row.get("penalties", [])]
        primary_rows_found = _count_primary_source_rows(ranked_candidates, plan)
        fallback_used = bool(plan.explicit_single_regulation and primary_rows_found < DEFAULT_MIN_PRIMARY_ROWS)
        wrong_source_excluded = [row for row in excluded_rows if "wrong_source_for_explicit_regulation" in row.get("penalties", [])]
        secondary_penalized = [row for row in ranked_candidates if "secondary_reference_for_explicit_regulation" in row.get("penalties", [])]
        self.last_selection_debug = {
            "broad_rows_excluded": bool(broad_excluded),
            "broad_rows_excluded_count": len(broad_excluded),
            "concrete_rows_selected": sum(1 for row in ranked_rows if row.get("concrete_ai_obligation")),
            "explicit_regulations": plan.explicit_regulations,
            "target_source_documents": plan.target_source_documents,
            "primary_source_rows_found": primary_rows_found,
            "fallback_used": fallback_used,
            "wrong_source_rows_excluded": len(wrong_source_excluded),
            "secondary_reference_rows_penalized": len(secondary_penalized),
            "final_selected_row_ids": [_row_debug_id(row) for row in ranked_rows],
            "diversity_groups_selected": _selected_gdpr_rights_groups(ranked_rows, plan),
            "per_source_retrieval_counts": _source_document_counts(ranked_candidates),
            "per_source_selected_counts": _source_document_counts(ranked_rows),
            "coverage_sources_present": _coverage_sources_present(ranked_rows, plan),
            "coverage_sources_missing": _coverage_sources_missing(ranked_rows, plan),
        }
        self.last_expansion_debug["diversity_groups"] = [
            {"group": group} for group in _selected_gdpr_rights_groups(ranked_rows, plan)
        ]
        self.last_expansion_debug["excluded"] = [_debug_row(row) for row in excluded_rows[:20]]
        self.last_expansion_debug["selected"] = [_debug_row(row) for row in ranked_rows[:10]]
        self.last_debug_rows = ranked_rows
        return ranked_rows

    def _run_graph_routes(self, plan: QueryPlan, limit: int) -> list[dict[str, Any]]:
        parameters = _query_parameters(plan, limit)
        rows: list[dict[str, Any]] = []
        if plan.cross_regulation and plan.target_source_documents:
            per_source_limit = max(5, min(limit, MIN_ROUTE_LIMIT // max(len(plan.target_source_documents), 1)))
            for source_document in plan.target_source_documents:
                source_terms = _mental_health_source_terms(plan, source_document) if getattr(plan, "is_mental_health_query", False) else []
                if source_terms:
                    rows.extend(
                        self._run_route(
                            f"mental_health_source_statement:{_canonical_source_document(source_document)}",
                            _TARGET_SOURCE_STATEMENT_QUERY,
                            {
                                **parameters,
                                "content_terms": source_terms,
                                "target_source_documents": [_canonical_source_document(source_document)],
                                "limit": per_source_limit,
                            },
                        )
                    )
                rows.extend(
                    self._run_route(
                        "target_source_statement",
                        _TARGET_SOURCE_STATEMENT_QUERY,
                        {
                            **parameters,
                            "target_source_documents": [_canonical_source_document(source_document)],
                            "limit": per_source_limit,
                        },
                    )
                )
        if _is_gdpr_data_subject_rights_query(plan) and plan.target_source_documents:
            for group, terms in GDPR_RIGHTS_GROUPS.items():
                rows.extend(
                    self._run_route(
                        f"gdpr_rights_statement:{group}",
                        _TARGET_SOURCE_STATEMENT_QUERY,
                        {
                            **parameters,
                            "content_terms": list(terms),
                            "limit": max(10, limit),
                        },
                    )
                )
        routes = [
            ("exact_phrase_entity", _EXACT_PHRASE_ENTITY_QUERY),
            ("statement", _STATEMENT_QUERY),
            ("actor_to_statement", _ACTOR_TO_STATEMENT_QUERY),
            ("statement_to_evidence", _STATEMENT_TO_EVIDENCE_QUERY),
        ]
        if (plan.explicit_single_regulation or _has_inferred_single_target(plan)) and plan.target_source_documents:
            routes.insert(0, ("target_source_statement", _TARGET_SOURCE_STATEMENT_QUERY))
        if plan.intent in {"exceptions", "permissions"}:
            routes.insert(1, ("permission_exception_statement", _PERMISSION_EXCEPTION_STATEMENT_QUERY))
        if _query_requests_ai(plan) and plan.concept_groups.get("topic"):
            routes.insert(2, ("ai_risk_statement", _AI_RISK_STATEMENT_QUERY))
        if _is_concrete_ai_obligation_query(plan):
            routes.insert(2, ("concrete_ai_obligation", _CONCRETE_AI_OBLIGATION_QUERY))
        for route_name, query in routes:
            rows.extend(self._run_route(route_name, query, parameters))

        if len(rows) < max(3, limit // 4):
            rows.extend(self._run_route("fallback_broad_statement", _FALLBACK_STATEMENT_QUERY, parameters))
        return rows

    def _run_route(self, route_name: str, query: str, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        route_rows = self.neo4j_client.run_query(query, parameters)
        for row in route_rows:
            row.setdefault("query_route", route_name)
        return route_rows

    def _expand_missing_evidence_seeds(
        self,
        ranked_rows: list[dict[str, Any]],
        plan: QueryPlan,
        limit: int,
    ) -> list[dict[str, Any]]:
        seeds = [row for row in ranked_rows if _is_evidence_expansion_seed(row, plan)]
        seeds.sort(key=lambda row: _entity_seed_priority(row, plan))
        seeds = seeds[:40]
        self.last_expansion_debug["seeds"] = [_debug_row(row) for row in seeds]
        if not seeds:
            return []

        parameters = {
            **_query_parameters(plan, limit),
            "seed_ids": _dedupe_ids([row.get("seed_id") for row in seeds]),
            "seed_names": _dedupe_values(
                [
                    str(value)
                    for row in seeds
                    for value in (row.get("seed_name"), row.get("related_name"), row.get("statement_name"))
                    if value and value != "Unknown"
                ]
            ),
            "seed_keys": _dedupe_values(
                [
                    str(value)
                    for row in seeds
                    for value in (row.get("seed_key"), row.get("related_key"), row.get("statement_key"))
                    if value
                ]
            ),
            "seed_terms": _seed_expansion_terms(seeds, plan),
        }
        if not parameters["seed_ids"] and not parameters["seed_names"] and not parameters["seed_keys"]:
            return []

        expanded = self._run_route("entity_evidence_expansion", _ENTITY_EVIDENCE_EXPANSION_QUERY, parameters)
        expanded.extend(self._run_route("entity_section_evidence_expansion", _ENTITY_SECTION_EVIDENCE_EXPANSION_QUERY, parameters))
        for row in expanded:
            row.setdefault("score", 24)
            row["expanded_from_entity_seed"] = True
        ranked_expanded = _dedupe_rows(expanded, plan)
        self.last_expansion_debug["expanded"] = [_debug_row(row) for row in ranked_expanded[:20]]
        return ranked_expanded


def expand_chunk_entities(
    chunk_ids: list[str],
    max_hops: int = 1,
    neo4j_client: Neo4jClient | None = None,
) -> dict[str, dict[str, Any]]:
    """Expand semantically-retrieved chunk IDs into linked entities and relationship triples.

    This is the retrieval-layer entry point that pairs with SemanticRetriever's
    vector search: pass the chunk IDs it returns to enrich each chunk with the
    graph entities it mentions and their immediate (subject, predicate, object)
    relationships. See Neo4jClient.expand_chunk_entities for the traversal.
    """
    client = neo4j_client or Neo4jClient(get_settings())
    return client.expand_chunk_entities(chunk_ids, max_hops=max_hops)
