from __future__ import annotations

import re
from typing import Any

from app.config import Settings, get_settings
from graph.neo4j_client import Neo4jClient
from rag.context_builder import is_suppressed_context_row
from rag.query_planner import PERMISSION_EXCEPTION_EXPANSIONS, QueryPlan, build_query_plan, extract_query_terms


GRAPH_RETRIEVAL_CATEGORIES = {
    "compliance_question",
    "graph_lookup_question",
    "follow_up_question",
    "evidence_question",
}

GRAPH_EXPANSION_RELATIONSHIPS = [
    "APPLIES_TO",
    "RELATED_TO",
    "HAS_REQUIREMENT",
    "IMPOSES_ON",
    "GRANTS_PERMISSION",
    "PROHIBITS",
    "HAS_EXCEPTION",
    "REQUIRES_AUTHORIZATION",
    "REQUIRES_DOCUMENTATION",
    "REQUIRES_ASSESSMENT",
    "REQUIRES_MONITORING",
    "REQUIRES_HUMAN_OVERSIGHT",
    "LIMITED_BY",
    "DISCLOSES_TO",
    "USES_FOR",
    "HAS_CONDITION",
    "REQUIRES_CONTROL",
    "MITIGATES_RISK",
    "HAS_CONTROL",
    "HAS_SAFEGUARD",
    "NOTIFIES",
]

EVIDENCE_RELATIONSHIPS = ["CITES", "REFERENCES", "CONTAINS", "HAS_SECTION", "DERIVED_FROM"]

MIN_ROUTE_LIMIT = 100
ROUTE_LIMIT_MULTIPLIER = 5
DEFAULT_MIN_PRIMARY_ROWS = 3

PERMISSION_EXCEPTION_SEARCH_TERMS = [
    "permitted",
    "may disclose",
    "may use",
    "use or disclose",
    "authorization not required",
    "without authorization",
    "opportunity to agree or object is not required",
    "required by law",
    "public health",
    "health oversight",
    "judicial",
    "law enforcement",
    "research",
    "serious threat",
    "workers compensation",
]

NEGATED_AUTHORIZATION_EVIDENCE_TERMS = [
    "not required",
    "without authorization",
    "authorization is not required",
    "no authorization required",
    "permitted",
    "may disclose",
    "may use or disclose",
    "authorization or opportunity to agree or object is not required",
    "except",
]

STRICT_NEGATED_AUTHORIZATION_TERMS = [
    "without authorization",
    "not required",
    "authorization is not required",
    "no authorization required",
    "authorization or opportunity to agree or object is not required",
    "not require authorization",
]

AUTHORIZATION_REQUIRED_TERMS = [
    "pursuant to and in compliance with an authorization",
    "authorization is required",
    "authorization required",
    "must obtain authorization",
    "requires authorization",
    "require authorization",
    "with authorization",
    "with a valid authorization",
    "pursuant to authorization",
]

CONCRETE_AI_OBLIGATION_TERMS = (
    "risk management system",
    "risk management process",
    "continuous iterative process",
    "lifecycle of a high risk ai system",
    "identifying and mitigating risks",
    "risk mitigation",
    "residual risk",
    "known and foreseeable risks",
    "risk management measures",
    "document and explain the choices",
    "technical documentation",
    "technical documentation update",
    "record keeping",
    "logs",
    "traceability",
    "instructions for use",
    "human oversight",
    "human oversight measures",
    "natural persons can oversee",
    "operational constraints",
    "human operator",
    "competence training and authority",
    "cybersecurity",
    "cyber resilience",
    "security controls",
    "data poisoning",
    "adversarial attacks",
    "robustness",
    "accuracy",
    "conformity assessment",
    "conformity assessment prior to market placement",
    "provider responsibility for conformity assessment",
    "quality management system",
    "post market monitoring",
    "serious incident",
    "corrective action",
    "fundamental rights impact assessment",
)

AI_OBLIGATION_TOPIC_GROUPS = {
    "documentation": ("technical documentation", "record keeping", "logs", "traceability", "instructions for use"),
    "risk_management": (
        "risk management",
        "continuous iterative process",
        "lifecycle",
        "risk mitigation",
        "residual risk",
        "known and foreseeable risks",
    ),
    "conformity_assessment": ("conformity assessment", "prior to market placement"),
    "monitoring": ("monitoring", "post market monitoring", "serious incident", "corrective action"),
    "cybersecurity": ("cybersecurity", "cyber resilience", "security controls", "data poisoning", "adversarial attacks", "robustness"),
    "human_oversight": ("human oversight", "natural persons can oversee", "human operator", "operational constraints"),
    "quality_management": ("quality management", "quality management system"),
}

BROAD_AI_STATEMENT_TERMS = (
    "high risk ai system rules",
    "obligations of providers and deployers",
    "ai risk generation",
    "ai alignment with union values",
    "exercise of data subject rights",
    "common rules",
    "regulation aims",
    "regulation purpose",
    "risk based approach",
    "union values",
    "social policy",
    "consumer rights",
    "data subject rights",
    "definition of ai system",
    "ai system characteristics",
    "methodology for identifying high risk ai systems",
    "promote ai literacy",
    "ai literacy requirement",
    "extraterritorial application",
    "application to union institutions",
    "harmonized rules application",
)

NON_OPERATIONAL_AI_STATEMENT_TERMS = (
    "voluntariness",
    "presumption of compliance",
    "classification",
    "definition",
    "exception",
    "exclusion",
    "methodology for identifying",
    "commission adopts",
)

GDPR_DATA_SUBJECT_RIGHTS_TERMS = (
    "right of access",
    "right to rectification",
    "right to erasure",
    "right to restriction",
    "right to data portability",
    "right to object",
    "automated decision making",
    "automated decision-making",
    "human intervention",
    "meaningful information",
    "profiling",
    "legal effects",
    "restriction of processing",
    "restrict processing",
    "transmit those data",
    "obtain confirmation",
    "inaccurate personal data",
)

GDPR_RIGHTS_GROUPS = {
    "access": ("right of access", "access", "obtain confirmation"),
    "rectification": ("right to rectification", "rectification", "inaccurate personal data"),
    "erasure": ("right to erasure", "erasure", "right to be forgotten"),
    "restriction": ("right to restriction", "restriction of processing", "restrict processing"),
    "portability": ("right to data portability", "data portability", "transmit those data"),
    "object": ("right to object", "object to processing"),
    "automated_decision_making": (
        "automated decision making",
        "automated decision-making",
        "profiling",
        "human intervention",
        "meaningful information",
        "legal effects",
    ),
}

GDPR_GENERIC_RIGHTS_NOISE_TERMS = (
    "personal data breach",
    "notify data subject of breach",
    "supervisory authority of breach",
    "third country",
    "data transfer",
    "binding corporate rules",
    "standard data protection clauses",
)


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


def _dedupe_rows(rows: list[dict[str, Any]], plan: QueryPlan | None = None) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        compact = _compact_row(row)
        statement_key = compact.get("statement_key") or compact.get("statement_name")
        if statement_key:
            key = ("statement", statement_key)
        elif compact.get("statement_name") or compact.get("evidence_text"):
            key = ("statement-evidence", compact.get("statement_name"), compact.get("evidence_text"))
        else:
            key = (
                "path",
                compact.get("seed_name"),
                compact.get("relationship"),
                compact.get("related_name"),
                compact.get("evidence_text"),
                compact.get("citation"),
            )
        if key in seen:
            continue
        seen.add(key)
        compact["score"] = _score_compact_row(compact, plan)
        unique.append(compact)
    unique.sort(key=lambda item: (-float(item.get("score") or 0), _score_row(item)))
    return unique


def _select_final_rows(
    rows: list[dict[str, Any]],
    plan: QueryPlan | None,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plan:
        return rows[:limit], []

    excluded: list[dict[str, Any]] = []
    evidence_rows = [row for row in rows if _has_evidence(row)]
    ai_evidence_rows = [row for row in evidence_rows if _row_matches_ai_system(row)]
    use_ai_filter = _query_requests_ai(plan) and not _query_requests_hipaa(plan) and bool(ai_evidence_rows)
    concrete_rows = [row for row in evidence_rows if row.get("concrete_ai_obligation")]
    exclude_broad = _is_concrete_ai_obligation_query(plan) and len(concrete_rows) >= min(limit, 6)
    final_pool: list[dict[str, Any]] = []

    for row in rows:
        if _is_wrong_source_missing_entity_seed(row, plan):
            row["penalties"] = _dedupe_debug([*(row.get("penalties") or []), "wrong_source_entity_seed"])
            excluded.append(row)
            continue
        if evidence_rows and _is_missing_evidence_entity_row(row):
            excluded.append(row)
            continue
        if use_ai_filter and _is_unrelated_hipaa_row_for_ai_query(row):
            excluded.append(row)
            continue
        if use_ai_filter and _has_only_weak_ai_concept_coverage(row):
            excluded.append(row)
            continue
        if exclude_broad and row.get("broad_ai_statement"):
            row["penalties"] = _dedupe_debug([*(row.get("penalties") or []), "broad_row_excluded_concrete_available"])
            excluded.append(row)
            continue
        final_pool.append(row)

    if not final_pool:
        if excluded and all("wrong_source_entity_seed" in (row.get("penalties") or []) for row in excluded):
            return [], excluded
        final_pool = rows
        excluded = []
    if _is_concrete_ai_obligation_query(plan):
        final_pool = _prioritize_ai_topic_diversity(final_pool)
    final_pool, regulation_excluded = _apply_regulation_selection(final_pool, plan, limit)
    excluded.extend(regulation_excluded)
    if _is_gdpr_data_subject_rights_query(plan):
        final_pool = _prioritize_gdpr_rights_diversity(final_pool)
    return final_pool[:limit], excluded


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": row.get("score"),
        "query_route": row.get("query_route"),
        "seed_id": row.get("seed_id"),
        "seed_key": row.get("seed_key"),
        "seed_name": _first_present(row, "seed_name", "seed_key", "seed_id"),
        "seed_labels": row.get("seed_labels") or [],
        "relationship": row.get("relationship"),
        "related_key": row.get("related_key"),
        "related_name": _first_present(row, "related_name", "related_key", "related_id"),
        "related_labels": row.get("related_labels") or [],
        "statement_key": row.get("statement_key"),
        "statement_name": _first_present(row, "statement_name", "statement_id"),
        "statement_labels": row.get("statement_labels") or [],
        "evidence_text": _first_present(row, "evidence_text", "relationship_evidence", "seed_evidence_text", "related_evidence_text"),
        "citation": _first_present(row, "citation"),
        "section_title": _first_present(row, "section_title", "chunk_section_title"),
        "page_number": _first_present(row, "page_number"),
        "article_number": _first_present(row, "article_number"),
        "clause_number": _first_present(row, "clause_number"),
        "source_document": _first_present(row, "source_document", "relationship_source_document"),
        "source_chunk_text": _first_present(row, "source_chunk_text"),
        "node_type": _first_present(row, "node_type", "seed_node_type", "related_node_type"),
        "aliases": row.get("aliases") or row.get("seed_aliases") or row.get("related_aliases") or [],
        "descriptions": row.get("descriptions") or row.get("seed_descriptions") or row.get("related_descriptions") or [],
        "original_entity_name": _first_present(row, "original_entity_name"),
        "relationship_path": _first_present(row, "relationship_path"),
        "expanded_from_entity_seed": bool(row.get("expanded_from_entity_seed")),
        "expanded_from_seed": _first_present(row, "expanded_from_seed", "original_entity_name"),
        "expansion_path": _first_present(row, "expansion_path", "relationship_path"),
        "seed_matched_terms": row.get("seed_matched_terms") or [],
    }


def _first_present(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            if value:
                return value[0] if len(value) == 1 else value
        elif value not in (None, ""):
            return value
    return None


def _score_row(row: dict[str, Any]) -> tuple[int, int, str]:
    has_evidence = 0 if row.get("evidence_text") or row.get("source_chunk_text") else 1
    has_statement = 0 if row.get("statement_name") else 1
    return (has_evidence, has_statement, str(row.get("statement_name") or row.get("seed_name") or ""))


def _score_compact_row(row: dict[str, Any], plan: QueryPlan | None) -> float:
    score = float(row.get("score") or 0)
    if not plan:
        return score

    searchable_text = _normalize_search_text(_row_searchable_text(row))
    haystacks = [
        _normalize_search_text(str(row.get("seed_name") or "")),
        _normalize_search_text(str(row.get("related_name") or "")),
        _normalize_search_text(str(row.get("statement_name") or "")),
        _normalize_search_text(str(row.get("evidence_text") or "")),
        _normalize_search_text(str(row.get("source_chunk_text") or "")),
        _normalize_search_text(str(row.get("citation") or "")),
        _normalize_search_text(str(row.get("article_number") or "")),
        _normalize_search_text(str(row.get("section_title") or "")),
    ]
    labels = set(row.get("statement_labels") or []) | set(row.get("seed_labels") or []) | set(row.get("related_labels") or [])
    relationship = row.get("relationship")
    debug = _new_score_debug()

    for phrase in plan.phrases:
        normalized_phrase = _normalize_search_text(phrase)
        if normalized_phrase in haystacks[0] or normalized_phrase in haystacks[1] or normalized_phrase in haystacks[2]:
            score += 8
            debug["matched_terms"].append(phrase)
        if any(normalized_phrase in haystack for haystack in haystacks[3:]):
            score += 4
            debug["matched_terms"].append(phrase)
    for term in plan.expansion_terms:
        normalized_term = _normalize_search_text(term)
        if not normalized_term:
            continue
        if normalized_term in haystacks[2]:
            score += 7
            debug["matched_terms"].append(term)
        elif any(normalized_term in haystack for haystack in haystacks[3:]):
            score += 5
            debug["matched_terms"].append(term)
        elif normalized_term in haystacks[0] or normalized_term in haystacks[1]:
            score += 3
            debug["matched_terms"].append(term)
    matched_terms = sum(1 for term in plan.terms if _normalize_search_text(term) in searchable_text)
    score += min(matched_terms, 8) * 1.5
    if labels.intersection(plan.preferred_statement_labels):
        score += 6
    if relationship in plan.preferred_relationships:
        score += 6
    if row.get("evidence_text") or row.get("source_chunk_text"):
        score += 8
    if row.get("citation"):
        score += 4
    if row.get("section_title") or row.get("page_number") or row.get("article_number") or row.get("source_document"):
        score += 3
    if set(plan.detected_actors).intersection({haystacks[0], haystacks[1]}):
        score += 5
    if row.get("query_route") == "exact_phrase_entity":
        score += 5
    if row.get("expanded_from_entity_seed"):
        seed_boost = 35 if _is_gdpr_data_subject_rights_query(plan) else 12
        score += seed_boost
        debug["boosts"].append("expanded_from_entity_seed")
        expanded_from_seed = str(row.get("expanded_from_seed") or row.get("seed_name") or "")
        if _is_gdpr_data_subject_rights_query(plan) and _infer_gdpr_rights_group(expanded_from_seed):
            score += 20
            debug["boosts"].append("important_exact_seed")
    if _is_phi_disclosure_plan(plan) and is_suppressed_context_row(row):
        score -= 120
        debug["penalties"].append("employment_record_noise")
    concept_score = _score_concept_coverage(row, plan, searchable_text, labels, relationship, debug)
    score += concept_score
    score += _score_mental_health_context(row, plan, searchable_text, debug)
    score += _score_gdpr_data_subject_rights(row, plan, debug)
    score += _score_regulation_source(row, plan, debug)
    concrete_score = _score_concrete_ai_obligation(row, plan, searchable_text, debug)
    score += concrete_score
    if plan.has_negated_authorization:
        score += _score_negated_authorization_row(row, labels, relationship, haystacks, plan)
    row["matched_concept_groups"] = _dedupe_debug(debug["matched_concept_groups"])
    row["matched_terms"] = _dedupe_debug(debug["matched_terms"])
    row["penalties"] = _dedupe_debug(debug["penalties"])
    row["boosts"] = _dedupe_debug(debug["boosts"])
    row["matched_topic_groups"] = _dedupe_debug(debug["matched_topic_groups"])
    row["concrete_ai_obligation"] = "concrete_obligation" in row["boosts"]
    row["broad_ai_statement"] = "broad_preamble" in row["penalties"]
    row["source_group"] = _infer_source_group(row, searchable_text)
    return round(score, 2)


def _new_score_debug() -> dict[str, list[str]]:
    return {"matched_concept_groups": [], "matched_terms": [], "matched_topic_groups": [], "boosts": [], "penalties": []}


def _score_concrete_ai_obligation(
    row: dict[str, Any],
    plan: QueryPlan,
    text: str,
    debug: dict[str, list[str]],
) -> float:
    if not _is_concrete_ai_obligation_query(plan):
        return 0.0

    score = 0.0
    direct_text = _normalize_search_text(_row_direct_obligation_text(row))
    concrete_matches = [term for term in CONCRETE_AI_OBLIGATION_TERMS if _concept_matches(direct_text, term)]
    requested_topics = _requested_ai_topic_groups(plan)
    matched_topics = [
        group
        for group, variants in AI_OBLIGATION_TOPIC_GROUPS.items()
        if any(_concept_matches(direct_text, variant) for variant in variants)
    ]
    matched_requested_topics = [group for group in matched_topics if not requested_topics or group in requested_topics]
    is_operational = _is_operational_ai_obligation_row(row, direct_text, concrete_matches)

    if concrete_matches and is_operational:
        score += 24 + min(len(concrete_matches), 5) * 4
        debug["boosts"].append("concrete_obligation")
        debug["matched_terms"].extend(concrete_matches)
    if matched_topics and is_operational:
        debug["matched_topic_groups"].extend(matched_topics)
    if matched_requested_topics and is_operational:
        score += len(matched_requested_topics) * 12
        debug["boosts"].append("requested_topic_coverage")
    if _row_matches_ai_system(row) and matched_requested_topics and is_operational:
        score += 12
        debug["boosts"].append("ai_object_plus_topic")
    if row.get("query_route") == "concrete_ai_obligation" and is_operational:
        score += 12
        debug["boosts"].append("concrete_ai_obligation_route")
    if "Definition" in set(row.get("statement_labels") or []) and not is_operational:
        score -= 30
        debug["penalties"].append("non_operational_definition")

    broad_matches = [term for term in BROAD_AI_STATEMENT_TERMS if _concept_matches(text, term)]
    if broad_matches and not (concrete_matches and is_operational):
        score -= 38
        debug["penalties"].append("broad_preamble")
        debug["matched_terms"].extend(broad_matches)
    return score


def _row_direct_obligation_text(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("seed_name") or ""),
        str(row.get("expanded_from_seed") or ""),
        str(row.get("related_name") or ""),
        str(row.get("statement_name") or ""),
        str(row.get("evidence_text") or ""),
    ]
    for key in ("aliases", "descriptions"):
        value = row.get(key)
        pieces.extend(str(item) for item in value) if isinstance(value, list) else pieces.append(str(value or ""))
    if not row.get("evidence_text"):
        pieces.append(str(row.get("source_chunk_text") or ""))
    return " ".join(pieces)


def _is_operational_ai_obligation_row(row: dict[str, Any], text: str, concrete_matches: list[str]) -> bool:
    if any(term in text for term in NON_OPERATIONAL_AI_STATEMENT_TERMS):
        return False
    labels = set(row.get("statement_labels") or [])
    if labels.intersection({"Obligation", "Requirement", "Control"}):
        return True
    if any(
        marker in text
        for marker in (
            "must ",
            "shall ",
            "required",
            "obliged",
            "establish",
            "maintain",
            "keep ",
            "carry out",
            "draw up",
            "ensure",
            "take ",
            "document",
            "update",
            "review",
            "identify",
            "mitigate",
            "implement",
            "provide",
        )
    ):
        return True
    statement_name = _normalize_search_text(str(row.get("statement_name") or ""))
    return any(_concept_matches(statement_name, term) for term in concrete_matches)


def _prioritize_ai_topic_diversity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for topic_group, variants in AI_OBLIGATION_TOPIC_GROUPS.items():
        candidates = [
            row
            for row in rows
            if id(row) not in selected_ids
            and topic_group in (row.get("matched_topic_groups") or [])
            and row.get("concrete_ai_obligation")
        ]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                0
                if any(
                    _concept_matches(_normalize_search_text(str(row.get("statement_name") or "")), variant)
                    for variant in variants
                )
                else 1,
                -float(row.get("score") or 0),
            )
        )
        selected = candidates[0]
        priority.append(selected)
        selected_ids.add(id(selected))
    return [*priority, *(row for row in rows if id(row) not in selected_ids)]


def _prioritize_gdpr_rights_diversity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for group in GDPR_RIGHTS_GROUPS:
        candidates = [
            row
            for row in rows
            if id(row) not in selected_ids and _infer_gdpr_rights_group(_row_direct_obligation_text(row)) == group
        ]
        if not candidates:
            continue
        candidates.sort(key=lambda row: (0 if row.get("expanded_from_entity_seed") else 1, -float(row.get("score") or 0)))
        selected = candidates[0]
        priority.append(selected)
        selected_ids.add(id(selected))
    return [*priority, *(row for row in rows if id(row) not in selected_ids)]


def _selected_gdpr_rights_groups(rows: list[dict[str, Any]], plan: QueryPlan | None) -> list[str]:
    if not plan or not _is_gdpr_data_subject_rights_query(plan):
        return []
    groups: list[str] = []
    seen: set[str] = set()
    for row in rows:
        group = _infer_gdpr_rights_group(_row_direct_obligation_text(row))
        if group and group not in seen:
            seen.add(group)
            groups.append(group)
    return groups


def _requested_ai_topic_groups(plan: QueryPlan) -> set[str]:
    question = _normalize_search_text(plan.normalized_question)
    return {
        group
        for group, variants in AI_OBLIGATION_TOPIC_GROUPS.items()
        if _concept_matches(question, group.replace("_", " "))
        or any(_concept_matches(question, variant) for variant in variants)
    }


def _score_regulation_source(
    row: dict[str, Any],
    plan: QueryPlan,
    debug: dict[str, list[str]],
) -> float:
    if not plan.target_source_documents:
        return 0.0
    source_document = _canonical_source_document(row.get("source_document"))
    target_sources = {_canonical_source_document(source) for source in plan.target_source_documents}
    if plan.explicit_single_regulation:
        if source_document in target_sources:
            debug["boosts"].append("primary_source_document")
            return 40.0
        debug["penalties"].append("secondary_reference_for_explicit_regulation")
        return -50.0
    if plan.cross_regulation and source_document in target_sources:
        debug["boosts"].append("requested_regulation_source")
        return 18.0
    if source_document in target_sources and getattr(plan, "inferred_regulations", []):
        debug["boosts"].append("inferred_regulation_source")
        return 24.0
    if getattr(plan, "inferred_regulations", []) and source_document and source_document not in target_sources:
        debug["penalties"].append("outside_inferred_regulation_scope")
        return -20.0
    return 0.0


def _apply_regulation_selection(
    rows: list[dict[str, Any]],
    plan: QueryPlan,
    limit: int,
    min_primary_rows: int = DEFAULT_MIN_PRIMARY_ROWS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not plan.target_source_documents:
        return rows, []

    target_sources = [_canonical_source_document(source) for source in plan.target_source_documents]

    if plan.explicit_single_regulation:
        primary_rows = [row for row in rows if _canonical_source_document(row.get("source_document")) in target_sources]
        if len(primary_rows) >= min_primary_rows:
            excluded: list[dict[str, Any]] = []
            for row in rows:
                if _canonical_source_document(row.get("source_document")) in target_sources:
                    continue
                row["penalties"] = _dedupe_debug([*(row.get("penalties") or []), "wrong_source_for_explicit_regulation"])
                excluded.append(row)
            return primary_rows, excluded
        return rows, []

    if plan.cross_regulation and len(target_sources) > 1:
        return _select_with_source_coverage(rows, target_sources, limit), []

    if getattr(plan, "inferred_regulations", []) and target_sources:
        target_rows = [row for row in rows if _canonical_source_document(row.get("source_document")) in target_sources]
        if target_rows:
            excluded = [
                row
                for row in rows
                if _canonical_source_document(row.get("source_document")) not in target_sources
            ]
            return target_rows, excluded

    return rows, []


def _select_with_source_coverage(rows: list[dict[str, Any]], target_sources: list[str], limit: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    per_source_quota = max(1, min(3, limit // max(len(target_sources), 1)))
    for source in target_sources:
        candidates = [
            row
            for row in rows
            if id(row) not in selected_ids and _canonical_source_document(row.get("source_document")) == source
        ][:per_source_quota]
        for row in candidates:
            selected.append(row)
            selected_ids.add(id(row))
    for row in rows:
        if len(selected) >= limit:
            break
        if id(row) not in selected_ids and _canonical_source_document(row.get("source_document")) in target_sources:
            selected.append(row)
            selected_ids.add(id(row))
    for row in rows:
        if len(selected) >= limit:
            break
        if id(row) not in selected_ids:
            selected.append(row)
            selected_ids.add(id(row))
    return selected


def _count_primary_source_rows(rows: list[dict[str, Any]], plan: QueryPlan) -> int:
    if not plan.explicit_single_regulation or not plan.target_source_documents:
        return 0
    target_sources = {_canonical_source_document(source) for source in plan.target_source_documents}
    return sum(1 for row in rows if _canonical_source_document(row.get("source_document")) in target_sources and _has_evidence(row))


def _source_document_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        source = _canonical_source_document(row.get("source_document")) or "unknown"
        counts[source] = counts.get(source, 0) + 1
    return counts


def _coverage_sources_present(rows: list[dict[str, Any]], plan: QueryPlan) -> list[str]:
    target_sources = [_canonical_source_document(source) for source in plan.target_source_documents]
    present = {_canonical_source_document(row.get("source_document")) for row in rows}
    return [source for source in target_sources if source in present]


def _coverage_sources_missing(rows: list[dict[str, Any]], plan: QueryPlan) -> list[str]:
    target_sources = [_canonical_source_document(source) for source in plan.target_source_documents]
    present = {_canonical_source_document(row.get("source_document")) for row in rows}
    return [source for source in target_sources if source not in present]


def _has_inferred_single_target(plan: QueryPlan) -> bool:
    return bool(getattr(plan, "inferred_regulations", [])) and not plan.cross_regulation and len(plan.target_source_documents) == 1


def _canonical_source_document(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("\\", "/").rsplit("/", 1)[-1]
    text = text.replace("-", "_")
    if text in {"gdpr", "gdpr.pdf"} or "general_data_protection_regulation" in text:
        return "gdpr.pdf"
    if text in {"hipaa", "hipaa.pdf"}:
        return "hipaa.pdf"
    if text in {"eu_ai_act", "eu_ai_act.pdf", "eu ai act", "eu ai act.pdf"} or "2024_1689" in text:
        return "eu_ai_act.pdf"
    return text


def _score_concept_coverage(
    row: dict[str, Any],
    plan: QueryPlan,
    text: str,
    labels: set[str],
    relationship: Any,
    debug: dict[str, list[str]],
) -> float:
    score = 0.0
    matched_groups = 0

    for group, variants in plan.concept_groups.items():
        matches = [variant for variant in variants if _concept_matches(text, variant)]
        if matches:
            matched_groups += 1
            debug["matched_concept_groups"].append(f"{group}:{matches[0]}")
            debug["matched_terms"].extend(matches)
    if matched_groups:
        score += matched_groups * 5
    if plan.required_concepts and matched_groups >= len(plan.required_concepts):
        score += 10

    has_high_risk = _concept_matches(text, "high-risk ai system") or _concept_matches(text, "high risk ai system")
    has_ai_system = _concept_matches(text, "ai system")
    has_ai_context = has_high_risk or has_ai_system
    query_requests_ai = _query_requests_ai(plan)
    query_requests_hipaa = any(
        _normalize_search_text(value) in {"hipaa", "protected health information", "covered entity"}
        for value in plan.concept_groups.get("domain", [])
    ) or "hipaa" in plan.normalized_question
    query_requests_gdpr = _query_requests_gdpr(plan)
    has_provider = _concept_matches(text, "provider")
    has_deployer = _concept_matches(text, "deployer")

    if query_requests_ai and has_high_risk:
        score += 8
    if query_requests_ai and has_ai_system:
        score += 6
    if query_requests_ai and has_provider and has_ai_context:
        score += 5
    if query_requests_ai and has_deployer and has_ai_context:
        score += 5
    if has_provider and not has_ai_context and "provider" in plan.ambiguous_terms:
        score -= 5
        debug["penalties"].append("generic_provider_only")
    if query_requests_hipaa and _infer_source_group(row, text) == "hipaa":
        score += 10
    if query_requests_hipaa and _infer_source_group(row, text) == "eu_ai_act" and not _has_hipaa_noise(text):
        score -= 15
        debug["penalties"].append("wrong_regulation_for_hipaa_query")
    if query_requests_ai and not has_ai_context and _has_modality_or_topic_only_match(debug):
        score -= 14
        debug["penalties"].append("missing_ai_system")
    if query_requests_ai and not query_requests_hipaa and _has_hipaa_noise(text) and not has_ai_context:
        score -= 12
        debug["penalties"].append("unrelated_hipaa_for_ai_query")
    if query_requests_gdpr and _infer_source_group(row, text) == "gdpr":
        score += 12
    if query_requests_gdpr and any(
        _concept_matches(text, term)
        for term in (
            "controller",
            "processor",
            "data subject",
            "personal data",
            "health data",
            "special category data",
            "article 9",
            "lawful basis",
            "explicit consent",
            "data protection impact assessment",
            "personal data breach",
            "right to erasure",
        )
    ):
        score += 8

    if plan.intent == "obligations":
        if labels.intersection({"Obligation", "Requirement"}):
            score += 4
        if "Definition" in labels and not labels.intersection({"Obligation", "Requirement"}):
            score -= 4
            debug["penalties"].append("definition_for_obligation_query")

    for term in (
        "technical documentation",
        "risk management",
        "risk management system",
        "risk mitigation",
        "residual risk",
        "known and foreseeable risks",
        "testing",
        "validation",
        "accuracy",
        "robustness",
        "cybersecurity",
        "human oversight",
        "post-market monitoring",
        "conformity assessment",
        "control measures",
        "safeguard",
        "safeguards",
    ):
        if _concept_matches(text, term):
            score += 4
            debug["matched_terms"].append(term)

    if any(_concept_matches(text, term) for term in ("eu_ai_act", "eu ai act", "regulation eu 2024 1689", "this regulation")):
        score += 3
    if any(_concept_matches(text, term) for term in ("gdpr", "general data protection regulation")):
        score += 3
    if relationship in plan.preferred_relationships:
        score += 3
    if row.get("evidence_text"):
        score += 2
    if row.get("source_document") or row.get("section_title") or row.get("article_number"):
        score += 2
    if not row.get("evidence_text") and not row.get("source_chunk_text"):
        score -= 3
        debug["penalties"].append("missing_evidence")

    if query_requests_ai and not query_requests_hipaa and _has_hipaa_noise(text) and not has_ai_context:
        score -= 8
        debug["penalties"].append("hipaa_noise")
        if any("object:" in item for item in debug["matched_concept_groups"]):
            pass
        else:
            debug["penalties"].append("missing_ai_system")

    return score


def _score_mental_health_context(
    row: dict[str, Any],
    plan: QueryPlan,
    text: str,
    debug: dict[str, list[str]],
) -> float:
    if not getattr(plan, "is_mental_health_query", False):
        return 0.0

    source_document = _canonical_source_document(row.get("source_document"))
    target_sources = {_canonical_source_document(source) for source in plan.target_source_documents}
    labels = set(getattr(plan, "mental_health_intent_labels", []) or [])
    score = 0.0

    use_case_terms = (
        "mental health",
        "psychiatric",
        "screening data",
        "risk score",
        "patient risk score",
        "clinical decision support",
        "diagnostic prediction",
        "health data",
        "special category data",
        "protected health information",
        "high risk ai system",
        "ai system",
    )
    if any(_concept_matches(text, term) for term in use_case_terms):
        score += 10
        debug["boosts"].append("mental_health_use_case_match")

    if "human_oversight" in labels:
        if source_document == "eu_ai_act.pdf" and any(
            _concept_matches(text, term)
            for term in ("human oversight", "high risk ai system", "provider", "deployer")
        ):
            score += 24
            debug["boosts"].append("mental_health_human_oversight_eu_ai_act")
        if source_document == "gdpr.pdf" and any(
            _concept_matches(text, term)
            for term in ("automated decision", "automated decision making", "profiling", "human intervention", "meaningful information")
        ):
            score += 24
            debug["boosts"].append("mental_health_human_oversight_gdpr")

    if "screening_data_risks" in labels:
        if source_document == "gdpr.pdf" and any(
            _concept_matches(text, term)
            for term in ("health data", "special category data", "data protection impact assessment", "lawful basis", "security")
        ):
            score += 20
            debug["boosts"].append("mental_health_screening_gdpr")
        if source_document == "hipaa.pdf" and any(
            _concept_matches(text, term)
            for term in ("protected health information", "phi", "disclosure", "covered entity", "business associate", "safeguards")
        ):
            score += 20
            debug["boosts"].append("mental_health_screening_hipaa")

    if "risk_score_safeguards" in labels:
        if source_document == "eu_ai_act.pdf" and any(
            _concept_matches(text, term)
            for term in ("risk management", "human oversight", "accuracy", "robustness", "cybersecurity")
        ):
            score += 18
            debug["boosts"].append("mental_health_risk_score_eu_ai_act")
        if source_document == "gdpr.pdf" and any(
            _concept_matches(text, term)
            for term in ("health data", "special category data", "data protection impact assessment", "automated decision", "security")
        ):
            score += 18
            debug["boosts"].append("mental_health_risk_score_gdpr")
        if source_document == "hipaa.pdf" and any(
            _concept_matches(text, term)
            for term in ("protected health information", "phi", "safeguards", "disclosure", "security")
        ):
            score += 18
            debug["boosts"].append("mental_health_risk_score_hipaa")

    if "transparency_explainability" in labels:
        if source_document == "gdpr.pdf" and any(
            _concept_matches(text, term)
            for term in ("transparency", "meaningful information", "automated decision", "profiling", "data subject")
        ):
            score += 20
            debug["boosts"].append("mental_health_transparency_gdpr")
        if source_document == "eu_ai_act.pdf" and any(
            _concept_matches(text, term)
            for term in ("transparency", "instructions for use", "technical documentation", "provider", "deployer")
        ):
            score += 20
            debug["boosts"].append("mental_health_transparency_eu_ai_act")

    if target_sources and source_document and source_document not in target_sources:
        score -= 18
        debug["penalties"].append("outside_mental_health_target_sources")
    if source_document == "hipaa.pdf" and "hipaa.pdf" not in target_sources and not labels.intersection({"screening_data_risks", "risk_score_safeguards", "hipaa_phi"}):
        score -= 28
        debug["penalties"].append("hipaa_not_requested_for_mental_health_intent")

    return score


def _mental_health_source_terms(plan: QueryPlan, source_document: str) -> list[str]:
    labels = set(getattr(plan, "mental_health_intent_labels", []) or [])
    source = _canonical_source_document(source_document)
    terms: list[str] = []

    if "human_oversight" in labels:
        if source == "eu_ai_act.pdf":
            terms.extend(["human oversight", "human oversight measures", "natural persons can oversee", "human operator", "operational constraints", "high-risk ai system"])
        elif source == "gdpr.pdf":
            terms.extend(["automated decision-making", "automated decision making", "profiling", "human intervention", "meaningful information", "data subject"])
    if "screening_data_risks" in labels:
        if source == "gdpr.pdf":
            terms.extend(["mental health data", "health data", "special category data", "data protection impact assessment", "security", "lawful basis"])
        elif source == "hipaa.pdf":
            terms.extend(["protected health information", "phi", "disclosure", "covered entity", "business associate", "safeguards"])
    if "risk_score_safeguards" in labels:
        if source == "eu_ai_act.pdf":
            terms.extend(["risk management", "human oversight", "accuracy", "robustness", "cybersecurity", "high-risk ai system"])
        elif source == "gdpr.pdf":
            terms.extend(["automated decision-making", "profiling", "special category data", "health data", "data protection impact assessment", "security"])
        elif source == "hipaa.pdf":
            terms.extend(["protected health information", "phi", "safeguards", "security", "disclosure", "minimum necessary"])
    if "transparency_explainability" in labels:
        if source == "eu_ai_act.pdf":
            terms.extend(["transparency", "instructions for use", "technical documentation", "explainability", "information to deployers"])
        elif source == "gdpr.pdf":
            terms.extend(["transparency", "meaningful information", "automated decision-making", "profiling", "data subject", "right of access"])
    if "technical_provider_obligations" in labels and source == "eu_ai_act.pdf":
        terms.extend(["technical documentation", "post-market monitoring", "provider", "conformity assessment", "quality management system", "risk management system"])
    if "hipaa_phi" in labels and source == "hipaa.pdf":
        terms.extend(["protected health information", "phi", "covered entity", "business associate", "authorization", "safeguards"])
    if "gdpr_data_protection" in labels and source == "gdpr.pdf":
        terms.extend(["controller", "processor", "special category data", "health data", "lawful basis", "explicit consent", "data protection impact assessment"])

    return _dedupe_values([_normalize_search_text(term) for term in terms])


def _score_gdpr_data_subject_rights(
    row: dict[str, Any],
    plan: QueryPlan,
    debug: dict[str, list[str]],
) -> float:
    if not _is_gdpr_data_subject_rights_query(plan):
        return 0.0
    direct_text = _normalize_search_text(_row_direct_obligation_text(row))
    matches = [term for term in GDPR_DATA_SUBJECT_RIGHTS_TERMS if _concept_matches(direct_text, term)]
    score = 0.0
    if matches:
        score += 95 + min(len(matches), 3) * 12
        debug["boosts"].append("gdpr_data_subject_right")
        debug["matched_terms"].extend(matches)
    elif any(_concept_matches(direct_text, term) for term in GDPR_GENERIC_RIGHTS_NOISE_TERMS):
        score -= 70
        debug["penalties"].append("generic_gdpr_rights_noise")
    return score


def _is_gdpr_data_subject_rights_query(plan: QueryPlan) -> bool:
    text = plan.normalized_question
    requested_terms = set(plan.terms) | set(plan.phrases) | set(plan.expansion_terms) | set(plan.detected_objects)
    return _query_requests_gdpr(plan) and (
        "data subject" in requested_terms
        or "data subject" in text
    ) and (
        "right" in text
        or bool(requested_terms.intersection(GDPR_DATA_SUBJECT_RIGHTS_TERMS))
    )


def _row_searchable_text(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "seed_name",
        "related_name",
        "statement_name",
        "evidence_text",
        "citation",
        "section_title",
        "article_number",
        "source_document",
        "source_chunk_text",
        "node_type",
    ):
        pieces.append(str(row.get(key) or ""))
    for key in ("seed_labels", "related_labels", "statement_labels", "aliases", "descriptions"):
        value = row.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        else:
            pieces.append(str(value or ""))
    return " ".join(pieces)


def _normalize_search_text(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("highriskaisystem", "high risk ai system")
    normalized = normalized.replace("high-risk artificial intelligence system", "high risk ai system")
    normalized = normalized.replace("artificial intelligence system", "ai system")
    replacements = {
        "high-r isk": "high-risk",
        "ai syste ms": "ai systems",
        "ai syst ems": "ai systems",
        "deplo yers": "deployers",
        "deplo yer": "deployer",
        "provid er": "provider",
        "provid ers": "providers",
        "risk-manag ement": "risk management",
        "risk manag ement": "risk management",
        "conf ormity assessment": "conformity assessment",
        "cybersecur ity": "cybersecurity",
        "post-mark et monito ring": "post market monitoring",
        "post-mark et monitoring": "post market monitoring",
        "record-k eeping": "record keeping",
        "safegua rds": "safeguards",
        "safe guards": "safeguards",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9_]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _concept_matches(text: str, concept: str) -> bool:
    normalized = _normalize_search_text(concept)
    if normalized in text:
        return True
    if normalized == "high risk ai system":
        return "high risk ai system" in text or ("high risk" in text and "ai system" in text)
    if normalized == "ai system":
        return "ai system" in text or "ai systems" in text
    if normalized == "provider":
        return re.search(r"\bproviders?\b", text) is not None
    if normalized == "deployer":
        return re.search(r"\bdeployers?\b", text) is not None
    return False


def _has_hipaa_noise(text: str) -> bool:
    return any(
        term in text
        for term in (
            "phi",
            "protected health information",
            "covered entity",
            "business associate",
            "health care provider",
            "hipaa",
        )
    )


def _query_requests_ai(plan: QueryPlan) -> bool:
    return "EU AI Act" in plan.explicit_regulations or "EU AI Act" in getattr(plan, "inferred_regulations", []) or any(
        _normalize_search_text(value) in {"ai system", "high risk ai system", "high risk artificial intelligence system"}
        or "ai system" in _normalize_search_text(value)
        or "artificial intelligence" in _normalize_search_text(value)
        for value in plan.concept_groups.get("object", [])
    ) or any(item in plan.normalized_question for item in ("ai system", "high-risk ai", "high risk ai", "artificial intelligence", "eu ai act"))


def _query_requests_hipaa(plan: QueryPlan) -> bool:
    return "HIPAA" in plan.explicit_regulations or "HIPAA" in getattr(plan, "inferred_regulations", []) or "hipaa" in plan.normalized_question or any(
        _normalize_search_text(value) in {"hipaa", "protected health information", "phi", "covered entity"}
        for value in plan.concept_groups.get("domain", [])
    )


def _query_requests_gdpr(plan: QueryPlan) -> bool:
    gdpr_domain_terms = {
        "gdpr",
        "general data protection regulation",
        "personal data",
        "health data",
        "special category data",
        "controller",
        "processor",
        "data subject",
    }
    return "GDPR" in plan.explicit_regulations or "GDPR" in getattr(plan, "inferred_regulations", []) or "gdpr" in plan.normalized_question or any(
        _normalize_search_text(value) in gdpr_domain_terms
        for value in plan.concept_groups.get("domain", [])
    )


def _is_concrete_ai_obligation_query(plan: QueryPlan) -> bool:
    return _query_requests_ai(plan) and (
        plan.intent == "obligations"
        or "concrete" in plan.normalized_question
        or bool(_requested_ai_topic_groups(plan))
    )


def _has_modality_or_topic_only_match(debug: dict[str, list[str]]) -> bool:
    groups = set(debug.get("matched_concept_groups") or [])
    has_object = any(group.startswith("object:") for group in groups)
    has_modality_or_topic = any(group.startswith(("modality:", "topic:")) for group in groups)
    return has_modality_or_topic and not has_object


def _row_matches_ai_system(row: dict[str, Any]) -> bool:
    text = _normalize_search_text(_row_searchable_text(row))
    return _concept_matches(text, "ai system") or _concept_matches(text, "high-risk ai system")


def _has_evidence(row: dict[str, Any]) -> bool:
    return bool(row.get("evidence_text") or row.get("source_chunk_text"))


def _is_missing_evidence_entity_row(row: dict[str, Any]) -> bool:
    labels = set(row.get("seed_labels") or []) | set(row.get("related_labels") or []) | set(row.get("statement_labels") or [])
    statement_name = str(row.get("statement_name") or "")
    return (
        not _has_evidence(row)
        and ("Entity" in labels or not statement_name or statement_name == "Unknown")
    )


def _is_evidence_expansion_seed(row: dict[str, Any], plan: QueryPlan) -> bool:
    if not _is_missing_evidence_entity_row(row):
        return False
    source_document = _canonical_source_document(row.get("source_document"))
    target_sources = {_canonical_source_document(source) for source in plan.target_source_documents}
    if plan.explicit_single_regulation and source_document and source_document not in target_sources:
        return False
    source_group = str(row.get("source_group") or "")
    source_group_document = _canonical_source_document(f"{source_group}.pdf") if source_group else ""
    if (
        plan.explicit_single_regulation
        and source_group in {"gdpr", "hipaa", "eu_ai_act"}
        and source_group_document
        and source_group_document not in target_sources
    ):
        return False
    if not _is_high_signal_expansion_seed(row, plan):
        return False
    if plan.explicit_regulations or plan.target_source_documents:
        return float(row.get("score") or 0) >= 12 and bool(
            row.get("matched_concept_groups")
            or row.get("source_group") in {"gdpr", "hipaa", "eu_ai_act"}
            or row.get("matched_terms")
        )
    if not _query_requests_ai(plan):
        return False
    matched_groups = set(row.get("matched_concept_groups") or [])
    has_object = any(item.startswith("object:") for item in matched_groups)
    has_topic = any(item.startswith("topic:") for item in matched_groups)
    return has_object or (has_topic and float(row.get("score") or 0) >= 15)


def _is_high_signal_expansion_seed(row: dict[str, Any], plan: QueryPlan) -> bool:
    text = _normalize_search_text(_row_direct_obligation_text(row))
    if _is_gdpr_data_subject_rights_query(plan):
        return bool(_infer_gdpr_rights_group(text))

    high_signal_terms = _high_signal_plan_terms(plan)
    if any(_concept_matches(text, term) for term in high_signal_terms):
        return True

    groups = set(row.get("matched_concept_groups") or [])
    return any(group.startswith(("object:", "topic:")) for group in groups) and not all(
        group.startswith(("domain:", "modality:")) for group in groups
    )


def _high_signal_plan_terms(plan: QueryPlan) -> list[str]:
    generic_terms = {
        "gdpr",
        "hipaa",
        "eu ai act",
        "eu_ai_act",
        "personal data",
        "data subject",
        "controller",
        "processor",
        "provider",
        "deployer",
        "obligation",
        "obligations",
        "requirement",
        "requirements",
        "compliance",
        "must",
        "should",
        "can",
    }
    raw_terms = [
        *plan.detected_objects,
        *plan.concept_groups.get("object", []),
        *plan.concept_groups.get("topic", []),
        *plan.expansion_terms,
    ]
    terms: list[str] = []
    for term in raw_terms:
        normalized = _normalize_search_text(str(term))
        if not normalized or normalized in generic_terms or len(normalized) < 4:
            continue
        terms.append(normalized)
    return _dedupe_values(terms)


def _seed_expansion_terms(seeds: list[dict[str, Any]], plan: QueryPlan) -> list[str]:
    terms: list[str] = []
    if _is_gdpr_data_subject_rights_query(plan):
        terms.extend(GDPR_DATA_SUBJECT_RIGHTS_TERMS)
    else:
        terms.extend(_high_signal_plan_terms(plan))
    for row in seeds:
        for value in (row.get("seed_name"), row.get("related_name"), row.get("statement_name")):
            normalized = _normalize_search_text(str(value or ""))
            if normalized and _is_high_signal_seed_name(normalized, plan):
                terms.append(normalized)
    return _dedupe_values(terms)


def _is_high_signal_seed_name(normalized_name: str, plan: QueryPlan) -> bool:
    if _is_gdpr_data_subject_rights_query(plan):
        return bool(_infer_gdpr_rights_group(normalized_name))
    return any(_concept_matches(normalized_name, term) for term in _high_signal_plan_terms(plan))


def _entity_seed_priority(row: dict[str, Any], plan: QueryPlan) -> tuple[int, float, str]:
    text = _row_direct_obligation_text(row)
    rights_group = _infer_gdpr_rights_group(text)
    route_rank = 0 if row.get("query_route") == "exact_phrase_entity" else 1
    rights_rank = 0 if _is_gdpr_data_subject_rights_query(plan) and rights_group else 1
    return (rights_rank, route_rank, -float(row.get("score") or 0), str(row.get("statement_name") or row.get("seed_name") or ""))


def _is_wrong_source_missing_entity_seed(row: dict[str, Any], plan: QueryPlan) -> bool:
    if not plan.explicit_single_regulation or not plan.target_source_documents or not _is_missing_evidence_entity_row(row):
        return False
    source_document = _canonical_source_document(row.get("source_document"))
    if not source_document:
        return False
    target_sources = {_canonical_source_document(source) for source in plan.target_source_documents}
    return source_document not in target_sources


def _is_unrelated_hipaa_row_for_ai_query(row: dict[str, Any]) -> bool:
    return row.get("source_group") == "hipaa" and not _row_matches_ai_system(row)


def _has_only_weak_ai_concept_coverage(row: dict[str, Any]) -> bool:
    if row.get("concrete_ai_obligation") and row.get("source_group") == "eu_ai_act":
        return False
    groups = set(row.get("matched_concept_groups") or [])
    has_object = any(item.startswith("object:") for item in groups)
    has_actor = any(item.startswith("actor:") for item in groups)
    has_topic = any(item.startswith("topic:") for item in groups)
    return not has_object and (has_actor or has_topic)


def _infer_source_group(row: dict[str, Any], text: str) -> str:
    source_document = _canonical_source_document(row.get("source_document"))
    if source_document == "gdpr.pdf":
        return "gdpr"
    if source_document == "eu_ai_act.pdf":
        return "eu_ai_act"
    if source_document == "hipaa.pdf":
        return "hipaa"
    if "eu ai act" in text or "regulation eu 2024 1689" in text or ("this regulation" in text and ("ai system" in text or "high risk" in text)):
        return "eu_ai_act"
    if "general data protection regulation" in text or re.search(r"\bgdpr\b", text):
        return "gdpr"
    if _has_hipaa_noise(text):
        return "hipaa"
    return "unknown"


def _dedupe_debug(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _debug_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _row_debug_id(row),
        "score": row.get("score"),
        "route": row.get("query_route"),
        "statement": row.get("statement_name"),
        "seed": row.get("seed_name"),
        "relationship": row.get("relationship"),
        "source_group": row.get("source_group"),
        "source_document": row.get("source_document"),
        "matched_concept_groups": row.get("matched_concept_groups") or [],
        "matched_terms": row.get("matched_terms") or [],
        "matched_topic_groups": row.get("matched_topic_groups") or [],
        "boosts": row.get("boosts") or [],
        "penalties": row.get("penalties") or [],
        "expanded_from_entity_seed": row.get("expanded_from_entity_seed") or False,
        "expanded_from_seed": row.get("expanded_from_seed") or row.get("original_entity_name"),
        "expansion_path": row.get("expansion_path") or row.get("relationship_path"),
    }


def _row_debug_id(row: dict[str, Any]) -> str:
    parts = [
        _canonical_source_document(row.get("source_document")) or "unknown",
        str(row.get("statement_key") or row.get("statement_name") or row.get("seed_name") or "unknown"),
        str(row.get("citation") or row.get("article_number") or ""),
    ]
    return "|".join(part.strip().replace("\n", " ") for part in parts)


def _score_negated_authorization_row(
    row: dict[str, Any],
    labels: set[str],
    relationship: Any,
    haystacks: list[str],
    plan: QueryPlan,
) -> float:
    score = 0.0
    evidence_text = str(row.get("evidence_text") or row.get("source_chunk_text") or "").lower()
    statement_name = str(row.get("statement_name") or "").lower()
    permission_phrase_present = _has_permission_exception_phrase(evidence_text) or _has_permission_exception_phrase(statement_name)
    negation_phrase_present = _has_negated_authorization_phrase(f"{statement_name} {evidence_text}")
    authorization_required_present = _has_authorization_required_phrase(statement_name, evidence_text)

    if "Permission" in labels:
        score += 8
    if "Exception" in labels:
        score += 16
    if "may disclose" in evidence_text or "may use or disclose" in evidence_text:
        score += 6
    if "without authorization" in evidence_text or "not required" in evidence_text:
        score += 6
    if "permitted disclosure" in statement_name or "permitted use" in statement_name:
        score += 5
    if any(term in evidence_text for term in PERMISSION_EXCEPTION_EXPANSIONS):
        score += 4
    if set(plan.detected_actors).intersection(set(haystacks[:2])):
        score += 4
    if authorization_required_present and not negation_phrase_present:
        score -= 80
    if "Prohibition" in labels and not permission_phrase_present:
        score -= 6
    if relationship == "REQUIRES_AUTHORIZATION" and not negation_phrase_present:
        score -= 5
    if labels.intersection({"Obligation", "Requirement"}) and not permission_phrase_present:
        score -= 4
    return score


def _has_permission_exception_phrase(text: str) -> bool:
    return any(term in text for term in NEGATED_AUTHORIZATION_EVIDENCE_TERMS) or any(
        term in text for term in PERMISSION_EXCEPTION_SEARCH_TERMS
    )


def _has_negated_authorization_phrase(text: str) -> bool:
    return any(term in text for term in STRICT_NEGATED_AUTHORIZATION_TERMS)


def _has_authorization_required_phrase(statement_name: str, evidence_text: str) -> bool:
    if "with authorization" in statement_name and "without authorization" not in statement_name:
        return True
    return any(term in evidence_text for term in AUTHORIZATION_REQUIRED_TERMS)


def _is_phi_disclosure_plan(plan: QueryPlan) -> bool:
    terms = set(plan.terms) | set(plan.phrases) | set(plan.expansion_terms) | set(plan.detected_objects)
    text = plan.normalized_question
    has_phi = bool(terms.intersection({"phi", "protected health information"})) or "protected health information" in text
    has_disclosure = any(term in terms for term in {"disclosure", "disclose", "use and disclosure"}) or "disclos" in text
    return has_phi and has_disclosure


def _query_parameters(plan: QueryPlan, limit: int) -> dict[str, Any]:
    search_terms = _dedupe_values([*plan.terms, *plan.phrases, *plan.expansion_terms])
    content_terms = [
        term
        for term in search_terms
        if _normalize_search_text(term) not in {"gdpr", "hipaa", "eu ai act", "eu_ai_act"}
    ]
    return {
        "question": plan.normalized_question,
        "terms": search_terms,
        "content_terms": content_terms or search_terms,
        "phrases": plan.phrases,
        "actors": plan.detected_actors,
        "objects": plan.detected_objects,
        "preferred_labels": plan.preferred_statement_labels,
        "preferred_relationships": plan.preferred_relationships,
        "graph_relationships": _dedupe_values([*GRAPH_EXPANSION_RELATIONSHIPS, *plan.preferred_relationships]),
        "evidence_relationships": EVIDENCE_RELATIONSHIPS,
        "permission_exception_terms": _dedupe_values([*PERMISSION_EXCEPTION_SEARCH_TERMS, *PERMISSION_EXCEPTION_EXPANSIONS]),
        "concrete_ai_terms": list(CONCRETE_AI_OBLIGATION_TERMS),
        "target_source_documents": [_canonical_source_document(source) for source in plan.target_source_documents],
        "limit": limit,
    }


def _merge_plans(parent: QueryPlan, child: QueryPlan) -> QueryPlan:
    child.preferred_statement_labels = _dedupe_values([*child.preferred_statement_labels, *parent.preferred_statement_labels])
    child.preferred_relationships = _dedupe_values([*child.preferred_relationships, *parent.preferred_relationships])
    child.expansion_terms = _dedupe_values([*child.expansion_terms, *parent.expansion_terms])
    child.detected_domains = _dedupe_values([*child.detected_domains, *parent.detected_domains])
    child.explicit_regulations = _dedupe_values([*child.explicit_regulations, *parent.explicit_regulations])
    child.target_source_documents = _dedupe_values([*child.target_source_documents, *parent.target_source_documents])
    child.explicit_single_regulation = parent.explicit_single_regulation
    child.cross_regulation = parent.cross_regulation
    child.is_mental_health_query = getattr(parent, "is_mental_health_query", False) or getattr(child, "is_mental_health_query", False)
    child.mental_health_intent_labels = _dedupe_values(
        [*getattr(child, "mental_health_intent_labels", []), *getattr(parent, "mental_health_intent_labels", [])]
    )
    child.inferred_regulations = _dedupe_values(
        [*getattr(child, "inferred_regulations", []), *getattr(parent, "inferred_regulations", [])]
    )
    child.mental_health_expansion_terms = _dedupe_values(
        [*getattr(child, "mental_health_expansion_terms", []), *getattr(parent, "mental_health_expansion_terms", [])]
    )
    return child


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = str(value).lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _dedupe_ids(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        parsed = str(value or "").strip()
        if not parsed:
            continue
        if parsed not in seen:
            seen.add(parsed)
            unique.append(parsed)
    return unique


def _infer_gdpr_rights_group(text: str) -> str:
    normalized = _normalize_search_text(text)
    for group, terms in GDPR_RIGHTS_GROUPS.items():
        if any(_concept_matches(normalized, term) for term in terms):
            return group
    return ""


_COMMON_RETURN = """
RETURN
    elementId(seed) AS seed_id,
    coalesce(seed.key, seed.id) AS seed_key,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS seed_name,
    labels(seed) AS seed_labels,
    type(rel) AS relationship,
    coalesce(related.key, related.id) AS related_key,
    coalesce(related.canonical_name, related.name, related.title, related.id, related.key) AS related_name,
    labels(related) AS related_labels,
    coalesce(statement.key, statement.id) AS statement_key,
    coalesce(statement.canonical_name, statement.id) AS statement_name,
    labels(statement) AS statement_labels,
    coalesce(seed.node_type, related.node_type, statement.node_type) AS node_type,
    coalesce(seed.aliases, related.aliases, statement.aliases, []) AS aliases,
    coalesce(seed.descriptions, related.descriptions, statement.descriptions, []) AS descriptions,
    coalesce(statement.evidence_text, head(coalesce(statement.evidence_texts, [])), head(coalesce(rel.evidence_texts, [])), head(coalesce(seed.evidence_texts, [])), head(coalesce(related.evidence_texts, []))) AS evidence_text,
    coalesce(
        head(coalesce(statement.citations, [])),
        head(coalesce(related.citations, [])),
        head(coalesce(rel.citations, []))
    ) AS citation,
    coalesce(statement.section_title, section.title, chunk.section_title, head(coalesce(rel.section_titles, []))) AS section_title,
    coalesce(statement.page_number, head(coalesce(rel.page_numbers, []))) AS page_number,
    coalesce(statement.article_number, head(coalesce(rel.article_numbers, []))) AS article_number,
    coalesce(statement.clause_number, head(coalesce(rel.clause_numbers, []))) AS clause_number,
    coalesce(statement.source_document, head(coalesce(statement.source_documents, [])), head(coalesce(rel.source_documents, [])), head(coalesce(seed.source_documents, [])), head(coalesce(related.source_documents, [])), regulation.name) AS source_document,
    chunk.text AS source_chunk_text,
    route_score AS score
LIMIT $limit
"""

_EXACT_PHRASE_ENTITY_QUERY = f"""
MATCH (seed:Entity)
WHERE any(phrase IN $phrases WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS phrase OR
    toLower(coalesce(seed.key, "")) CONTAINS phrase OR
    toLower(coalesce(seed.node_type, "")) CONTAINS phrase OR
    any(label IN labels(seed) WHERE toLower(label) CONTAINS phrase) OR
    any(alias IN coalesce(seed.aliases, []) WHERE toLower(alias) CONTAINS phrase) OR
    any(description IN coalesce(seed.descriptions, []) WHERE toLower(description) CONTAINS phrase) OR
    any(evidence IN coalesce(seed.evidence_texts, []) WHERE toLower(evidence) CONTAINS phrase) OR
    any(source IN coalesce(seed.source_documents, []) WHERE toLower(source) CONTAINS phrase)
)
WITH seed, 25 AS route_score LIMIT $limit
OPTIONAL MATCH (seed)-[rel]-(related)
WHERE related IS NOT NULL
  AND type(rel) IN $graph_relationships
WITH seed, rel, related,
     CASE
        WHEN "Statement" IN labels(seed) THEN seed
        WHEN "Statement" IN labels(related) THEN related
        ELSE null
     END AS statement,
     route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_STATEMENT_QUERY = f"""
MATCH (seed:Statement)
WHERE any(term IN $terms WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(seed.key, "")) CONTAINS term OR
    toLower(coalesce(seed.evidence_text, "")) CONTAINS term OR
    toLower(coalesce(seed.source_document, "")) CONTAINS term OR
    any(source IN coalesce(seed.source_documents, []) WHERE toLower(source) CONTAINS term) OR
    any(description IN coalesce(seed.descriptions, []) WHERE toLower(description) CONTAINS term) OR
    any(evidence IN coalesce(seed.evidence_texts, []) WHERE toLower(evidence) CONTAINS term)
)
WITH seed, 15 AS route_score LIMIT $limit
OPTIONAL MATCH (seed)-[rel]-(related)
WHERE related IS NOT NULL
  AND (type(rel) IN $graph_relationships OR type(rel) IN $evidence_relationships)
WITH seed, rel, related, seed AS statement, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_TARGET_SOURCE_STATEMENT_QUERY = f"""
MATCH (seed:Statement)
WHERE coalesce(seed.source_document, head(coalesce(seed.source_documents, [])), "") IN $target_source_documents
  AND any(term IN $content_terms WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(seed.key, "")) CONTAINS term OR
    toLower(coalesce(seed.evidence_text, "")) CONTAINS term OR
    any(citation IN coalesce(seed.citations, []) WHERE toLower(citation) CONTAINS term) OR
    any(description IN coalesce(seed.descriptions, []) WHERE toLower(description) CONTAINS term) OR
    any(evidence IN coalesce(seed.evidence_texts, []) WHERE toLower(evidence) CONTAINS term)
  )
WITH seed, 48 AS route_score LIMIT $limit
OPTIONAL MATCH (seed)-[rel]-(related)
WHERE related IS NOT NULL
  AND (type(rel) IN $graph_relationships OR type(rel) IN $evidence_relationships)
WITH seed, rel, related, seed AS statement, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_PERMISSION_EXCEPTION_STATEMENT_QUERY = f"""
MATCH (statement:Statement)
WHERE any(label IN labels(statement) WHERE label IN ["Permission", "Exception"])
  AND any(term IN $permission_exception_terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS term) OR
    any(description IN coalesce(statement.descriptions, []) WHERE toLower(description) CONTAINS term)
  )
WITH statement,
     CASE
        WHEN any(label IN labels(statement) WHERE label IN ["Permission", "Exception"]) THEN 30
        ELSE 20
     END AS route_score
LIMIT $limit
OPTIONAL MATCH (actor:Entity)-[actor_rel]-(statement)
WHERE any(actor_term IN $actors WHERE
    toLower(coalesce(actor.canonical_name, "")) CONTAINS actor_term OR
    any(alias IN coalesce(actor.aliases, []) WHERE toLower(alias) CONTAINS actor_term)
)
WITH statement, actor, actor_rel,
     CASE WHEN actor IS NOT NULL THEN route_score + 6 ELSE route_score END AS route_score
OPTIONAL MATCH (statement)-[evidence_rel]-(evidence_related)
WHERE evidence_related IS NOT NULL
  AND (type(evidence_rel) IN $evidence_relationships OR type(evidence_rel) IN $graph_relationships)
WITH coalesce(actor, statement) AS seed,
     coalesce(actor_rel, evidence_rel) AS rel,
     coalesce(evidence_related, actor) AS related,
     statement,
     route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_AI_RISK_STATEMENT_QUERY = f"""
MATCH (statement:Statement)
OPTIONAL MATCH (statement)-[:REFERENCES|CITES|DERIVED_FROM]-(matched_chunk:SourceChunk)
WHERE any(term IN $terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.key, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS term) OR
    any(description IN coalesce(statement.descriptions, []) WHERE toLower(description) CONTAINS term) OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS term
)
  AND (
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "high-risk" OR
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "high risk" OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS "high-risk" OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS "high risk" OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS "ai system" OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS "high-risk" OR toLower(evidence) CONTAINS "high risk" OR toLower(evidence) CONTAINS "ai system") OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "high-r isk" OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "high-risk" OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "high risk" OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "ai syste ms" OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "ai system"
  )
  AND (
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "risk" OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS "risk" OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS "risk") OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "risk" OR
    toLower(coalesce(matched_chunk.text, "")) CONTAINS "risk-manag ement" OR
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "human oversight" OR
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "technical documentation" OR
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "conformity assessment" OR
    toLower(coalesce(statement.canonical_name, "")) CONTAINS "post-market monitoring"
  )
WITH statement, coalesce(matched_chunk, null) AS matched_chunk, 28 AS route_score LIMIT $limit
OPTIONAL MATCH (seed:Entity)-[rel]-(statement)
WHERE type(rel) IN $graph_relationships
WITH coalesce(seed, statement) AS seed,
     rel,
     statement AS related,
     statement,
     matched_chunk,
     route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
WITH seed, rel, related, statement, coalesce(chunk, matched_chunk) AS chunk, route_score
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_CONCRETE_AI_OBLIGATION_QUERY = f"""
MATCH (statement:Statement)
OPTIONAL MATCH (statement)-[:REFERENCES|CITES|DERIVED_FROM]-(matched_chunk:SourceChunk)
WITH statement, matched_chunk,
     replace(toLower(
        coalesce(statement.canonical_name, "") + " " +
        coalesce(statement.evidence_text, "") + " " +
        reduce(value = "", item IN coalesce(statement.aliases, []) | value + " " + item) + " " +
        reduce(value = "", item IN coalesce(statement.descriptions, []) | value + " " + item)
     ), "-", " ") AS searchable
WHERE any(term IN $concrete_ai_terms WHERE searchable CONTAINS term)
   OR searchable CONTAINS "high-r isk"
   OR searchable CONTAINS "ai syste ms"
   OR searchable CONTAINS "risk-manag ement"
   OR searchable CONTAINS "conf ormity assessment"
   OR searchable CONTAINS "cybersecur ity"
   OR searchable CONTAINS "post-mark et monito ring"
   OR searchable CONTAINS "record-k eeping"
   OR searchable CONTAINS "provid er"
   OR searchable CONTAINS "deplo yer"
WITH statement, matched_chunk, 36 AS route_score LIMIT $limit
OPTIONAL MATCH (seed:Entity)-[rel]-(statement)
WHERE type(rel) IN $graph_relationships
WITH coalesce(seed, statement) AS seed, rel, statement AS related, statement, matched_chunk, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
WITH seed, rel, related, statement, coalesce(chunk, matched_chunk) AS chunk, route_score
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_ACTOR_TO_STATEMENT_QUERY = f"""
MATCH (seed:Entity)
WHERE any(actor IN $actors WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS actor OR
    any(alias IN coalesce(seed.aliases, []) WHERE toLower(alias) CONTAINS actor)
)
WITH seed, 20 AS route_score LIMIT $limit
MATCH (seed)-[rel]-(statement)
WHERE type(rel) IN $graph_relationships
  AND any(label IN labels(statement) WHERE label IN ["Statement", "Obligation", "Requirement", "Permission", "Prohibition", "Exception", "Definition", "Risk", "Control"])
WITH seed, rel, statement AS related, statement, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_ENTITY_EVIDENCE_EXPANSION_QUERY = """
MATCH (seed)
WHERE elementId(seed) IN $seed_ids
   OR toLower(coalesce(seed.key, seed.id, "")) IN $seed_keys
   OR toLower(coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key, "")) IN $seed_names
MATCH path = (seed)-[*1..2]-(statement)
WHERE all(path_rel IN relationships(path) WHERE type(path_rel) IN $graph_relationships OR type(path_rel) IN $evidence_relationships)
  AND any(label IN labels(statement) WHERE label IN ["Statement", "Obligation", "Requirement", "Permission", "Exception", "Risk", "Control"])
  AND (size($target_source_documents) = 0 OR coalesce(statement.source_document, head(coalesce(statement.source_documents, [])), "") IN $target_source_documents)
  AND coalesce(statement.evidence_text, head(coalesce(statement.evidence_texts, [])), "") <> ""
  AND any(term IN $seed_terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.key, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS term) OR
    any(citation IN coalesce(statement.citations, []) WHERE toLower(citation) CONTAINS term) OR
    any(description IN coalesce(statement.descriptions, []) WHERE toLower(description) CONTAINS term)
  )
WITH seed, path, statement,
     last(relationships(path)) AS rel,
     70 - (length(path) * 4) AS route_score
OPTIONAL MATCH (statement)-[:REFERENCES|CITES|DERIVED_FROM]-(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
RETURN
    elementId(seed) AS seed_id,
    coalesce(seed.key, seed.id) AS seed_key,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS seed_name,
    labels(seed) AS seed_labels,
    type(rel) AS relationship,
    coalesce(statement.key, statement.id) AS related_key,
    coalesce(statement.canonical_name, statement.name, statement.title, statement.id, statement.key) AS related_name,
    labels(statement) AS related_labels,
    coalesce(statement.key, statement.id) AS statement_key,
    coalesce(statement.canonical_name, statement.id) AS statement_name,
    labels(statement) AS statement_labels,
    coalesce(seed.node_type, statement.node_type) AS node_type,
    coalesce(seed.aliases, statement.aliases, []) AS aliases,
    coalesce(seed.descriptions, statement.descriptions, []) AS descriptions,
    coalesce(statement.evidence_text, head(coalesce(statement.evidence_texts, [])), chunk.text) AS evidence_text,
    coalesce(
        head(coalesce(statement.citations, [])),
        head(coalesce(rel.citations, []))
    ) AS citation,
    coalesce(statement.section_title, section.title, chunk.section_title, head(coalesce(rel.section_titles, []))) AS section_title,
    coalesce(statement.page_number, head(coalesce(rel.page_numbers, []))) AS page_number,
    coalesce(statement.article_number, head(coalesce(rel.article_numbers, []))) AS article_number,
    coalesce(statement.clause_number, head(coalesce(rel.clause_numbers, []))) AS clause_number,
    coalesce(statement.source_document, head(coalesce(statement.source_documents, [])), head(coalesce(rel.source_documents, [])), head(coalesce(seed.source_documents, [])), regulation.name) AS source_document,
    chunk.text AS source_chunk_text,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS original_entity_name,
    reduce(path_text = "", path_rel IN relationships(path) | path_text + CASE WHEN path_text = "" THEN "" ELSE " > " END + type(path_rel)) AS relationship_path,
    reduce(path_text = "", path_rel IN relationships(path) | path_text + CASE WHEN path_text = "" THEN "" ELSE " > " END + type(path_rel)) AS expansion_path,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS expanded_from_seed,
    true AS expanded_from_entity_seed,
    route_score AS score
LIMIT $limit
"""

_ENTITY_SECTION_EVIDENCE_EXPANSION_QUERY = """
MATCH (statement:Statement)
WHERE (size($target_source_documents) = 0 OR coalesce(statement.source_document, head(coalesce(statement.source_documents, [])), "") IN $target_source_documents)
  AND coalesce(statement.evidence_text, head(coalesce(statement.evidence_texts, [])), "") <> ""
  AND any(term IN $seed_terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.key, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS term) OR
    any(citation IN coalesce(statement.citations, []) WHERE toLower(citation) CONTAINS term) OR
    any(description IN coalesce(statement.descriptions, []) WHERE toLower(description) CONTAINS term)
  )
OPTIONAL MATCH (statement)-[:REFERENCES|CITES|DERIVED_FROM]-(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
WHERE section IS NULL OR any(term IN $seed_terms WHERE toLower(coalesce(section.title, "")) CONTAINS term OR toLower(coalesce(chunk.text, "")) CONTAINS term)
WITH statement, chunk, section, 58 AS route_score LIMIT $limit
OPTIONAL MATCH (seed:Entity)
WHERE elementId(seed) IN $seed_ids
   OR toLower(coalesce(seed.key, seed.id, "")) IN $seed_keys
   OR toLower(coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key, "")) IN $seed_names
WITH seed, statement, chunk, section, route_score
RETURN
    elementId(seed) AS seed_id,
    coalesce(seed.key, seed.id) AS seed_key,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS seed_name,
    labels(seed) AS seed_labels,
    "SECTION_TERM_MATCH" AS relationship,
    coalesce(statement.key, statement.id) AS related_key,
    coalesce(statement.canonical_name, statement.name, statement.title, statement.id, statement.key) AS related_name,
    labels(statement) AS related_labels,
    coalesce(statement.key, statement.id) AS statement_key,
    coalesce(statement.canonical_name, statement.id) AS statement_name,
    labels(statement) AS statement_labels,
    coalesce(seed.node_type, statement.node_type) AS node_type,
    coalesce(seed.aliases, statement.aliases, []) AS aliases,
    coalesce(seed.descriptions, statement.descriptions, []) AS descriptions,
    coalesce(statement.evidence_text, head(coalesce(statement.evidence_texts, [])), chunk.text) AS evidence_text,
    head(coalesce(statement.citations, [])) AS citation,
    coalesce(statement.section_title, section.title, chunk.section_title) AS section_title,
    statement.page_number AS page_number,
    statement.article_number AS article_number,
    statement.clause_number AS clause_number,
    coalesce(statement.source_document, head(coalesce(statement.source_documents, []))) AS source_document,
    chunk.text AS source_chunk_text,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS original_entity_name,
    "SECTION_TERM_MATCH" AS relationship_path,
    "SECTION_TERM_MATCH" AS expansion_path,
    coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key) AS expanded_from_seed,
    true AS expanded_from_entity_seed,
    route_score AS score
LIMIT $limit
"""

_STATEMENT_TO_EVIDENCE_QUERY = f"""
MATCH (statement:Statement)
WHERE any(term IN $terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.key, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
    toLower(coalesce(statement.source_document, "")) CONTAINS term OR
    any(source IN coalesce(statement.source_documents, []) WHERE toLower(source) CONTAINS term) OR
    any(description IN coalesce(statement.descriptions, []) WHERE toLower(description) CONTAINS term) OR
    any(evidence IN coalesce(statement.evidence_texts, []) WHERE toLower(evidence) CONTAINS term)
)
WITH statement, 18 AS route_score LIMIT $limit
MATCH (statement)-[rel]-(related)
WHERE type(rel) IN $evidence_relationships
WITH statement AS seed, rel, related, statement, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""

_FALLBACK_STATEMENT_QUERY = f"""
MATCH (seed:Statement)
WHERE any(term IN $terms WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(seed.key, "")) CONTAINS term OR
    toLower(coalesce(seed.evidence_text, "")) CONTAINS term OR
    toLower(coalesce(seed.source_document, "")) CONTAINS term OR
    any(source IN coalesce(seed.source_documents, []) WHERE toLower(source) CONTAINS term) OR
    any(evidence IN coalesce(seed.evidence_texts, []) WHERE toLower(evidence) CONTAINS term) OR
    any(description IN coalesce(seed.descriptions, []) WHERE toLower(description) CONTAINS term)
)
WITH seed,
     CASE
        WHEN any(label IN labels(seed) WHERE label IN $preferred_labels) THEN 14
        ELSE 8
     END AS route_score
LIMIT $limit
OPTIONAL MATCH (seed)-[rel]-(related)
WHERE related IS NOT NULL
WITH seed, rel, related, seed AS statement, route_score
OPTIONAL MATCH (statement)-[:REFERENCES]->(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
{_COMMON_RETURN}
"""
