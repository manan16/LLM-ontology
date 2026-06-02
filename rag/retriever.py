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
        }

    def retrieve(
        self,
        question: str,
        limit: int = 30,
        conversation_history: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        plan = build_query_plan(question, conversation_history)
        self.last_query_plan = plan
        self.last_debug_rows = []
        self.last_expansion_debug = {"seeds": [], "expanded": [], "excluded": [], "selected": []}

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
        self.last_expansion_debug["excluded"] = [_debug_row(row) for row in excluded_rows[:20]]
        self.last_expansion_debug["selected"] = [_debug_row(row) for row in ranked_rows[:10]]
        self.last_debug_rows = ranked_rows
        return ranked_rows

    def _run_graph_routes(self, plan: QueryPlan, limit: int) -> list[dict[str, Any]]:
        parameters = _query_parameters(plan, limit)
        rows: list[dict[str, Any]] = []
        routes = [
            ("exact_phrase_entity", _EXACT_PHRASE_ENTITY_QUERY),
            ("statement", _STATEMENT_QUERY),
            ("actor_to_statement", _ACTOR_TO_STATEMENT_QUERY),
            ("statement_to_evidence", _STATEMENT_TO_EVIDENCE_QUERY),
        ]
        if plan.intent in {"exceptions", "permissions"}:
            routes.insert(1, ("permission_exception_statement", _PERMISSION_EXCEPTION_STATEMENT_QUERY))
        if _query_requests_ai(plan) and plan.concept_groups.get("topic"):
            routes.insert(2, ("ai_risk_statement", _AI_RISK_STATEMENT_QUERY))
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
        seeds = [row for row in ranked_rows[:25] if _is_evidence_expansion_seed(row, plan)]
        self.last_expansion_debug["seeds"] = [_debug_row(row) for row in seeds]
        if not seeds:
            return []

        parameters = {
            **_query_parameters(plan, limit),
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
        }
        if not parameters["seed_names"] and not parameters["seed_keys"]:
            return []

        expanded = self._run_route("entity_evidence_expansion", _ENTITY_EVIDENCE_EXPANSION_QUERY, parameters)
        for row in expanded:
            row.setdefault("score", 24)
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
    use_ai_filter = _query_requests_ai(plan) and bool(ai_evidence_rows)
    final_pool: list[dict[str, Any]] = []

    for row in rows:
        if evidence_rows and _is_missing_evidence_entity_row(row):
            excluded.append(row)
            continue
        if use_ai_filter and _is_unrelated_hipaa_row_for_ai_query(row):
            excluded.append(row)
            continue
        if use_ai_filter and _has_only_weak_ai_concept_coverage(row):
            excluded.append(row)
            continue
        final_pool.append(row)

    if not final_pool:
        final_pool = rows
        excluded = []
    return final_pool[:limit], excluded


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": row.get("score"),
        "query_route": row.get("query_route"),
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
    if _is_phi_disclosure_plan(plan) and is_suppressed_context_row(row):
        score -= 120
        debug["penalties"].append("employment_record_noise")
    concept_score = _score_concept_coverage(row, plan, searchable_text, labels, relationship, debug)
    score += concept_score
    if plan.has_negated_authorization:
        score += _score_negated_authorization_row(row, labels, relationship, haystacks, plan)
    row["matched_concept_groups"] = _dedupe_debug(debug["matched_concept_groups"])
    row["matched_terms"] = _dedupe_debug(debug["matched_terms"])
    row["penalties"] = _dedupe_debug(debug["penalties"])
    row["source_group"] = _infer_source_group(row, searchable_text)
    return round(score, 2)


def _new_score_debug() -> dict[str, list[str]]:
    return {"matched_concept_groups": [], "matched_terms": [], "penalties": []}


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
    if query_requests_ai and _has_hipaa_noise(text) and not has_ai_context:
        score -= 12
        debug["penalties"].append("unrelated_hipaa_for_ai_query")

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
    if relationship in plan.preferred_relationships:
        score += 3
    if row.get("evidence_text"):
        score += 2
    if row.get("source_document") or row.get("section_title") or row.get("article_number"):
        score += 2
    if not row.get("evidence_text") and not row.get("source_chunk_text"):
        score -= 3
        debug["penalties"].append("missing_evidence")

    if _has_hipaa_noise(text) and not has_ai_context:
        score -= 8
        debug["penalties"].append("hipaa_noise")
        if any("object:" in item for item in debug["matched_concept_groups"]):
            pass
        else:
            debug["penalties"].append("missing_ai_system")

    return score


def _row_searchable_text(row: dict[str, Any]) -> str:
    pieces: list[str] = []
    for key in (
        "seed_name",
        "related_name",
        "statement_name",
        "evidence_text",
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
        "provid ers": "providers",
        "risk-manag ement": "risk management",
        "risk manag ement": "risk management",
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
    return any(
        _normalize_search_text(value) in {"ai system", "high risk ai system", "high risk artificial intelligence system"}
        or "ai system" in _normalize_search_text(value)
        or "artificial intelligence" in _normalize_search_text(value)
        for value in plan.concept_groups.get("object", [])
    ) or any(item in plan.normalized_question for item in ("ai system", "high-risk ai", "high risk ai", "artificial intelligence"))


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
    if not _query_requests_ai(plan) or not _is_missing_evidence_entity_row(row):
        return False
    matched_groups = set(row.get("matched_concept_groups") or [])
    has_object = any(item.startswith("object:") for item in matched_groups)
    has_topic = any(item.startswith("topic:") for item in matched_groups)
    return has_object or (has_topic and float(row.get("score") or 0) >= 15)


def _is_unrelated_hipaa_row_for_ai_query(row: dict[str, Any]) -> bool:
    return row.get("source_group") == "hipaa" and not _row_matches_ai_system(row)


def _has_only_weak_ai_concept_coverage(row: dict[str, Any]) -> bool:
    groups = set(row.get("matched_concept_groups") or [])
    has_object = any(item.startswith("object:") for item in groups)
    has_actor = any(item.startswith("actor:") for item in groups)
    has_topic = any(item.startswith("topic:") for item in groups)
    return not has_object and (has_actor or has_topic)


def _infer_source_group(row: dict[str, Any], text: str) -> str:
    source_document = str(row.get("source_document") or "").lower()
    if "eu_ai_act" in source_document or "eu ai act" in text or "regulation eu 2024 1689" in text or "this regulation" in text:
        return "eu_ai_act"
    if "hipaa" in source_document or _has_hipaa_noise(text):
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
        "score": row.get("score"),
        "route": row.get("query_route"),
        "statement": row.get("statement_name"),
        "seed": row.get("seed_name"),
        "relationship": row.get("relationship"),
        "source_group": row.get("source_group"),
        "source_document": row.get("source_document"),
        "matched_concept_groups": row.get("matched_concept_groups") or [],
        "matched_terms": row.get("matched_terms") or [],
        "penalties": row.get("penalties") or [],
    }


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
        score += 8
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
    return {
        "question": plan.normalized_question,
        "terms": search_terms,
        "phrases": plan.phrases,
        "actors": plan.detected_actors,
        "objects": plan.detected_objects,
        "preferred_labels": plan.preferred_statement_labels,
        "preferred_relationships": plan.preferred_relationships,
        "graph_relationships": _dedupe_values([*GRAPH_EXPANSION_RELATIONSHIPS, *plan.preferred_relationships]),
        "evidence_relationships": EVIDENCE_RELATIONSHIPS,
        "permission_exception_terms": _dedupe_values([*PERMISSION_EXCEPTION_SEARCH_TERMS, *PERMISSION_EXCEPTION_EXPANSIONS]),
        "limit": limit,
    }


def _merge_plans(parent: QueryPlan, child: QueryPlan) -> QueryPlan:
    child.preferred_statement_labels = _dedupe_values([*child.preferred_statement_labels, *parent.preferred_statement_labels])
    child.preferred_relationships = _dedupe_values([*child.preferred_relationships, *parent.preferred_relationships])
    child.expansion_terms = _dedupe_values([*child.expansion_terms, *parent.expansion_terms])
    child.detected_domains = _dedupe_values([*child.detected_domains, *parent.detected_domains])
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


_COMMON_RETURN = """
RETURN
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
WHERE toLower(coalesce(seed.key, seed.id, "")) IN $seed_keys
   OR toLower(coalesce(seed.canonical_name, seed.name, seed.title, seed.id, seed.key, "")) IN $seed_names
MATCH path = (seed)-[*1..2]-(statement)
WHERE all(path_rel IN relationships(path) WHERE type(path_rel) IN $graph_relationships OR type(path_rel) IN $evidence_relationships)
  AND any(label IN labels(statement) WHERE label IN ["Statement", "Obligation", "Requirement", "Permission", "Exception", "Risk", "Control"])
WITH seed, path, statement,
     last(relationships(path)) AS rel,
     34 - length(path) AS route_score
OPTIONAL MATCH (statement)-[:REFERENCES|CITES|DERIVED_FROM]-(chunk:SourceChunk)
OPTIONAL MATCH (section:Section)-[:CONTAINS]->(chunk)
OPTIONAL MATCH (regulation:Regulation)-[:HAS_SECTION]->(section)
RETURN
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
