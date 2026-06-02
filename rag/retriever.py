from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from graph.neo4j_client import Neo4jClient
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
    "NOTIFIES",
]

EVIDENCE_RELATIONSHIPS = ["CITES", "REFERENCES", "CONTAINS", "HAS_SECTION", "DERIVED_FROM"]

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

    def retrieve(
        self,
        question: str,
        limit: int = 30,
        conversation_history: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        plan = build_query_plan(question, conversation_history)
        self.last_query_plan = plan
        self.last_debug_rows = []

        if plan.category not in GRAPH_RETRIEVAL_CATEGORIES:
            return []

        retrieval_questions = plan.sub_questions or [plan.normalized_question]
        rows: list[dict[str, Any]] = []
        query_limit = max(limit * 2, limit)
        for retrieval_question in retrieval_questions:
            sub_plan = build_query_plan(retrieval_question, conversation_history)
            merged_plan = _merge_plans(plan, sub_plan)
            if not merged_plan.terms and not merged_plan.phrases and not merged_plan.expansion_terms:
                continue
            rows.extend(self._run_graph_routes(merged_plan, query_limit))

        ranked_rows = _dedupe_rows(rows, plan)[:limit]
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

    haystacks = [
        str(row.get("seed_name") or "").lower(),
        str(row.get("related_name") or "").lower(),
        str(row.get("statement_name") or "").lower(),
        str(row.get("evidence_text") or "").lower(),
        str(row.get("source_chunk_text") or "").lower(),
    ]
    labels = set(row.get("statement_labels") or []) | set(row.get("seed_labels") or []) | set(row.get("related_labels") or [])
    relationship = row.get("relationship")

    for phrase in plan.phrases:
        if phrase in haystacks[0] or phrase in haystacks[1] or phrase in haystacks[2]:
            score += 8
        if any(phrase in haystack for haystack in haystacks[3:]):
            score += 4
    matched_terms = sum(1 for term in plan.terms if any(term in haystack for haystack in haystacks))
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
    if plan.has_negated_authorization:
        score += _score_negated_authorization_row(row, labels, relationship, haystacks, plan)
    return round(score, 2)


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
    coalesce(statement.source_document, head(coalesce(rel.source_documents, [])), head(coalesce(seed.source_documents, [])), head(coalesce(related.source_documents, [])), regulation.name) AS source_document,
    chunk.text AS source_chunk_text,
    route_score AS score
LIMIT $limit
"""

_EXACT_PHRASE_ENTITY_QUERY = f"""
MATCH (seed:Entity)
WHERE any(phrase IN $phrases WHERE
    toLower(coalesce(seed.canonical_name, "")) CONTAINS phrase OR
    toLower(coalesce(seed.key, "")) CONTAINS phrase OR
    any(alias IN coalesce(seed.aliases, []) WHERE toLower(alias) CONTAINS phrase)
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
    toLower(coalesce(seed.evidence_text, "")) CONTAINS term OR
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

_STATEMENT_TO_EVIDENCE_QUERY = f"""
MATCH (statement:Statement)
WHERE any(term IN $terms WHERE
    toLower(coalesce(statement.canonical_name, "")) CONTAINS term OR
    toLower(coalesce(statement.evidence_text, "")) CONTAINS term OR
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
    toLower(coalesce(seed.evidence_text, "")) CONTAINS term OR
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
