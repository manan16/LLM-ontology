"""Retrieval routing: query construction and per-route parameters.

The Cypher query templates for each retrieval route (entity/statement/actor/
evidence/permission/exception), the route/category configuration, and the
parameter/plan builders that feed them. Depends on scoring/dedup helpers; the
GraphRetriever class in core drives these routes.
"""

from __future__ import annotations

from typing import Any

from rag.query_planner import PERMISSION_EXCEPTION_EXPANSIONS, QueryPlan
from rag.retriever.dedup import _dedupe_values
from rag.retriever.scoring import (
    CONCRETE_AI_OBLIGATION_TERMS,
    PERMISSION_EXCEPTION_SEARCH_TERMS,
    _canonical_source_document,
    _normalize_search_text,
)


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
