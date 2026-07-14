from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.logger import get_logger


logger = get_logger(__name__)

# Relationship types that connect a Statement to an Entity (see graph/writer.py
# _link_statement_entities). A SourceChunk reaches its entities *through* the
# statements that REFERENCE it — there is no direct (:Chunk)-[:MENTIONS]->(:Entity)
# edge in this graph.
STATEMENT_ENTITY_LINK_RELATIONSHIPS = ["APPLIES_TO", "RELATED_TO", "HAS_EXCEPTION", "CITES"]

# Entity-to-Entity relationship types used for one-hop expansion. These are the
# real names declared in graph/writer.py ALLOWED_REL_TYPES (the task's "IMPOSES"
# is actually IMPOSES_ON in this schema).
ENTITY_EXPANSION_RELATIONSHIPS = ["RELATED_TO", "IMPOSES_ON", "APPLIES_TO"]

MAX_EXPANSION_HOPS = 5


class Neo4jConnectionError(RuntimeError):
    pass


def _coerce_hops(max_hops: int) -> int:
    try:
        hops = int(max_hops)
    except (TypeError, ValueError):
        hops = 1
    return max(1, min(hops, MAX_EXPANSION_HOPS))


def build_chunk_expansion_query(max_hops: int) -> str:
    """Build the Cypher that expands SourceChunk nodes into entity triples.

    Traverses (:SourceChunk)<-[:REFERENCES]-(:Statement)-[link]->(:Entity) to find
    the entities a chunk mentions, then follows one-to-``max_hops`` hops of
    Entity-to-Entity relationships. Returns one row per relationship along each
    matched path (plus a null-edge row for linked entities with no outgoing
    relationship), so the caller can assemble (subject, predicate, object) triples.
    """
    hops = _coerce_hops(max_hops)
    rel_pattern = "|".join(ENTITY_EXPANSION_RELATIONSHIPS)
    return f"""
    UNWIND $chunk_ids AS chunk_id
    MATCH (c:SourceChunk {{id: chunk_id}})<-[:REFERENCES]-(:Statement)-[link]->(e:Entity)
    WHERE type(link) IN $link_rel_types
    OPTIONAL MATCH path = (e)-[:{rel_pattern}*1..{hops}]->(:Entity)
    WITH chunk_id, e, path
    UNWIND (CASE WHEN path IS NULL THEN [null] ELSE relationships(path) END) AS rel
    RETURN chunk_id AS chunk_id,
           e.key AS entity_key,
           e.canonical_name AS entity_name,
           e.node_type AS entity_type,
           CASE WHEN rel IS NULL THEN null ELSE startNode(rel).key END AS subject_key,
           CASE WHEN rel IS NULL THEN null ELSE startNode(rel).canonical_name END AS subject_name,
           type(rel) AS predicate,
           CASE WHEN rel IS NULL THEN null ELSE endNode(rel).key END AS object_key,
           CASE WHEN rel IS NULL THEN null ELSE endNode(rel).canonical_name END AS object_name
    """


def assemble_chunk_expansion(
    chunk_ids: Iterable[str],
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Group flat expansion rows into a dict keyed by chunk_id.

    Each value is ``{"entities": [...], "triples": [(subject, predicate, object), ...]}``.
    Entities and triples are de-duplicated while preserving first-seen order.
    """
    result: dict[str, dict[str, Any]] = {}
    entity_seen: dict[str, set[str]] = {}
    triple_seen: dict[str, set[tuple[str, str, str]]] = {}

    def ensure(chunk_id: str) -> None:
        if chunk_id not in result:
            result[chunk_id] = {"entities": [], "triples": []}
            entity_seen[chunk_id] = set()
            triple_seen[chunk_id] = set()

    for chunk_id in chunk_ids:
        ensure(chunk_id)

    for row in rows:
        chunk_id = row.get("chunk_id")
        if chunk_id is None:
            continue
        ensure(chunk_id)

        entity_key = row.get("entity_key")
        if entity_key and entity_key not in entity_seen[chunk_id]:
            entity_seen[chunk_id].add(entity_key)
            result[chunk_id]["entities"].append(
                {
                    "key": entity_key,
                    "name": row.get("entity_name"),
                    "node_type": row.get("entity_type"),
                }
            )

        subject = row.get("subject_name")
        predicate = row.get("predicate")
        obj = row.get("object_name")
        if subject and predicate and obj:
            triple = (subject, predicate, obj)
            if triple not in triple_seen[chunk_id]:
                triple_seen[chunk_id].add(triple)
                result[chunk_id]["triples"].append(triple)

    return result


class Neo4jClient:
    def __init__(self, settings: Settings) -> None:
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        try:
            logger.debug("Verifying Neo4j connectivity uri=%s database=%s", self.uri, self.database)
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(
                f"Could not connect to Neo4j at {self.uri}. "
                "If you want a local database, start Neo4j so it listens on this Bolt address. "
                "If you want Neo4j Aura, update NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and NEO4J_DATABASE in .env."
            ) from exc
        except AuthError as exc:
            raise Neo4jConnectionError(
                f"Connected to Neo4j at {self.uri}, but authentication failed for user {self.username}. "
                "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env."
            ) from exc
        except ConfigurationError as exc:
            raise Neo4jConnectionError(
                f"Neo4j configuration is invalid for URI {self.uri}. Check NEO4J_URI in .env."
            ) from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        logger.debug(
            "Running Neo4j query database=%s query_head=%s param_keys=%s",
            self.database,
            " ".join(query.strip().split())[:160],
            sorted((parameters or {}).keys()),
        )
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
            logger.debug("Neo4j query completed database=%s records=%s", self.database, len(records))
            return records

    def run_statements(self, statements: Iterable[str]) -> None:
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                logger.debug("Running schema statement: %s", statement)
                session.run(statement)

    def expand_chunk_entities(
        self,
        chunk_ids: list[str],
        max_hops: int = 1,
    ) -> dict[str, dict[str, Any]]:
        """Expand chunk IDs into their linked entities and one-hop relationship triples.

        Given chunk IDs (e.g. from the semantic retriever's vector search), traverse
        (:SourceChunk)<-[:REFERENCES]-(:Statement)-[link]->(:Entity) and up to
        ``max_hops`` of Entity-to-Entity relationships, returning a dict keyed by
        chunk_id whose values hold the linked entities and (subject, predicate,
        object) triples.
        """
        if not chunk_ids:
            return {}
        hops = _coerce_hops(max_hops)
        query = build_chunk_expansion_query(hops)
        parameters = {
            "chunk_ids": list(chunk_ids),
            "link_rel_types": STATEMENT_ENTITY_LINK_RELATIONSHIPS,
            "max_hops": hops,
        }
        logger.debug("Expanding %s chunk(s) into entities max_hops=%s", len(chunk_ids), hops)
        rows = self.run_query(query, parameters)
        return assemble_chunk_expansion(list(chunk_ids), rows)
