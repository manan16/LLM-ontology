from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.config import Settings
from app.logger import get_logger
from graph.neo4j_client import Neo4jClient


logger = get_logger(__name__)

CHUNK_LABEL = "SourceChunk"
EMBEDDING_PROPERTY = "embedding"


def ensure_vector_index(neo4j_client: Neo4jClient, settings: Settings) -> None:
    """Create the SourceChunk vector index if it does not already exist.

    Uses the native Neo4j 5.x vector index (no external vector database). The
    statement is idempotent thanks to ``IF NOT EXISTS``.
    """
    query = f"""
    CREATE VECTOR INDEX {settings.vector_index_name} IF NOT EXISTS
    FOR (c:{CHUNK_LABEL}) ON (c.{EMBEDDING_PROPERTY})
    OPTIONS {{ indexConfig: {{
        `vector.dimensions`: $dimensions,
        `vector.similarity_function`: $similarity
    }} }}
    """
    logger.info(
        "Ensuring Neo4j vector index name=%s dimensions=%s similarity=%s",
        settings.vector_index_name,
        settings.vector_dimensions,
        settings.vector_similarity,
    )
    neo4j_client.run_query(
        query,
        {"dimensions": settings.vector_dimensions, "similarity": settings.vector_similarity},
    )


def count_chunks_missing_embeddings(neo4j_client: Neo4jClient) -> int:
    query = f"""
    MATCH (c:{CHUNK_LABEL})
    WHERE c.{EMBEDDING_PROPERTY} IS NULL AND c.text IS NOT NULL
    RETURN count(c) AS missing
    """
    rows = neo4j_client.run_query(query)
    return int(rows[0]["missing"]) if rows else 0


def iter_chunks_missing_embeddings(
    neo4j_client: Neo4jClient,
    batch_size: int = 128,
) -> Iterator[list[dict[str, Any]]]:
    """Yield batches of chunks that still need embeddings.

    Each yielded item is ``{"chunk_id": str, "text": str}``. Chunks that already
    carry an ``embedding`` property are skipped, which keeps the backfill
    idempotent and safe to re-run.
    """
    query = f"""
    MATCH (c:{CHUNK_LABEL})
    WHERE c.{EMBEDDING_PROPERTY} IS NULL AND c.text IS NOT NULL
    RETURN c.id AS chunk_id, c.text AS text
    ORDER BY c.id
    SKIP $skip LIMIT $limit
    """
    skip = 0
    while True:
        batch = neo4j_client.run_query(query, {"skip": skip, "limit": batch_size})
        if not batch:
            return
        yield batch
        # We remove embeddings-less chunks as we go, so SKIP stays at 0 only if
        # writes succeed. Advancing SKIP keeps progress even if a write is retried.
        skip += len(batch)


def store_chunk_embeddings(
    neo4j_client: Neo4jClient,
    embeddings: list[dict[str, Any]],
) -> int:
    """Persist embeddings onto existing SourceChunk nodes.

    ``embeddings`` is a list of ``{"chunk_id": str, "embedding": list[float]}``.
    Uses ``db.setNodeVectorProperty`` so the values are stored in the compact
    vector format the index expects. Returns the number of nodes updated.
    """
    if not embeddings:
        return 0
    query = f"""
    UNWIND $rows AS row
    MATCH (c:{CHUNK_LABEL} {{id: row.chunk_id}})
    CALL db.create.setNodeVectorProperty(c, '{EMBEDDING_PROPERTY}', row.embedding)
    RETURN count(c) AS updated
    """
    rows = neo4j_client.run_query(query, {"rows": embeddings})
    updated = int(rows[0]["updated"]) if rows else 0
    logger.debug("Stored embeddings for %s chunks", updated)
    return updated
