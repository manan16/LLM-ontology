from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings
from app.logger import get_logger
from graph.embeddings import CHUNK_LABEL, ensure_vector_index
from graph.neo4j_client import Neo4jClient
from rag.embedding_service import EmbeddingService, get_embedding_service


logger = get_logger(__name__)

# Vector search over SourceChunk nodes, joined back to their Section/Regulation
# so downstream context building has the same fields graph rows carry.
_VECTOR_QUERY = f"""
CALL db.index.vector.queryNodes($index_name, $top_k, $embedding) YIELD node, score
MATCH (sec:Section)-[:CONTAINS]->(node:{CHUNK_LABEL})
OPTIONAL MATCH (reg:Regulation)-[:HAS_SECTION]->(sec)
RETURN node.id AS chunk_id,
       node.text AS source_chunk_text,
       node.section_title AS section_title,
       sec.id AS section_id,
       reg.id AS source_document,
       score AS similarity
ORDER BY score DESC
"""


class SemanticRetriever:
    """Dense retrieval over Chunk embeddings using a Neo4j vector index."""

    def __init__(
        self,
        neo4j_client: Neo4jClient | None = None,
        embedding_service: EmbeddingService | None = None,
        settings: Settings | None = None,
        ensure_index: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.neo4j_client = neo4j_client or Neo4jClient(self.settings)
        self.embedding_service = embedding_service or get_embedding_service(self.settings)
        if ensure_index:
            try:
                ensure_vector_index(self.neo4j_client, self.settings)
            except Exception:
                logger.exception("Could not ensure vector index; semantic search may be unavailable")

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return the top-k chunks most similar to ``query`` with similarity scores.

        Each returned row is shaped like the graph retriever's rows so it can be
        merged straight into the existing context/evidence pipeline.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        k = int(top_k or self.settings.semantic_top_k)

        try:
            embedding = self.embedding_service.embed_query(cleaned)
        except Exception:
            logger.exception("Failed to embed query for semantic retrieval")
            return []

        try:
            records = self.neo4j_client.run_query(
                _VECTOR_QUERY,
                {"index_name": self.settings.vector_index_name, "top_k": k, "embedding": embedding},
            )
        except Exception:
            logger.exception("Neo4j vector search failed index=%s", self.settings.vector_index_name)
            return []

        rows: list[dict[str, Any]] = []
        for record in records:
            similarity = float(record.get("similarity") or 0.0)
            if similarity < min_score:
                continue
            rows.append(self._to_row(record, similarity))
        logger.debug("Semantic retrieval returned %s chunks for top_k=%s", len(rows), k)
        return rows

    @staticmethod
    def _to_row(record: dict[str, Any], similarity: float) -> dict[str, Any]:
        section_title = record.get("section_title") or "Semantic match"
        return {
            "query_route": "semantic_vector",
            "statement_name": section_title,
            "seed_name": section_title,
            "related_labels": [CHUNK_LABEL],
            "source_chunk_text": record.get("source_chunk_text"),
            "evidence_text": None,
            "section_title": record.get("section_title"),
            "section_id": record.get("section_id"),
            "source_document": record.get("source_document"),
            "chunk_id": record.get("chunk_id"),
            "similarity": round(similarity, 4),
            # Scale cosine similarity (0..1) into the graph rows' integer-ish
            # scoring band so ranking stays comparable across both retrievers.
            "score": round(similarity * 100, 2),
            "semantic": True,
        }
