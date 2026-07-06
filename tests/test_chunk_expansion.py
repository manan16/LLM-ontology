from __future__ import annotations

from typing import Any

from graph.neo4j_client import (
    ENTITY_EXPANSION_RELATIONSHIPS,
    Neo4jClient,
    assemble_chunk_expansion,
    build_chunk_expansion_query,
)
from rag.retriever import expand_chunk_entities


class FixtureGraphClient:
    """In-memory stand-in for Neo4j over a tiny fixture graph.

    Models the real chunk->entity path used by the expansion query:
    (:SourceChunk)<-[:REFERENCES]-(:Statement)-[link]->(:Entity), then one or
    more hops of Entity-to-Entity relationships. ``run_query`` reproduces the
    row contract of build_chunk_expansion_query so the *real* assembly logic in
    Neo4jClient.expand_chunk_entities is exercised end to end.
    """

    def __init__(self) -> None:
        # chunk_id -> entities directly linked through its statements.
        self.chunk_entities: dict[str, list[str]] = {
            "chunk-1": ["provider", "high_risk_ai_system"],
            "chunk-empty": [],
        }
        self.entity_names = {
            "provider": "Provider",
            "high_risk_ai_system": "High-Risk AI System",
            "risk_management_system": "Risk Management System",
        }
        self.entity_types = {
            "provider": "Provider",
            "high_risk_ai_system": "HighRiskAISystem",
            "risk_management_system": "Control",
        }
        # entity_key -> list of (predicate, object_key) outgoing entity edges.
        self.edges: dict[str, list[tuple[str, str]]] = {
            "provider": [("APPLIES_TO", "high_risk_ai_system")],
            "high_risk_ai_system": [("RELATED_TO", "risk_management_system")],
        }

    # -- expansion delegates to the production logic under test --
    def expand_chunk_entities(self, chunk_ids: list[str], max_hops: int = 1) -> dict[str, Any]:
        return Neo4jClient.expand_chunk_entities(self, chunk_ids, max_hops=max_hops)

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = parameters or {}
        max_hops = int(params.get("max_hops", 1))
        rows: list[dict[str, Any]] = []
        for chunk_id in params.get("chunk_ids", []):
            for anchor in self.chunk_entities.get(chunk_id, []):
                paths = self._paths_from(anchor, max_hops)
                if not paths:
                    rows.append(self._row(chunk_id, anchor, None))
                    continue
                for path in paths:
                    for subject_key, predicate, object_key in path:
                        rows.append(self._row(chunk_id, anchor, (subject_key, predicate, object_key)))
        return rows

    def _paths_from(self, anchor: str, max_hops: int) -> list[list[tuple[str, str, str]]]:
        paths: list[list[tuple[str, str, str]]] = []

        def dfs(node: str, path: list[tuple[str, str, str]]) -> None:
            if path:
                paths.append(list(path))
            if len(path) >= max_hops:
                return
            for predicate, obj in self.edges.get(node, []):
                path.append((node, predicate, obj))
                dfs(obj, path)
                path.pop()

        dfs(anchor, [])
        return paths

    def _row(self, chunk_id: str, anchor: str, rel: tuple[str, str, str] | None) -> dict[str, Any]:
        row = {
            "chunk_id": chunk_id,
            "entity_key": anchor,
            "entity_name": self.entity_names.get(anchor),
            "entity_type": self.entity_types.get(anchor),
            "subject_key": None,
            "subject_name": None,
            "predicate": None,
            "object_key": None,
            "object_name": None,
        }
        if rel is not None:
            subject_key, predicate, object_key = rel
            row.update(
                {
                    "subject_key": subject_key,
                    "subject_name": self.entity_names.get(subject_key),
                    "predicate": predicate,
                    "object_key": object_key,
                    "object_name": self.entity_names.get(object_key),
                }
            )
        return row


def test_build_query_targets_real_schema_not_assumed_labels() -> None:
    query = build_chunk_expansion_query(max_hops=1)
    # Real model uses SourceChunk + REFERENCES, not the task's assumed :Chunk/:MENTIONS.
    assert "SourceChunk" in query
    assert "REFERENCES" in query
    assert "MENTIONS" not in query
    for rel_type in ENTITY_EXPANSION_RELATIONSHIPS:
        assert rel_type in query


def test_expand_returns_expected_triples_for_known_chunk() -> None:
    client = FixtureGraphClient()

    result = client.expand_chunk_entities(["chunk-1"], max_hops=1)

    assert set(result) == {"chunk-1"}
    entities = {entity["name"] for entity in result["chunk-1"]["entities"]}
    assert entities == {"Provider", "High-Risk AI System"}

    triples = set(result["chunk-1"]["triples"])
    assert triples == {
        ("Provider", "APPLIES_TO", "High-Risk AI System"),
        ("High-Risk AI System", "RELATED_TO", "Risk Management System"),
    }


def test_expand_via_retriever_entry_point_and_empty_chunk() -> None:
    client = FixtureGraphClient()

    result = expand_chunk_entities(["chunk-1", "chunk-empty"], neo4j_client=client)

    # Every requested chunk id is present, even when it has no linked entities.
    assert set(result) == {"chunk-1", "chunk-empty"}
    assert result["chunk-empty"] == {"entities": [], "triples": []}
    assert ("Provider", "APPLIES_TO", "High-Risk AI System") in result["chunk-1"]["triples"]


def test_expand_empty_input_returns_empty_dict() -> None:
    client = FixtureGraphClient()
    assert client.expand_chunk_entities([]) == {}


def test_assemble_dedupes_entities_and_triples() -> None:
    rows = [
        {
            "chunk_id": "c1",
            "entity_key": "provider",
            "entity_name": "Provider",
            "entity_type": "Provider",
            "subject_name": "Provider",
            "predicate": "APPLIES_TO",
            "object_name": "High-Risk AI System",
        },
        # duplicate entity + duplicate triple should collapse
        {
            "chunk_id": "c1",
            "entity_key": "provider",
            "entity_name": "Provider",
            "entity_type": "Provider",
            "subject_name": "Provider",
            "predicate": "APPLIES_TO",
            "object_name": "High-Risk AI System",
        },
    ]

    result = assemble_chunk_expansion(["c1"], rows)

    assert len(result["c1"]["entities"]) == 1
    assert result["c1"]["triples"] == [("Provider", "APPLIES_TO", "High-Risk AI System")]
