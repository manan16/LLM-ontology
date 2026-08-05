from __future__ import annotations

from typing import Any

import pytest

from rag.retrieval_service import RetrievalResult, RetrievalService


# ---------------------------------------------------------------------------
# Spies / fakes. The embedding model lives inside the semantic retriever; the
# graph database (Neo4j) lives inside the graph retriever's neo4j_client. The
# spies let us assert which subsystems a given mode does — and does NOT — touch.
# ---------------------------------------------------------------------------
class SpyEmbedding:
    def __init__(self) -> None:
        self.embed_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.embed_calls += 1
        return [0.1, 0.2, 0.3]


class SpyNeo4j:
    """Stand-in for Neo4jClient used for graph traversal + chunk expansion."""

    def __init__(self) -> None:
        self.query_calls = 0

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.query_calls += 1
        return []

    def expand_chunk_entities(self, chunk_ids: list[str], max_hops: int = 1) -> dict[str, Any]:
        self.query_calls += 1
        return {}


class FakeSemanticRetriever:
    """Semantic retriever whose retrieve() exercises the (spied) embedding model."""

    def __init__(self, rows: list[dict[str, Any]], embedding: SpyEmbedding | None = None) -> None:
        self._rows = rows
        self.embedding = embedding or SpyEmbedding()
        self.retrieve_calls = 0

    def retrieve(self, query: str, top_k: int | None = None, min_score: float = 0.0) -> list[dict[str, Any]]:
        self.retrieve_calls += 1
        self.embedding.embed_query(query)  # dense search always embeds the query
        return [dict(row) for row in self._rows]


class FakeGraphRetriever:
    """Graph retriever whose retrieve() would go through the (spied) Neo4j client."""

    def __init__(self, rows: list[dict[str, Any]], neo4j_client: SpyNeo4j) -> None:
        self._rows = rows
        self.neo4j_client = neo4j_client
        self.retrieve_calls = 0
        self.last_query_plan = None

    def retrieve(self, question: str, limit: int = 30, conversation_history: Any = None) -> list[dict[str, Any]]:
        self.retrieve_calls += 1
        self.neo4j_client.run_query("GRAPH_TRAVERSAL", {"q": question, "limit": limit})
        return [dict(row) for row in self._rows]


def graph_row(name: str, score: float, evidence: str) -> dict[str, Any]:
    return {
        "statement_name": name,
        "statement_labels": ["Statement", "Obligation"],
        "evidence_text": evidence,
        "source_document": "eu_ai_act.pdf",
        "score": score,
        "query_route": "statement",
    }


def semantic_row(index: int, similarity: float, text: str) -> dict[str, Any]:
    return {
        "statement_name": f"Semantic Section {index}",
        "related_labels": ["SourceChunk"],
        "source_chunk_text": text,
        "source_document": "eu_ai_act.pdf",
        "chunk_id": f"chunk-{index}",
        "similarity": similarity,
        "score": round(similarity * 100, 2),
        "query_route": "semantic_vector",
        "semantic": True,
    }


class Harness:
    """Bundles spies + a RetrievalService wired to fakes, tracking provider use."""

    def __init__(self, graph_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> None:
        self.neo4j = SpyNeo4j()
        self.embedding = SpyEmbedding()
        self.graph = FakeGraphRetriever(graph_rows, neo4j_client=self.neo4j)
        self.semantic = FakeSemanticRetriever(semantic_rows, embedding=self.embedding)
        self.graph_provider_calls = 0
        self.semantic_provider_calls = 0

        def graph_provider() -> Any:
            self.graph_provider_calls += 1
            return self.graph

        def semantic_provider() -> Any:
            self.semantic_provider_calls += 1
            return self.semantic

        self.service = RetrievalService(
            graph_provider=graph_provider,
            semantic_provider=semantic_provider,
        )


@pytest.fixture
def harness() -> Harness:
    return Harness(
        graph_rows=[
            graph_row("Provider Compliance", 40, "Providers shall establish a risk-management system."),
            graph_row("Deployer Duty", 20, "Deployers shall monitor operation."),
        ],
        semantic_rows=[
            semantic_row(1, 0.9, "A novel passage about oversight of new AI features."),
            semantic_row(2, 0.7, "Another possibly relevant passage on monitoring."),
        ],
    )


def _assert_result_shape(result: Any, mode: str, top_k: int) -> None:
    assert isinstance(result, RetrievalResult)
    assert result.mode == mode
    assert result.top_k == top_k
    assert isinstance(result.rows, list)
    assert all(isinstance(row, dict) for row in result.rows)
    assert isinstance(result.graph_expansion, dict)
    assert isinstance(result.timings_ms, dict)
    assert result.row_count == len(result.rows)


def test_semantic_mode_shape_and_never_queries_neo4j(harness: Harness) -> None:
    result = harness.service.retrieve("what applies to providers?", mode="semantic", top_k=5, debug=True)

    _assert_result_shape(result, "semantic", top_k=5)
    assert [row["statement_name"] for row in result.rows] == ["Semantic Section 1", "Semantic Section 2"]
    assert result.graph_expansion == {}
    assert "semantic_ms" in result.timings_ms
    assert result.debug["mode"] == "semantic"

    # Semantic mode used the semantic retriever (and thus the embedding model)...
    assert harness.semantic.retrieve_calls == 1
    assert harness.embedding.embed_calls == 1
    # ...but never constructed or used the graph retriever / Neo4j.
    assert harness.graph_provider_calls == 0
    assert harness.graph.retrieve_calls == 0
    assert harness.neo4j.query_calls == 0


def test_graph_mode_shape_and_never_calls_embedding_model(harness: Harness) -> None:
    result = harness.service.retrieve("what applies to providers?", mode="graph", top_k=8, debug=True)

    _assert_result_shape(result, "graph", top_k=8)
    assert [row["statement_name"] for row in result.rows] == ["Provider Compliance", "Deployer Duty"]
    assert result.graph_expansion == {}
    assert "graph_ms" in result.timings_ms
    assert result.debug["mode"] == "graph"

    # Graph mode used the graph retriever + Neo4j traversal...
    assert harness.graph.retrieve_calls == 1
    assert harness.neo4j.query_calls == 1
    # ...but never constructed the semantic retriever or called the embedding model.
    assert harness.semantic_provider_calls == 0
    assert harness.semantic.retrieve_calls == 0
    assert harness.embedding.embed_calls == 0


def test_hybrid_mode_shape_delegates_to_hybrid_retriever(harness: Harness) -> None:
    result = harness.service.retrieve("what applies to providers?", mode="hybrid", top_k=7, debug=True)

    _assert_result_shape(result, "hybrid", top_k=7)
    # Rows are returned in weighted-hybrid-score order (rank_rows), matching the
    # production ranking in rag_service.answer_question — NOT the raw graph-first /
    # semantic-appended merge order. With graph scores 40/20 (normalised 1.0/0.5),
    # semantic similarities 0.9/0.7, and default weights 0.4 semantic / 0.6 graph:
    #   Provider Compliance = 0.6*1.0 = 0.60
    #   Semantic Section 1  = 0.4*0.9 = 0.36
    #   Deployer Duty       = 0.6*0.5 = 0.30
    #   Semantic Section 2  = 0.4*0.7 = 0.28
    assert [row["statement_name"] for row in result.rows] == [
        "Provider Compliance",
        "Semantic Section 1",
        "Deployer Duty",
        "Semantic Section 2",
    ]
    # rank_rows attached the weighted score fields to every row.
    assert [row["hybrid_score"] for row in result.rows] == [0.6, 0.36, 0.3, 0.28]
    assert all(
        {"semantic_score", "graph_relevance_score", "hybrid_score"} <= row.keys()
        for row in result.rows
    )
    # Hybrid used both subsystems.
    assert harness.graph.retrieve_calls == 1
    assert harness.semantic.retrieve_calls == 1
    assert harness.embedding.embed_calls == 1
    assert result.timings_ms.get("semantic_added") == 2
    assert result.debug["mode"] == "hybrid"


def test_unknown_mode_raises() -> None:
    service = RetrievalService(
        graph_provider=lambda: None,
        semantic_provider=lambda: None,
    )
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        service.retrieve("q", mode="lexical")  # type: ignore[arg-type]


def test_graph_mode_on_vague_query_yields_empty_result_not_error() -> None:
    """Graph-only retrieval on a paraphrase with no matchable entities.

    This is an expected, reportable outcome (graph-only failing on paraphrase,
    contrasted with semantic/hybrid succeeding) — it must return a valid, empty
    RetrievalResult, never raise. Uses the *real* GraphRetriever + build_query_plan
    with a spy Neo4j client so no live database is required.
    """
    from rag.query_planner import build_query_plan
    from rag.retriever import GraphRetriever

    class SpyNeo4j:
        def __init__(self) -> None:
            self.query_calls = 0

        def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
            self.query_calls += 1
            return []

    vague = "What about the thing we were just discussing yesterday?"
    # Precondition: the planner extracts no matchable compliance entities/phrases.
    plan = build_query_plan(vague)
    assert not plan.phrases
    assert not plan.expansion_terms

    real_graph = GraphRetriever(neo4j_client=SpyNeo4j())
    service = RetrievalService(
        graph_provider=lambda: real_graph,
        semantic_provider=lambda: None,
    )

    result = service.retrieve(vague, mode="graph", top_k=10)

    _assert_result_shape(result, "graph", top_k=10)
    assert result.rows == []  # graph-only signals "nothing found" as an empty list
    assert result.graph_expansion == {}
    assert "graph_ms" in result.timings_ms


def test_uniform_result_shape_is_consumable_without_mode_branches(harness: Harness) -> None:
    # A downstream consumer only needs result.rows regardless of mode.
    for mode in ("semantic", "graph", "hybrid"):
        result = harness.service.retrieve("q", mode=mode, top_k=6)
        rows = result.rows  # what context_builder.build_context receives
        assert isinstance(rows, list)
        # Every mode yields the same row-dict shape (has a text field to render).
        for row in rows:
            assert "evidence_text" in row or "source_chunk_text" in row
