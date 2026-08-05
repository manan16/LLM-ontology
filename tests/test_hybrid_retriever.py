from __future__ import annotations

from typing import Any

from app.config import get_settings
from rag.hybrid_retriever import HybridRetriever


# Single source of truth for the default weights: app/config.py's Settings.
_SETTINGS = get_settings()
DEFAULT_SEMANTIC_WEIGHT = _SETTINGS.hybrid_semantic_weight
DEFAULT_GRAPH_WEIGHT = _SETTINGS.hybrid_graph_weight


class FakeGraphRetriever:
    """Graph retriever stub; optionally carries a neo4j_client for expansion."""

    def __init__(self, rows: list[dict[str, Any]], neo4j_client: Any | None = None) -> None:
        self._rows = rows
        if neo4j_client is not None:
            self.neo4j_client = neo4j_client

    def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


class FakeSemanticRetriever:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def retrieve(self, question: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


def graph_row(name: str, score: float, evidence: str) -> dict[str, Any]:
    return {
        "statement_name": name,
        "statement_labels": ["Statement", "Obligation"],
        "evidence_text": evidence,
        "source_document": "eu_ai_act.pdf",
        "score": score,
        "query_route": "statement",
        "_retriever_ranked": True,
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


def _make_hybrid(graph_rows: list[dict[str, Any]], semantic_rows: list[dict[str, Any]]) -> HybridRetriever:
    return HybridRetriever(
        graph_retriever=FakeGraphRetriever(graph_rows),
        semantic_provider=lambda: FakeSemanticRetriever(semantic_rows),
    )


def test_merge_appends_semantic_after_graph_and_dedupes() -> None:
    graph_rows = [graph_row("Provider Compliance", 40, "Providers shall establish a risk-management system.")]
    semantic_rows = [
        semantic_row(1, 0.9, "A novel passage about oversight of new AI features."),
        # duplicate of the graph evidence text -> must be dropped
        semantic_row(2, 0.8, "Providers shall establish a risk-management system."),
    ]
    hybrid = _make_hybrid(graph_rows, semantic_rows)

    result = hybrid.retrieve("what applies to providers?", limit=30)

    assert result.semantic_added == 1
    assert [row.get("statement_name") for row in result.rows] == [
        "Provider Compliance",
        "Semantic Section 1",
    ]
    # Graph row stays first (priority); expansion is empty without a neo4j_client.
    assert result.rows[0].get("semantic") is None
    assert result.graph_expansion == {}


def test_rank_orders_by_weighted_hybrid_score() -> None:
    # Graph scores 40/20 -> normalised 1.0/0.5; semantic sims 0.9/0.5.
    # Default graph-favored weights: hybrid = 0.4*sem + 0.6*graph_norm:
    #   Graph High  = 0.6*1.0 = 0.60
    #   Semantic 1  = 0.4*0.9 = 0.36
    #   Graph Low   = 0.6*0.5 = 0.30
    #   Semantic 2  = 0.4*0.5 = 0.20
    # -> graph-confirmed evidence leads; a strong semantic passage (0.36) still
    #    outranks the weak graph row (0.30).
    graph_rows = [
        graph_row("Graph High", 40, "High graph evidence."),
        graph_row("Graph Low", 20, "Low graph evidence."),
    ]
    semantic_rows = [
        semantic_row(1, 0.9, "Strong semantic passage."),
        semantic_row(2, 0.5, "Weak semantic passage."),
    ]
    hybrid = _make_hybrid(graph_rows, semantic_rows)
    result = hybrid.retrieve("q", limit=30)

    ranked = hybrid.rank_rows(result.rows)
    order = [row.get("statement_name") for row in ranked]
    assert order == ["Graph High", "Semantic Section 1", "Graph Low", "Semantic Section 2"]

    top = ranked[0]
    assert top["statement_name"] == "Graph High"
    assert top["semantic_score"] == 0.0
    assert top["graph_relevance_score"] == 1.0
    assert top["hybrid_score"] == round(DEFAULT_GRAPH_WEIGHT * 1.0, 6)

    semantic_one = next(row for row in ranked if row["statement_name"] == "Semantic Section 1")
    assert semantic_one["semantic_score"] == 0.9
    assert semantic_one["graph_relevance_score"] == 0.0
    assert semantic_one["hybrid_score"] == round(DEFAULT_SEMANTIC_WEIGHT * 0.9, 6)


def test_rank_preserves_relative_graph_order() -> None:
    graph_rows = [
        graph_row("Graph High", 40, "High graph evidence."),
        graph_row("Graph Mid", 30, "Mid graph evidence."),
        graph_row("Graph Low", 10, "Low graph evidence."),
    ]
    hybrid = _make_hybrid(graph_rows, [])
    ranked = hybrid.rank_rows(hybrid.retrieve("q").rows)
    assert [row["statement_name"] for row in ranked] == ["Graph High", "Graph Mid", "Graph Low"]


def test_rank_weights_are_configurable() -> None:
    graph_rows = [graph_row("Graph High", 40, "High graph evidence.")]
    semantic_rows = [semantic_row(1, 0.9, "Strong semantic passage.")]
    hybrid = _make_hybrid(graph_rows, semantic_rows)
    result = hybrid.retrieve("q")

    # Graph-only weighting (1.0/0.0) must rank the graph row first.
    ranked = hybrid.rank_rows(result.rows, weights=(0.0, 1.0))
    assert ranked[0]["statement_name"] == "Graph High"
    assert ranked[0]["hybrid_score"] == 1.0
    assert ranked[1]["hybrid_score"] == 0.0


def test_semantic_unavailable_falls_back_to_graph_only() -> None:
    graph_rows = [graph_row("Provider Compliance", 40, "Providers shall comply.")]
    hybrid = HybridRetriever(
        graph_retriever=FakeGraphRetriever(graph_rows),
        semantic_provider=lambda: None,
    )
    result = hybrid.retrieve("q")
    assert result.semantic_added == 0
    assert [row["statement_name"] for row in result.rows] == ["Provider Compliance"]


def test_semantic_retrieval_error_degrades_gracefully() -> None:
    class BrokenSemantic:
        def retrieve(self, question: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("boom")

    graph_rows = [graph_row("Provider Compliance", 40, "Providers shall comply.")]
    hybrid = HybridRetriever(
        graph_retriever=FakeGraphRetriever(graph_rows),
        semantic_provider=lambda: BrokenSemantic(),
    )
    result = hybrid.retrieve("q")
    assert result.semantic_added == 0
    assert len(result.rows) == 1
