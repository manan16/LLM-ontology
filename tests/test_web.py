from __future__ import annotations

from typing import Any

import pytest

from web.app import app
from web.services import rag_service


@pytest.fixture(autouse=True)
def reset_rag_service_retriever(monkeypatch: Any) -> None:
    # Default the semantic layer to *empty* deterministic input (never the live
    # Neo4j/embeddings stack). This controls the input without mocking away the
    # unconfirmed policy — tests that exercise the semantic-only path install a
    # FakeSemanticRetriever carrying chunks via install_semantic().
    install_semantic(monkeypatch, [])
    rag_service.close_retriever()
    yield
    rag_service.close_retriever()


class FakeRetriever:
    last_query_plan = type(
        "Plan",
        (),
        {
            "category": "compliance_question",
            "intent": "obligations",
            "phrases": ["high-risk ai system"],
            "detected_actors": ["provider"],
            "detected_objects": ["high-risk ai system"],
            "detected_domains": ["EU AI Act"],
            "concept_groups": {"object": ["high-risk ai system"]},
            "ambiguous_terms": [],
            "expansion_terms": ["risk management"],
            "sub_questions": [],
        },
    )()
    last_expansion_debug = {"seeds": [], "expanded": [], "excluded": [], "selected": []}

    def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
        return [
            {
                "statement_name": "Provider Compliance",
                "statement_labels": ["Statement", "Obligation"],
                "evidence_text": "Providers of high-risk AI systems should ensure compliance with this Regulation.",
                "source_document": "eu_ai_act.pdf",
                "citation": "Article 16",
                "score": 42,
                "query_route": "statement",
                "source_group": "eu_ai_act",
                "seed_name": "Provider",
                "seed_labels": ["Entity"],
                "relationship": "HAS_REQUIREMENT",
                "matched_concept_groups": ["object:high-risk ai system"],
            }
        ]


class FakeAnswerGenerator:
    calls = 0

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def generate(self, question: str, context: str) -> str:
        type(self).calls += 1
        return "Grounded answer"


class FakeSemanticRetriever:
    """Deterministic stand-in for the vector-search retriever."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def retrieve(self, question: str, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


def semantic_chunk_row(index: int, text: str) -> dict[str, Any]:
    """A semantic (vector-search) row as produced by SemanticRetriever."""
    similarity = round(0.9 - index * 0.05, 4)
    return {
        "query_route": "semantic_vector",
        "statement_name": f"Semantic Section {index}",
        "seed_name": f"Semantic Section {index}",
        "related_labels": ["SourceChunk"],
        "source_chunk_text": text,
        "evidence_text": None,
        "section_title": f"Semantic Section {index}",
        "section_id": f"sec-{index}",
        "source_document": "eu_ai_act.pdf",
        "chunk_id": f"chunk-{index}",
        "similarity": similarity,
        "score": round(similarity * 100, 2),
        "semantic": True,
    }


def install_semantic(monkeypatch: Any, rows: list[dict[str, Any]]) -> None:
    monkeypatch.setattr(rag_service, "_get_semantic_retriever", lambda: FakeSemanticRetriever(rows))


def test_answer_question_returns_answer_evidence_context_and_debug(monkeypatch: Any) -> None:
    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", FakeRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", FakeAnswerGenerator)

    result = rag_service.answer_question(
        "What obligations apply to providers of high-risk AI systems?",
        show_context=True,
        debug=True,
    )

    assert result["answer"] == "Grounded answer"
    assert result["confidence"] == "confirmed"
    assert result["evidence_type"] == "graph_confirmed"
    assert result["evidence"][0]["statement"] == "Provider Compliance"
    assert result["evidence"][0]["evidence_type"] == "graph_confirmed"
    assert "Providers of high-risk AI systems" in result["context"]
    assert result["debug"]["query_plan"]["intent"] == "obligations"
    assert result["debug"]["top_rows"][0]["score"] == 42
    assert result["graph"]["nodes"]
    assert result["graph"]["edges"]
    assert any(node["label"] == "eu_ai_act.pdf" for node in result["graph"]["nodes"])
    assert any(node["type"] == "evidence" and node.get("evidence_id") == "1" for node in result["graph"]["nodes"])
    assert any(edge["relationship_type"] == "HAS_REQUIREMENT" for edge in result["graph"]["edges"])
    assert any(edge["label"] == "ranked_for_question" for edge in result["graph"]["edges"])
    assert result["metrics"]["top_k"] == 30
    assert result["metrics"]["rows_retrieved"] == 1
    assert result["metrics"]["evidence_items"] == 1
    assert result["metrics"]["source_coverage"]["eu_ai_act"] >= 1
    assert {"routing_ms", "retrieval_ms", "graph_query_ms", "ranking_ms", "generation_ms", "total_ms"}.issubset(
        result["metrics"]
    )


def test_answer_question_reuses_single_graph_retriever(monkeypatch: Any) -> None:
    class CountingRetriever(FakeRetriever):
        instances = 0

        def __init__(self) -> None:
            type(self).instances += 1

    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", CountingRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", FakeAnswerGenerator)

    first = rag_service.answer_question("What must providers document?")
    second = rag_service.answer_question("Which GDPR rights apply?")

    assert first["answer"] == "Grounded answer"
    assert second["answer"] == "Grounded answer"
    assert CountingRetriever.instances == 1
    assert rag_service._get_retriever() is rag_service._get_retriever()


def test_no_evidence_when_graph_and_semantic_both_empty(monkeypatch: Any) -> None:
    """Genuine no-evidence: neither graph nor semantic retrieval found anything."""

    class EmptyRetriever(FakeRetriever):
        def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
            return []

    class CountingAnswerGenerator(FakeAnswerGenerator):
        calls = 0

    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", EmptyRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", CountingAnswerGenerator)
    install_semantic(monkeypatch, [])

    result = rag_service.answer_question("What safeguards are needed for a new concept?", debug=True)

    assert result["answer"] == rag_service.NO_EVIDENCE_MESSAGE
    assert result["evidence"] == []
    assert result["response_source"] == "no_evidence"
    assert result["confidence"] == "none"
    assert result["evidence_type"] == "none"
    assert result["metrics"]["evidence_items"] == 0
    assert result["metrics"]["no_evidence"] is True
    assert result["metrics"]["generation_ms"] == 0
    assert result["graph"]["meta"]["source"] == "no_evidence"
    assert len(result["graph"]["nodes"]) == 1
    assert result["graph"]["edges"] == []
    assert CountingAnswerGenerator.calls == 0


def test_graph_empty_with_semantic_chunks_returns_unconfirmed(monkeypatch: Any) -> None:
    """Graph returns nothing, semantic returns 3 chunks -> unconfirmed/semantic_only.

    The chunks must survive into the response as context, but no confident
    determination/verdict may be produced from them.
    """

    class EmptyRetriever(FakeRetriever):
        def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
            return []

    def _determination_must_not_run(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError("determination LLM must not run for semantic-only evidence")

    chunks = [
        semantic_chunk_row(1, "A new concept related to safeguards for emerging AI features."),
        semantic_chunk_row(2, "Possibly relevant discussion of monitoring for novel systems."),
        semantic_chunk_row(3, "Additional similar passage about oversight of new capabilities."),
    ]

    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", EmptyRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(rag_service, "DeterminationGenerator", _determination_must_not_run)
    install_semantic(monkeypatch, chunks)

    result = rag_service.answer_question("What safeguards are needed for a new concept?", show_context=True)

    # Distinct status, not a stamped verdict.
    assert result["response_source"] == "semantic_only"
    assert result["confidence"] == "unconfirmed"
    assert result["evidence_type"] == "semantic_only"
    assert result["determination"]["verdict"] == "unconfirmed"
    assert result["determination"]["obligations"] == []

    # Semantic hits are still present for transparency (not silently dropped).
    assert len(result["evidence"]) == 3
    assert all(item["evidence_type"] == "semantic_only" for item in result["evidence"])
    assert "A new concept related to safeguards" in result["evidence"][0]["evidence_text"]

    # Context is rendered with the distinct unconfirmed Metadata banner.
    assert "[Metadata]" in result["context"]
    assert "semantic_only" in result["context"]
    assert "unconfirmed" in result["context"]


def test_answer_question_rejects_empty_question() -> None:
    try:
        rag_service.answer_question("")
    except ValueError as exc:
        assert "Please enter" in str(exc)
    else:
        raise AssertionError("empty question should raise ValueError")


def test_flask_ask_route_returns_clean_json(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "web.app.answer_question",
        lambda **kwargs: {
            "question": kwargs["question"],
            "answer": "Grounded answer",
            "evidence": [],
            "graph": {"nodes": [], "edges": []},
            "context": "",
            "debug": {},
            "metrics": {"top_k": 30},
            "response_source": "live",
        },
    )

    client = app.test_client()
    response = client.post("/ask", json={"question": "What is required?"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"] == "Grounded answer"
    assert data["graph"] == {"nodes": [], "edges": []}
    assert data["metrics"]["top_k"] == 30
    assert data["response_source"] == "live"
    assert data["error"] is None


def _hybrid_production_result(**kwargs: Any) -> dict[str, Any]:
    # The top-level answer/determination are always the hybrid-gated production
    # result, regardless of the requested retrieval mode.
    return {
        "question": kwargs.get("question", ""),
        "answer": "Hybrid production answer",
        "evidence": [{"id": 1, "statement": "S", "evidence_text": "E", "evidence_type": "graph_confirmed"}],
        "graph": {"nodes": [], "edges": []},
        "context": "",
        "debug": {"top_rows": []},
        "determination": {"verdict": "obligations_apply", "obligations": [], "summary": "x"},
        "confidence": "confirmed",
        "evidence_type": "graph_confirmed",
        "metrics": {"top_k": 30},
        "response_source": "live",
    }


def test_ask_route_accepts_each_mode_and_echoes_it_in_debug(monkeypatch: Any) -> None:
    from rag.retrieval_service import RetrievalResult

    monkeypatch.setattr("web.app.answer_question", _hybrid_production_result)

    called_modes: list[str] = []

    def fake_retrieval(query: str, mode: str, top_k: int = 10, debug: bool = False) -> RetrievalResult:
        called_modes.append(mode)
        return RetrievalResult(
            mode=mode,
            query=query,
            rows=[{"statement_name": "X", "source_chunk_text": "txt", "semantic": mode == "semantic"}],
            top_k=top_k,
            timings_ms={"graph_ms": 1},
            debug={"mode": mode, "row_count": 1},
        )

    monkeypatch.setattr("web.app.retrieval_retrieve", fake_retrieval)

    client = app.test_client()
    for mode in ("semantic", "graph", "hybrid"):
        response = client.post(
            "/ask", json={"question": "What must providers do?", "mode": mode, "debug": True}
        )
        assert response.status_code == 200, mode
        data = response.get_json()

        # Mode echoed back in the debug field (and top-level).
        assert data["debug"]["mode"] == mode
        assert data["mode"] == mode

        # Top-level answer/determination stay hybrid-gated regardless of mode.
        assert data["answer"] == "Hybrid production answer"
        assert data["determination"]["verdict"] == "obligations_apply"

        # Ablation modes carry a comparison view; hybrid does not re-run retrieval.
        if mode == "hybrid":
            assert "comparison" not in data["debug"]
        else:
            assert data["debug"]["comparison"]["mode"] == mode
            assert data["debug"]["comparison"]["row_count"] == 1

    # retrieval_service was only invoked for the two ablation modes.
    assert called_modes == ["semantic", "graph"]


def test_ask_route_rejects_invalid_mode(monkeypatch: Any) -> None:
    # answer_question must not even run for an invalid mode.
    def _must_not_run(**kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError("answer_question must not run for an invalid mode")

    monkeypatch.setattr("web.app.answer_question", _must_not_run)

    client = app.test_client()
    response = client.post("/ask", json={"question": "q", "mode": "lexical"})

    assert response.status_code == 400
    data = response.get_json()
    assert "mode" in (data.get("error") or "").lower()


def test_flask_index_renders_dashboard() -> None:
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Compliance-Aware RAG Assistant" in response.data
    assert b"Live backend" in response.data
    assert b"Research demo" in response.data
    assert b"Responses are generated from retrieved regulatory text and are not legal advice." in response.data
    assert b"rail-icon" not in response.data
    assert b"Workspace navigation" not in response.data


def test_flask_ask_route_handles_empty_question() -> None:
    client = app.test_client()
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]
    assert data["answer"] == ""
    assert data["graph"] == {"nodes": [], "edges": []}
    assert data["metrics"] == {}


def test_health_route_reports_neo4j_content_counts(monkeypatch: Any) -> None:
    class FakeRecord(dict):
        pass

    class FakeSession:
        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def run(self, query: str) -> Any:
            counts = {
                "MATCH (n:Regulation) RETURN count(n) AS count": 3,
                "MATCH (n:Statement) RETURN count(n) AS count": 12,
                "MATCH (n:SourceChunk) RETURN count(n) AS count": 40,
            }

            class Result:
                def single(self) -> FakeRecord:
                    return FakeRecord({"count": counts.get(query, 1)})

            return Result()

    class FakeDriver:
        def session(self, database: str) -> FakeSession:
            return FakeSession()

        def close(self) -> None:
            return None

    class FakeGraphDatabase:
        @staticmethod
        def driver(*_args: Any, **_kwargs: Any) -> FakeDriver:
            return FakeDriver()

    monkeypatch.setattr("web.app.GraphDatabase", FakeGraphDatabase)

    client = app.test_client()
    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["neo4j"]["status"] == "ok"
    assert data["neo4j"]["counts"] == {
        "regulations": 3,
        "statements": 12,
        "source_chunks": 40,
    }