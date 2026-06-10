from __future__ import annotations

from typing import Any

import pytest

from web.app import app
from web.services import rag_service


@pytest.fixture(autouse=True)
def reset_rag_service_retriever() -> None:
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
    assert result["evidence"][0]["statement"] == "Provider Compliance"
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


def test_answer_question_returns_no_evidence_state_without_generation(monkeypatch: Any) -> None:
    class EmptyRetriever(FakeRetriever):
        def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
            return []

    class CountingAnswerGenerator(FakeAnswerGenerator):
        calls = 0

    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", EmptyRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", CountingAnswerGenerator)

    result = rag_service.answer_question("What safeguards are needed for a new concept?", debug=True)

    assert result["answer"] == rag_service.NO_EVIDENCE_MESSAGE
    assert result["evidence"] == []
    assert result["response_source"] == "no_evidence"
    assert result["metrics"]["evidence_items"] == 0
    assert result["metrics"]["no_evidence"] is True
    assert result["metrics"]["generation_ms"] == 0
    assert result["graph"]["meta"]["source"] == "no_evidence"
    assert len(result["graph"]["nodes"]) == 1
    assert result["graph"]["edges"] == []
    assert CountingAnswerGenerator.calls == 0


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


def test_flask_index_renders_dashboard() -> None:
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"Complaince Aware RAG Assistant" in response.data
    assert b"Live backend mode" in response.data
    assert b"Preset cache disabled" in response.data
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
