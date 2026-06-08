from __future__ import annotations

from typing import Any

from web.app import app
from web.services import rag_service


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
                "score": 42,
                "query_route": "statement",
                "source_group": "eu_ai_act",
                "matched_concept_groups": ["object:high-risk ai system"],
            }
        ]


class FakeAnswerGenerator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def generate(self, question: str, context: str) -> str:
        return "Grounded answer"


def test_answer_question_returns_answer_evidence_context_and_debug(monkeypatch: Any) -> None:
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
    assert result["metrics"]["top_k"] == 30
    assert result["metrics"]["rows_retrieved"] == 1
    assert result["metrics"]["evidence_items"] == 1
    assert result["metrics"]["source_coverage"]["eu_ai_act"] >= 1
    assert {"routing_ms", "retrieval_ms", "graph_query_ms", "ranking_ms", "generation_ms", "total_ms"}.issubset(
        result["metrics"]
    )


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
            "context": "",
            "debug": {},
            "metrics": {"top_k": 30},
        },
    )

    client = app.test_client()
    response = client.post("/ask", json={"question": "What is required?"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"] == "Grounded answer"
    assert data["metrics"]["top_k"] == 30
    assert data["error"] is None


def test_flask_index_renders_dashboard() -> None:
    client = app.test_client()
    response = client.get("/")

    assert response.status_code == 200
    assert b"TrustGraph Health" in response.data
    assert b"Evidence search active" in response.data


def test_flask_ask_route_handles_empty_question() -> None:
    client = app.test_client()
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 400
    data = response.get_json()
    assert data["error"]
    assert data["answer"] == ""
    assert data["metrics"] == {}
