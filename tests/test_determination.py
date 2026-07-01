from __future__ import annotations

from typing import Any

import pytest

from rag import determination as determination_module
from rag.determination import (
    DeterminationGenerator,
    VERDICT_INSUFFICIENT,
    VERDICT_UNAVAILABLE,
)
from web.app import app as flask_app
from web.services import rag_service


# ---------------------------------------------------------------------------
# DeterminationGenerator — structured second pass over retrieved evidence.
# ---------------------------------------------------------------------------


class FakeOllamaClient:
    def __init__(self, payload: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.model = "test-model"
        self.calls: list[dict[str, Any]] = []

    def chat_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"system": system_prompt, "user": user_prompt, "schema": schema})
        if self.error is not None:
            raise self.error
        assert self.payload is not None
        return self.payload


def test_determination_is_derived_from_evidence_and_answer() -> None:
    client = FakeOllamaClient(
        payload={
            "verdict": "obligations_apply",
            "obligations": ["Establish a risk-management system [E1]."],
            "summary": "Providers must satisfy lifecycle obligations.",
        }
    )
    gen = DeterminationGenerator(ollama_client=client)

    result = gen.generate(
        question="What obligations apply to providers of high-risk AI systems?",
        context="[E1] Risk management system ... eu_ai_act.pdf",
        answer="Providers must establish a risk-management system [E1].",
    )

    assert result == {
        "verdict": "obligations_apply",
        "obligations": ["Establish a risk-management system [E1]."],
        "summary": "Providers must satisfy lifecycle obligations.",
    }
    # The prompt must carry the retrieved evidence and the grounded answer.
    assert "Risk management system" in client.calls[0]["user"]
    assert "Providers must establish" in client.calls[0]["user"]


def test_empty_context_returns_insufficient_without_calling_llm() -> None:
    client = FakeOllamaClient(payload={"verdict": "obligations_apply", "obligations": [], "summary": "x"})
    gen = DeterminationGenerator(ollama_client=client)

    result = gen.generate(question="anything", context="   ", answer="")

    assert result["verdict"] == VERDICT_INSUFFICIENT
    assert result["obligations"] == []
    assert client.calls == []  # LLM not invoked when there is no evidence


def test_llm_failure_is_logged_not_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeOllamaClient(error=RuntimeError("ollama down"))
    gen = DeterminationGenerator(ollama_client=client)

    with caplog.at_level("ERROR"):
        result = gen.generate(question="q", context="[E1] evidence", answer="answer")

    assert result["verdict"] == VERDICT_UNAVAILABLE
    assert result["obligations"] == []
    assert any("Determination synthesis failed" in record.message for record in caplog.records)


def test_out_of_enum_verdict_is_coerced_to_insufficient() -> None:
    client = FakeOllamaClient(
        payload={"verdict": "definitely_fine", "obligations": ["should be dropped"], "summary": "s"}
    )
    gen = DeterminationGenerator(ollama_client=client)

    result = gen.generate(question="q", context="[E1] evidence", answer="answer")

    assert result["verdict"] == VERDICT_INSUFFICIENT
    assert result["obligations"] == []  # insufficient verdict never carries obligations


def test_insufficient_verdict_from_model_drops_obligations() -> None:
    client = FakeOllamaClient(
        payload={"verdict": "insufficient_evidence", "obligations": ["leaked"], "summary": "not enough"}
    )
    gen = DeterminationGenerator(ollama_client=client)

    result = gen.generate(question="q", context="[E1] evidence", answer="answer")

    assert result == {"verdict": "insufficient_evidence", "obligations": [], "summary": "not enough"}


# ---------------------------------------------------------------------------
# rag_service.answer_question — determination is part of the contract.
# ---------------------------------------------------------------------------


class FakeRetriever:
    last_query_plan = None
    last_expansion_debug = {"seeds": [], "expanded": [], "excluded": [], "selected": []}

    def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
        return [
            {
                "statement_name": "Provider Compliance",
                "statement_labels": ["Statement", "Obligation"],
                "evidence_text": "Providers of high-risk AI systems shall establish a risk-management system.",
                "source_document": "eu_ai_act.pdf",
                "citation": "Article 9",
                "score": 42,
                "query_route": "statement",
                "source_group": "eu_ai_act",
            }
        ]


class EmptyRetriever(FakeRetriever):
    def retrieve(self, question: str, limit: int = 30) -> list[dict[str, Any]]:
        return []


class FakeAnswerGenerator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def generate(self, question: str, context: str) -> str:
        return "Grounded answer [E1]."


class FakeDeterminationGenerator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def generate(self, question: str, context: str, answer: str) -> dict[str, Any]:
        return {
            "verdict": "obligations_apply",
            "obligations": ["Establish a risk-management system [E1]."],
            "summary": "Providers must satisfy obligations.",
        }


@pytest.fixture(autouse=True)
def reset_retriever() -> Any:
    rag_service.close_retriever()
    yield
    rag_service.close_retriever()


def test_answer_question_includes_backend_determination(monkeypatch: Any) -> None:
    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", FakeRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(rag_service, "DeterminationGenerator", FakeDeterminationGenerator)

    result = rag_service.answer_question("What obligations apply to providers?", show_context=True)

    assert result["determination"]["verdict"] == "obligations_apply"
    assert result["determination"]["obligations"] == ["Establish a risk-management system [E1]."]
    assert result["determination"]["summary"]
    assert "determination_ms" in result["metrics"] or "total_ms" in result["metrics"]


def test_no_evidence_returns_insufficient_determination(monkeypatch: Any) -> None:
    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", EmptyRetriever)

    def _should_not_run(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - guard
        raise AssertionError("determination LLM must not run when there is no evidence")

    monkeypatch.setattr(rag_service, "DeterminationGenerator", _should_not_run)

    result = rag_service.answer_question("totally unrelated question", show_context=True)

    assert result["response_source"] == "no_evidence"
    assert result["determination"]["verdict"] == VERDICT_INSUFFICIENT
    assert result["determination"]["obligations"] == []


# ---------------------------------------------------------------------------
# /ask route surfaces the determination field.
# ---------------------------------------------------------------------------


def test_ask_route_surfaces_determination(monkeypatch: Any) -> None:
    monkeypatch.setattr(rag_service, "_retriever", None)
    monkeypatch.setattr(rag_service, "GraphRetriever", FakeRetriever)
    monkeypatch.setattr(rag_service, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(rag_service, "DeterminationGenerator", FakeDeterminationGenerator)

    client = flask_app.test_client()
    response = client.post("/ask", json={"question": "What obligations apply to providers?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["determination"]["verdict"] == "obligations_apply"
    assert body["determination"]["obligations"]
    assert body["error"] is None
