from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ask import build_parser
from rag.answer_generator import AnswerGenerator
from rag.context_builder import build_context
from rag.prompts import RAG_PROMPT_TEMPLATE, build_rag_prompt
from rag.query_planner import build_query_plan
from rag.retriever import GraphRetriever, _COMMON_RETURN, _dedupe_rows, extract_query_terms


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, parameters or {}))
        if "MATCH (seed:Entity)" in query:
            return [
                {
                    "seed_name": "Covered Entity",
                    "seed_labels": ["Entity", "CoveredEntity"],
                    "relationship": "APPLIES_TO",
                    "related_name": "Minimum Necessary Requirement",
                    "related_labels": ["Statement", "Requirement"],
                    "statement_name": "Minimum Necessary Requirement",
                    "statement_labels": ["Statement", "Requirement"],
                    "evidence_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                    "citation": "45 CFR 164.502",
                    "section_title": "Uses and disclosures",
                    "page_number": 12,
                    "article_number": None,
                    "clause_number": None,
                    "source_document": "hipaa.pdf",
                    "source_chunk_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                }
            ]
        return [
            {
                "seed_name": "Minimum Necessary Requirement",
                "seed_labels": ["Statement", "Requirement"],
                "relationship": "REFERENCES",
                "related_name": "hipaa.pdf:sec-1:0",
                "related_labels": ["SourceChunk"],
                "statement_name": "Minimum Necessary Requirement",
                "statement_labels": ["Statement", "Requirement"],
                "evidence_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                "citation": "45 CFR 164.502",
                "source_document": "hipaa.pdf",
            }
        ]


class FakeOllamaClient:
    def __init__(self) -> None:
        self.model = "fake"
        self.calls: list[tuple[str, str]] = []

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "Grounded answer"


def test_extract_query_terms_keeps_domain_phrases_and_removes_stopwords() -> None:
    terms = extract_query_terms("When can a covered entity disclose PHI without authorization?")

    assert "covered entity" in terms
    assert "phi" in terms
    assert "authorization" in terms
    assert "when" not in terms
    assert "can" not in terms


def test_query_plan_categorizes_expands_and_decomposes_compliance_questions() -> None:
    plan = build_query_plan("What obligations do providers and deployers have for high-risk AI systems?")

    assert plan.category == "compliance_question"
    assert plan.intent == "obligations"
    assert plan.detected_actors == ["provider", "deployer"]
    assert "high-risk ai system" in plan.phrases
    assert "technical documentation" in plan.expansion_terms
    assert plan.needs_decomposition is True
    assert plan.sub_questions == [
        "what obligations do providers have for high-risk ai systems?",
        "what obligations do deployers have for high-risk ai systems?",
    ]


def test_query_plan_rewrites_follow_up_when_history_is_available() -> None:
    plan = build_query_plan(
        "What about providers?",
        ["What obligations apply under the EU AI Act for high-risk AI systems?"],
    )

    assert plan.category == "follow_up_question"
    assert plan.normalized_question == "what obligations apply to providers under the eu ai act?"
    assert plan.detected_actors == ["provider"]


def test_without_authorization_query_plan_prefers_permissions_and_exceptions() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")

    assert plan.intent in {"exceptions", "permissions"}
    assert plan.has_negated_authorization is True
    assert "Permission" in plan.preferred_statement_labels
    assert "Exception" in plan.preferred_statement_labels
    assert plan.preferred_relationships[0] == "HAS_EXCEPTION"
    assert "GRANTS_PERMISSION" in plan.preferred_relationships
    assert "REQUIRES_AUTHORIZATION" not in plan.preferred_relationships[:3]
    assert "required by law" in plan.expansion_terms
    assert "may disclose" in plan.expansion_terms


def test_without_authorization_ranking_prefers_permission_exception_rows() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")
    rows = [
        {
            "statement_name": "Authorization Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "A covered entity must obtain authorization for this disclosure.",
            "score": 30,
        },
        {
            "statement_name": "Disclosure Prohibition",
            "statement_labels": ["Statement", "Prohibition"],
            "relationship": "LIMITED_BY",
            "evidence_text": "A covered entity may not disclose protected health information.",
            "score": 28,
        },
        {
            "statement_name": "Permitted Disclosure Required By Law",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose protected health information without authorization when required by law.",
            "score": 10,
        },
        {
            "statement_name": "Authorization Exception Public Health",
            "statement_labels": ["Statement", "Exception"],
            "relationship": "HAS_EXCEPTION",
            "evidence_text": "Authorization or opportunity to agree or object is not required for public health activities.",
            "score": 10,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] in {
        "Permitted Disclosure Required By Law",
        "Authorization Exception Public Health",
    }
    assert ranked[1]["statement_name"] in {
        "Permitted Disclosure Required By Law",
        "Authorization Exception Public Health",
    }
    assert ranked[-1]["statement_name"] in {"Authorization Requirement", "Disclosure Prohibition"}


def test_without_authorization_downranks_with_authorization_rows() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")
    rows = [
        {
            "statement_name": "Permitted Disclosure with Authorization",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI pursuant to and in compliance with an authorization.",
            "score": 80,
        },
        {
            "statement_name": "Permitted Disclosure Public Health",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI without authorization for public health activities.",
            "score": 10,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Permitted Disclosure Public Health"
    assert ranked[-1]["statement_name"] == "Permitted Disclosure with Authorization"


def test_build_context_deduplicates_and_respects_max_chars() -> None:
    rows = [
        {
            "statement_name": "Disclose PHI",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "HAS_EXCEPTION",
            "related_name": "Authorization Exception",
            "evidence_text": "Covered entities may disclose PHI without authorization for treatment.",
            "citation": "45 CFR 164.506",
            "source_document": "hipaa.pdf",
        },
        {
            "statement_name": "Disclose PHI",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "HAS_EXCEPTION",
            "related_name": "Authorization Exception",
            "evidence_text": "Covered entities may disclose PHI without authorization for treatment.",
            "citation": "45 CFR 164.506",
            "source_document": "hipaa.pdf",
        },
        {
            "statement_name": "Very long item",
            "evidence_text": "x" * 500,
        },
    ]

    context = build_context(rows, max_chars=260)

    assert context.count("[1]") == 1
    assert "Covered entities may disclose PHI" in context
    assert "Very long item" not in context
    assert len(context) <= 260


def test_context_builder_prefers_permitted_disclosure_before_authorization_requirements() -> None:
    rows = [
        {
            "statement_name": "Authorization Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "A covered entity must obtain authorization for uses and disclosures.",
            "score": 100,
        },
        {
            "statement_name": "Permitted Disclosure Required By Law",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI without authorization when required by law.",
            "score": 20,
        },
    ]

    context = build_context(rows)

    assert context.index("Permitted Disclosure Required By Law") < context.index("Authorization Requirement")


def test_context_builder_groups_treatment_payment_operations_rows() -> None:
    rows = [
        {
            "statement_name": "Treatment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for treatment without authorization.",
            "score": 30,
        },
        {
            "statement_name": "Payment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for payment without authorization.",
            "score": 29,
        },
        {
            "statement_name": "Health Care Operations Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for health care operations without authorization.",
            "score": 28,
        },
        {
            "statement_name": "OHCA Disclosure",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "An organized health care arrangement may use or disclose PHI for operations.",
            "score": 27,
        },
    ]

    context = build_context(rows)

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "Statement: Treatment Permission" not in context
    assert "Statement: Payment Permission" not in context
    assert "Statement: Health Care Operations Permission" not in context
    assert "- OHCA Disclosure" in context


def test_context_builder_does_not_group_unrelated_rows_into_tpo() -> None:
    rows = [
        {
            "statement_name": "Treatment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for treatment without authorization.",
            "score": 40,
        },
        {
            "statement_name": "Payment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for payment without authorization.",
            "score": 39,
        },
        {
            "statement_name": "Emergency Disclosure - Directory",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "Emergency directory disclosure may be made when the individual has an opportunity to agree or object.",
            "score": 38,
        },
        {
            "statement_name": "Business Associate Disclosure",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "DISCLOSES_TO",
            "evidence_text": "A covered entity may disclose PHI to a business associate under a business associate agreement.",
            "score": 37,
        },
        {
            "statement_name": "Authorization for Research",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "Authorization for research-related treatment may be required unless an exception applies.",
            "score": 36,
        },
    ]

    context = build_context(rows)
    tpo_block = context.split("[2]")[0]

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "A covered entity may use or disclose PHI for treatment without authorization." in tpo_block
    assert "A covered entity may use or disclose PHI for payment without authorization." in tpo_block
    assert "Emergency Disclosure - Directory" in context
    assert "Business Associate Disclosure" in context
    assert "Authorization for Research" in context
    assert "Emergency directory disclosure" not in tpo_block
    assert "business associate" not in tpo_block.lower()
    assert "research-related treatment" not in tpo_block.lower()


def test_retriever_formats_and_deduplicates_mocked_neo4j_rows() -> None:
    client = FakeNeo4jClient()
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    rows = retriever.retrieve("What obligations apply to covered entities?", limit=10)

    assert len(client.calls) >= 4
    assert len(rows) == 1
    assert rows[0]["query_route"] == "exact_phrase_entity"
    assert rows[0]["score"] > 0
    assert rows[0]["seed_name"] == "Covered Entity"
    assert rows[0]["relationship"] == "APPLIES_TO"
    assert rows[0]["statement_name"] == "Minimum Necessary Requirement"
    assert rows[0]["evidence_text"] == "A covered entity must make reasonable efforts to limit PHI disclosure."
    assert rows[0]["citation"] == "45 CFR 164.502"


def test_answer_prompt_contains_grounding_and_no_invention_instructions() -> None:
    prompt = build_rag_prompt("Question?", "Context")

    assert "using only the retrieved knowledge graph context" in prompt
    assert "Do not invent legal requirements" in prompt
    assert "graph does not contain enough evidence" in prompt
    assert "Evidence used" in prompt
    assert "without authorization" in prompt
    assert "{question}" in RAG_PROMPT_TEMPLATE


def test_retriever_citation_expression_uses_citation_arrays_only() -> None:
    assert re.search(r"statement\.citation(?!s)", _COMMON_RETURN) is None
    assert re.search(r"related\.citation(?!s)", _COMMON_RETURN) is None
    assert "head(coalesce(statement.citations, []))" in _COMMON_RETURN
    assert "head(coalesce(related.citations, []))" in _COMMON_RETURN
    assert "head(coalesce(rel.citations, []))" in _COMMON_RETURN


def test_answer_generator_uses_ollama_text_wrapper() -> None:
    client = FakeOllamaClient()
    generator = AnswerGenerator(ollama_client=client)  # type: ignore[arg-type]

    answer = generator.generate("What is required?", "[1]\nEvidence: Test")

    assert answer == "Grounded answer"
    assert client.model == "qwen3:8b"
    assert len(client.calls) == 1
    assert "What is required?" in client.calls[0][1]


def test_answer_generator_model_override_wins() -> None:
    client = FakeOllamaClient()
    AnswerGenerator(ollama_client=client, model="llama3.1")  # type: ignore[arg-type]

    assert client.model == "llama3.1"


def test_answer_generator_short_circuits_empty_context() -> None:
    client = FakeOllamaClient()
    generator = AnswerGenerator(ollama_client=client)  # type: ignore[arg-type]

    answer = generator.generate("What is required?", "")

    assert "does not contain enough evidence" in answer
    assert client.calls == []


def test_ask_parser_accepts_arguments_without_llm() -> None:
    args = build_parser().parse_args(
        [
            "When can a covered entity disclose PHI without authorization?",
            "--limit",
            "5",
            "--show-context",
            "--debug-retrieval",
            "--model",
            "llama3.1",
        ]
    )

    assert args.question.startswith("When can")
    assert args.limit == 5
    assert args.show_context is True
    assert args.debug_retrieval is True
    assert args.model == "llama3.1"
