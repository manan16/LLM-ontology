from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.run_evaluation import build_result_row, concept_matches, evaluate_question, keyword_matches


def test_keyword_matching_handles_hyphenation_and_spacing() -> None:
    text = "Providers of high-risk AI systems must keep technical documentation."

    assert keyword_matches("high risk AI system", text)
    assert keyword_matches("technical documentation", text)


def test_concept_matching_expands_camel_case_labels() -> None:
    text = "The evidence discusses protected health information and human oversight requirements."

    assert concept_matches("ProtectedHealthInformation", text)
    assert concept_matches("HumanOversightRequirement", text)


def test_build_result_row_computes_deterministic_metrics() -> None:
    question = {
        "id": "mh_001",
        "category": "mental_health_cross_regulation",
        "question": "What compliance obligations apply?",
        "expected_regulations": ["GDPR", "HIPAA", "EU AI Act"],
        "expected_concepts": ["ProtectedHealthInformation", "HumanOversightRequirement"],
        "expected_evidence_keywords": ["protected health information", "human oversight", "technical documentation"],
        "notes": "baseline",
    }
    result = {
        "answer": (
            "HIPAA protects protected health information. The EU AI Act requires human oversight "
            "and technical documentation for high-risk AI systems. Article 14 applies."
        ),
        "evidence": [
            {
                "statement": "HIPAA PHI safeguards",
                "source_group": "hipaa",
                "source_document": "hipaa.pdf",
                "evidence_text": "A covered entity protects protected health information.",
                "citation": "45 CFR 164.502",
            },
            {
                "statement": "EU AI Act oversight",
                "source_group": "eu_ai_act",
                "source_document": "eu_ai_act.pdf",
                "evidence_text": "High-risk AI systems require human oversight and technical documentation.",
                "citation": "Article 14",
            },
            {
                "statement": "GDPR health data",
                "source_group": "gdpr",
                "source_document": "gdpr.pdf",
                "evidence_text": "GDPR regulates health data.",
                "citation": "Article 9",
            },
        ],
        "debug": {"top_rows": []},
    }

    row = build_result_row(question, result, latency_seconds=1.25)

    assert row["retrieved_regulations"] == ["GDPR", "HIPAA", "EU AI Act"]
    assert row["regulation_coverage"] == 1.0
    assert row["concept_coverage"] == 1.0
    assert row["evidence_keyword_coverage"] == 1.0
    assert row["citation_coverage"] == 1.0
    assert row["status"] == "ok"


def test_evaluator_mh_006_uses_shared_debug_context_sources(monkeypatch) -> None:
    question = {
        "id": "mh_006",
        "category": "mental_health_diagnostics",
        "question": "What human oversight is required for AI-supported mental health diagnosis?",
        "expected_regulations": ["EU AI Act", "GDPR"],
        "expected_concepts": ["HumanOversightRequirement", "AutomatedDecisionMaking"],
        "expected_evidence_keywords": ["human oversight", "human intervention"],
    }

    def fake_answer_question(question_text, limit=30, show_context=False, debug=False, model=None):
        assert question_text == question["question"]
        assert debug is True
        return {
            "answer": "EU AI Act requires human oversight; GDPR supports human intervention.",
            "evidence": [
                {
                    "statement": "Human oversight",
                    "source_group": "eu_ai_act",
                    "source_document": "eu_ai_act.pdf",
                    "evidence_text": "High-risk AI systems require human oversight.",
                    "citation": "Article 14",
                },
                {
                    "statement": "Automated decision-making",
                    "source_group": "gdpr",
                    "source_document": "gdpr.pdf",
                    "evidence_text": "Data subjects may obtain human intervention.",
                    "citation": "Article 22",
                },
            ],
            "debug": {
                "query_plan": {
                    "is_mental_health_query": True,
                    "mental_health_intent_labels": ["human_oversight"],
                    "explicit_regulations": [],
                    "inferred_regulations": ["EU AI Act", "GDPR"],
                    "target_source_documents": ["eu_ai_act.pdf", "gdpr.pdf"],
                },
                "selection": {
                    "context_row_ids_rendered": [
                        "eu_ai_act.pdf|Human oversight|Article 14",
                        "gdpr.pdf|Automated decision-making|Article 22",
                    ],
                    "coverage_sources_present": ["eu_ai_act.pdf", "gdpr.pdf"],
                    "coverage_sources_missing": [],
                },
                "top_rows": [
                    {"source_group": "hipaa", "source_document": "hipaa.pdf", "statement": "Legacy wrong row"}
                ],
            },
        }

    monkeypatch.setattr("evaluation.run_evaluation.answer_question", fake_answer_question)

    row = evaluate_question(question)

    assert set(row["source_documents"]) == {"eu_ai_act.pdf", "gdpr.pdf"}
    assert row["source_documents"] != ["hipaa.pdf"]
    assert row["inferred_regulations"] == ["EU AI Act", "GDPR"]
    assert row["target_source_documents"] == ["eu_ai_act.pdf", "gdpr.pdf"]
    assert row["coverage_sources_present"] == ["eu_ai_act.pdf", "gdpr.pdf"]
    assert row["coverage_sources_missing"] == []
    assert row["mental_health_query"] is True
    assert row["mental_health_intent_labels"] == ["human_oversight"]
