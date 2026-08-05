from __future__ import annotations

import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.run_evaluation import (
    ANSWER_DEPENDENT_FIELDS,
    NOT_APPLICABLE,
    build_result_row,
    concept_matches,
    evaluate_question,
    find_retrieved_regulations,
    keyword_matches,
    score_citation_accuracy,
    score_faithfulness,
)
from evaluation.run_evaluation import NO_EVIDENCE_MESSAGE


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


def test_provenance_gated_metrics_full_audit(monkeypatch) -> None:
    """End-to-end metric check with a stubbed RAG call and known provenance.

    Exercises T1-T5: clean text split, provenance-gated regulation detection,
    structural aliases, the re-weighted overall_score, and the matched/missed audit.
    """
    question = {
        "id": "t6_001",
        "category": "cross_regulation",
        "question": "What oversight applies to automated decisions?",
        "expected_regulations": ["GDPR", "EU AI Act"],
        "expected_concepts": ["HumanOversightRequirement", "AutomatedDecisionMaking"],
        "expected_evidence_keywords": ["human oversight", "data minimization"],
        "notes": "t6",
    }

    # The GDPR-sourced statement deliberately contains HIPAA aliases ("covered
    # entity", "phi"); provenance must keep HIPAA out of retrieved_regulations.
    def fake_answer_question(question_text, limit=30, show_context=False, debug=False, model=None):
        assert debug is True
        return {
            "answer": "The EU AI Act Article 14 requires human oversight.",
            "evidence": [
                {
                    "statement": "GDPR safeguards",
                    "source_group": "gdpr",
                    "source_document": "gdpr.pdf",
                    "evidence_text": "A covered entity processes phi under data protection rules.",
                    "citation": "Article 9",
                },
                {
                    "statement": "oversight",
                    "source_group": "eu_ai_act",
                    "source_document": "eu_ai_act.pdf",
                    "evidence_text": "High-risk AI systems require human oversight.",
                    "citation": "Article 14",
                },
            ],
            "debug": {"top_rows": []},
        }

    monkeypatch.setattr("evaluation.run_evaluation.answer_question", fake_answer_question)
    row = evaluate_question(question)

    # Provenance-gated regulation detection (HIPAA aliases inside GDPR text ignored).
    assert row["retrieved_regulations"] == ["GDPR", "EU AI Act"]
    assert "HIPAA" not in row["retrieved_regulations"]

    # Retrieval coverages computed against evidence_text.
    assert row["regulation_coverage"] == 1.0
    assert row["concept_coverage"] == 0.5
    assert row["evidence_keyword_coverage"] == 0.5
    assert row["retrieval_recall"] == 0.667

    # Answer relevance scored against the answer only.
    assert row["answer_relevance_score"] == 0.5

    # Structural citation/grounding signals and their accurately-named aliases.
    assert row["citation_coverage"] == 1.0
    assert row["has_citations"] is True
    assert row["faithfulness_score"] == 1.0
    assert row["evidence_grounding"] == 1.0

    # Single weighted average over orthogonal axes, each counted once.
    # 0.40*0.667 + 0.30*0.5 + 0.15*1.0 + 0.15*1.0 = 0.717
    assert row["overall_score"] == 0.717

    # Matched vs missed audit trail.
    assert row["matched_regulations"] == ["GDPR", "EU AI Act"]
    assert row["missed_regulations"] == []
    assert row["matched_concepts"] == ["HumanOversightRequirement"]
    assert row["missed_concepts"] == ["AutomatedDecisionMaking"]
    assert row["matched_evidence_keywords"] == ["human oversight"]
    assert row["missed_evidence_keywords"] == ["data minimization"]
    assert row["match_audit"]["regulations"]["missed"] == []
    assert row["match_audit"]["concepts"]["missed"] == ["AutomatedDecisionMaking"]
    assert row["match_audit"]["evidence_keywords"]["matched"] == ["human oversight"]


def test_score_citation_accuracy_none_when_unannotated() -> None:
    # No ground truth yet -> None (distinct from a genuine 0.0), so faithfulness
    # falls back to the structural 0/0.5/1.0 heuristic.
    assert score_citation_accuracy(["Article 5(1)(e)"], []) is None

    evidence = [{"statement": "x", "evidence_text": "y"}]
    # Structural fallback: citations + evidence present -> 1.0.
    assert score_faithfulness("Article 5 applies.", ["Article 5"], evidence, []) == 1.0
    # Structural fallback: evidence only -> 0.5.
    assert score_faithfulness("Some answer.", [], evidence, []) == 0.5


def test_score_citation_accuracy_matches_across_surface_variants() -> None:
    expected = ["Article 5(1)(e)"]
    # Each of these is the same citation written differently and must count as a match.
    assert score_citation_accuracy(["Art. 5(1)(e)"], expected) == 1.0
    assert score_citation_accuracy(["GDPR Art 5.1.e"], expected) == 1.0
    assert score_citation_accuracy(["Article 5(1)(e)"], expected) == 1.0

    # When annotated, faithfulness returns the accuracy score, not the structural 1.0.
    evidence = [{"statement": "x", "evidence_text": "y"}]
    assert score_faithfulness("Art. 5(1)(e) applies.", ["Art. 5(1)(e)"], evidence, expected) == 1.0


def test_score_citation_accuracy_mismatch_scores_zero() -> None:
    expected = ["Article 5(1)(e)"]
    # Wrong article -> canonical keys differ ("8" vs "51e") -> 0.0, not a false 1.0.
    assert score_citation_accuracy(["Article 8"], expected) == 0.0

    # Partial: one of two expected citations found -> 0.5.
    assert score_citation_accuracy(["Art. 5(1)(e)"], ["Article 5(1)(e)", "Article 8"]) == 0.5

    # Faithfulness reflects the mismatch even though citations+evidence are present.
    evidence = [{"statement": "x", "evidence_text": "y"}]
    assert score_faithfulness("Article 8 applies.", ["Article 8"], evidence, expected) == 0.0


_ABSTENTION_QUESTION = {
    "id": "stress_001",
    "category": "stress",
    "question": "Does the Martian Data Act permit exporting brain-implant telemetry?",
    "expected_regulations": [],
    "expected_concepts": [],
    "expected_evidence_keywords": [],
    "expected_citations": [],
    "expects_abstention": True,
}


def test_expects_abstention_correct_scores_one() -> None:
    # System correctly abstains (insufficient_evidence verdict) -> pass.
    result = {
        "answer": NO_EVIDENCE_MESSAGE,
        "evidence": [],
        "determination": {"verdict": "insufficient_evidence", "obligations": [], "summary": ""},
        "debug": {"top_rows": []},
    }
    row = build_result_row(_ABSTENTION_QUESTION, result, latency_seconds=1.0)

    assert row["expects_abstention"] is True
    assert row["abstention_correct"] is True
    # Retrieval/answer/citation axes are meaningless here -> pass/fail only.
    assert row["overall_score"] == 1.0


def test_expects_abstention_incorrect_scores_zero() -> None:
    # System answers with a confident verdict when it should have abstained -> fail.
    result = {
        "answer": "The Martian Data Act imposes export obligations under Article 5.",
        "evidence": [
            {
                "statement": "export rule",
                "source_group": "gdpr",
                "source_document": "gdpr.pdf",
                "evidence_text": "Controllers must document transfers.",
                "citation": "Article 5",
            }
        ],
        "determination": {"verdict": "obligations_apply", "obligations": ["x"], "summary": "y"},
        "debug": {"top_rows": []},
    }
    row = build_result_row(_ABSTENTION_QUESTION, result, latency_seconds=1.0)

    assert row["expects_abstention"] is True
    assert row["abstention_correct"] is False
    assert row["overall_score"] == 0.0


def test_non_abstention_question_unaffected() -> None:
    # A normal question (expects_abstention absent) keeps the weighted-blend score
    # and is never pinned to 0.0/1.0 by the abstention branch.
    question = {
        "id": "gdpr_x",
        "category": "gdpr_only",
        "question": "What oversight applies?",
        "expected_regulations": ["GDPR", "EU AI Act"],
        "expected_concepts": ["HumanOversightRequirement", "AutomatedDecisionMaking"],
        "expected_evidence_keywords": ["human oversight", "data minimization"],
        "expected_citations": [],
    }
    result = {
        "answer": "The EU AI Act Article 14 requires human oversight.",
        "evidence": [
            {
                "statement": "oversight",
                "source_group": "eu_ai_act",
                "source_document": "eu_ai_act.pdf",
                "evidence_text": "High-risk AI systems require human oversight.",
                "citation": "Article 14",
            },
            {
                "statement": "gdpr",
                "source_group": "gdpr",
                "source_document": "gdpr.pdf",
                "evidence_text": "GDPR safeguards personal data.",
                "citation": "Article 9",
            },
        ],
        "determination": {"verdict": "obligations_apply", "obligations": ["x"], "summary": "y"},
        "debug": {"top_rows": []},
    }
    row = build_result_row(question, result, latency_seconds=1.0)

    assert row["expects_abstention"] is False
    assert row["abstention_correct"] is False
    # Partial coverage -> a genuine weighted score, not the abstention pass/fail.
    assert 0.0 < row["overall_score"] < 1.0


def test_alias_fallback_only_when_provenance_absent() -> None:
    """Aliases attribute a regulation only for items lacking provenance."""
    # HIPAA aliases inside a GDPR-sourced statement -> not HIPAA.
    sourced = [
        {
            "source_group": "gdpr",
            "source_document": "gdpr.pdf",
            "evidence_text": "A covered entity may process phi under GDPR.",
        }
    ]
    assert find_retrieved_regulations(sourced, []) == ["GDPR"]

    # Same aliases on an item with no provenance -> fallback flags HIPAA.
    unsourced = [{"evidence_text": "A covered entity must protect phi."}]
    assert find_retrieved_regulations(unsourced, []) == ["HIPAA"]


# ---------------------------------------------------------------------------
# Task 1: --mode dispatch + not_applicable handling for retrieval-only modes.
# ---------------------------------------------------------------------------
from rag.retrieval_service import RetrievalResult  # noqa: E402


_ABLATION_QUESTION = {
    "id": "abl_001",
    "category": "cross_regulation",
    "question": "What oversight applies to automated decisions?",
    "expected_regulations": ["GDPR", "EU AI Act"],
    "expected_concepts": ["HumanOversightRequirement", "AutomatedDecisionMaking"],
    "expected_evidence_keywords": ["human oversight", "data minimization"],
    "notes": "ablation",
}


def _ablation_rows() -> list[dict]:
    # GDPR row via source_group provenance; EU AI Act row is a semantic chunk that
    # only carries source_chunk_text + source_document (exercises text coalescing
    # and source_document-based provenance).
    return [
        {
            "statement_name": "GDPR safeguards",
            "source_group": "gdpr",
            "source_document": "gdpr.pdf",
            "evidence_text": "A covered entity processes phi under data protection rules.",
            "citation": "Article 9",
            "score": 40,
        },
        {
            "statement_name": "Oversight chunk",
            "source_document": "eu_ai_act.pdf",
            "source_chunk_text": "High-risk AI systems require human oversight.",
            "citation": "Article 14",
            "similarity": 0.9,
            "score": 90,
            "semantic": True,
        },
    ]


def test_hybrid_mode_dispatch_keeps_answer_metrics(monkeypatch) -> None:
    def fake_answer_question(question_text, limit=30, show_context=False, debug=False, model=None):
        return {
            "answer": "EU AI Act Article 14 requires human oversight.",
            "evidence": [
                {
                    "statement": "oversight",
                    "source_group": "eu_ai_act",
                    "source_document": "eu_ai_act.pdf",
                    "evidence_text": "High-risk AI systems require human oversight.",
                    "citation": "Article 14",
                },
                {
                    "statement": "gdpr",
                    "source_group": "gdpr",
                    "source_document": "gdpr.pdf",
                    "evidence_text": "GDPR safeguards personal data.",
                    "citation": "Article 9",
                },
            ],
            "debug": {"top_rows": []},
        }

    monkeypatch.setattr("evaluation.run_evaluation.answer_question", fake_answer_question)

    row = evaluate_question(_ABLATION_QUESTION, mode="hybrid")

    assert row["mode"] == "hybrid"
    # Answer-dependent fields stay numeric/real for hybrid (not NOT_APPLICABLE).
    for field in ANSWER_DEPENDENT_FIELDS:
        assert row[field] != NOT_APPLICABLE
    assert isinstance(row["overall_score"], float)


def test_semantic_mode_dispatch_uses_retrieval_service(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve(query, mode, top_k=10, debug=False):
        captured["mode"] = mode
        captured["top_k"] = top_k
        captured["debug"] = debug
        return RetrievalResult(
            mode=mode,
            query=query,
            rows=_ablation_rows(),
            top_k=top_k,
            debug={"mode": mode, "query_plan": {"category": "compliance_question"}},
        )

    monkeypatch.setattr("evaluation.run_evaluation.retrieve_by_mode", fake_retrieve)
    # answer_question must NOT be called in retrieval-only modes.
    monkeypatch.setattr(
        "evaluation.run_evaluation.answer_question",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("answer_question called in semantic mode")),
    )

    row = evaluate_question(_ABLATION_QUESTION, limit=15, mode="semantic")

    # Routed through the retrieval service with the right args.
    assert captured == {"mode": "semantic", "top_k": 15, "debug": True}
    assert row["mode"] == "semantic"

    # Retrieval-side metrics computed from RetrievalResult.rows.
    assert row["retrieved_regulations"] == ["GDPR", "EU AI Act"]
    assert row["regulation_coverage"] == 1.0
    assert row["concept_coverage"] == 0.5
    assert row["evidence_keyword_coverage"] == 0.5
    assert row["retrieval_recall"] == 0.667

    # Every answer/determination-dependent field is NOT_APPLICABLE (never 0).
    for field in ANSWER_DEPENDENT_FIELDS:
        assert row[field] == NOT_APPLICABLE, field


def test_graph_mode_dispatch_marks_answer_fields_not_applicable(monkeypatch) -> None:
    def fake_retrieve(query, mode, top_k=10, debug=False):
        return RetrievalResult(mode=mode, query=query, rows=_ablation_rows(), top_k=top_k, debug={})

    monkeypatch.setattr("evaluation.run_evaluation.retrieve_by_mode", fake_retrieve)

    row = evaluate_question(_ABLATION_QUESTION, mode="graph")

    assert row["mode"] == "graph"
    assert row["retrieval_recall"] == 0.667
    for field in ANSWER_DEPENDENT_FIELDS:
        assert row[field] == NOT_APPLICABLE, field


def test_build_result_row_default_mode_is_backward_compatible() -> None:
    # No mode passed -> hybrid semantics, mode column present, answer fields real.
    result = {
        "answer": "Article 14 human oversight.",
        "evidence": [
            {
                "statement": "oversight",
                "source_group": "eu_ai_act",
                "source_document": "eu_ai_act.pdf",
                "evidence_text": "High-risk AI systems require human oversight.",
                "citation": "Article 14",
            }
        ],
        "debug": {"top_rows": []},
    }
    row = build_result_row(_ABLATION_QUESTION, result, latency_seconds=1.0)
    assert row["mode"] == "hybrid"
    assert row["overall_score"] != NOT_APPLICABLE


# ---------------------------------------------------------------------------
# Task 2: compare_modes aggregation.
# ---------------------------------------------------------------------------
from evaluation import compare_modes  # noqa: E402


def _result_row(category: str, mode: str, retrieval_recall: float, **overrides) -> dict:
    row = {
        "category": category,
        "mode": mode,
        "retrieval_recall": retrieval_recall,
        "concept_coverage": overrides.get("concept_coverage", 0.5),
        "regulation_coverage": overrides.get("regulation_coverage", 1.0),
        "evidence_keyword_coverage": overrides.get("evidence_keyword_coverage", 0.5),
    }
    if mode == "hybrid":
        row["overall_score"] = overrides.get("overall_score", 0.7)
        row["faithfulness_score"] = overrides.get("faithfulness_score", 1.0)
    else:
        row["overall_score"] = NOT_APPLICABLE
        row["faithfulness_score"] = NOT_APPLICABLE
    return row


def _results_by_mode() -> dict[str, list[dict]]:
    return {
        "semantic": [
            _result_row("gdpr_only", "semantic", 0.6),
            _result_row("hipaa_only", "semantic", 0.4),
        ],
        "graph": [
            _result_row("gdpr_only", "graph", 0.8),
            _result_row("hipaa_only", "graph", 0.9),
        ],
        "hybrid": [
            _result_row("gdpr_only", "hybrid", 1.0, overall_score=0.8),
            _result_row("gdpr_only", "hybrid", 0.5, overall_score=0.6),
            _result_row("hipaa_only", "hybrid", 0.7, overall_score=0.7),
        ],
    }


def test_compare_modes_aggregate_means_and_not_applicable() -> None:
    aggregation = compare_modes.aggregate(_results_by_mode())

    gdpr = aggregation["categories"]["gdpr_only"]
    assert gdpr["display"] == "GDPR"
    assert gdpr["counts"] == {"semantic": 1, "graph": 1, "hybrid": 2}

    # Shared retrieval metric: mean across the category's rows per mode.
    assert gdpr["metrics"]["semantic"]["retrieval_recall"] == 0.6
    assert gdpr["metrics"]["graph"]["retrieval_recall"] == 0.8
    assert gdpr["metrics"]["hybrid"]["retrieval_recall"] == 0.75  # mean(1.0, 0.5)

    # Hybrid-only metrics: numeric for hybrid, NOT_APPLICABLE for the others.
    assert gdpr["metrics"]["hybrid"]["overall_score"] == 0.7  # mean(0.8, 0.6)
    assert gdpr["metrics"]["semantic"]["overall_score"] == NOT_APPLICABLE
    assert gdpr["metrics"]["graph"]["faithfulness_score"] == NOT_APPLICABLE

    # Overall (all categories) per mode.
    assert aggregation["overall"]["metrics"]["semantic"]["retrieval_recall"] == 0.5  # mean(0.6, 0.4)
    assert aggregation["overall"]["counts"]["hybrid"] == 3


def test_compare_modes_load_and_write_roundtrip(tmp_path) -> None:
    results_by_mode = _results_by_mode()
    for mode, rows in results_by_mode.items():
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        (mode_dir / "evaluation_results.json").write_text(json.dumps(rows), encoding="utf-8")

    loaded = compare_modes.load_results(tmp_path)
    assert [row["retrieval_recall"] for row in loaded["semantic"]] == [0.6, 0.4]

    aggregation = compare_modes.aggregate(loaded)
    md_path, json_path = compare_modes.write_comparison(aggregation, tmp_path)

    assert md_path.exists() and json_path.exists()
    markdown = md_path.read_text(encoding="utf-8")
    assert "## retrieval_recall" in markdown
    assert "n/a" in markdown  # hybrid-only metric rendered n/a for semantic/graph
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert reloaded["categories"]["gdpr_only"]["metrics"]["hybrid"]["overall_score"] == 0.7
