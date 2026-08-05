"""Smoke test: the H1 scoring harness runs end-to-end on a 2-chunk fixture.

This exists so evaluation/extraction_eval.py does not silently rot. It does not
assert specific metric values (those depend on the real gold set); it asserts the
scorer runs without error and returns the expected structure on a tiny fixture
covering a true positive, a wrong-statement-type mismatch, a missing gold node,
and a matched relationship.
"""

from __future__ import annotations

from evaluation.extraction_eval import CATEGORIES, score_h1


def _gold_doc() -> dict:
    return {
        "human_validated": True,
        "claude_reviewed": True,
        "records": [
            {
                "regulation": "GDPR",
                "chunk_id": "chunk-a",
                "annotations": [
                    {
                        "statement_type": "obligation",
                        "evidence_text": "The controller shall implement appropriate measures.",
                        "nodes": [
                            {"node_type": "Actor", "canonical_name": "Controller",
                             "raw_text": "controller", "confidence": 0.9},
                        ],
                        "relationships": [
                            {"relationship_type": "IMPOSES_ON",
                             "source_canonical_name": "Security Obligation",
                             "source_node_type": "Obligation",
                             "target_canonical_name": "Controller",
                             "target_node_type": "Actor"},
                        ],
                    },
                    {
                        "statement_type": "prohibition",
                        "evidence_text": "Processing of special categories shall be prohibited.",
                        "nodes": [],
                        "relationships": [],
                    },
                ],
            },
            {
                "regulation": "HIPAA",
                "chunk_id": "chunk-b",
                "annotations": [
                    {
                        "statement_type": "requirement",
                        "evidence_text": "A covered entity must provide a written denial.",
                        "nodes": [],
                        "relationships": [],
                    },
                ],
            },
        ],
    }


def _raw_doc() -> dict:
    return {
        "model": "test-model",
        "records": [
            {
                "regulation": "GDPR",
                "chunk": {"chunk_id": "chunk-a",
                          "text": "The controller shall implement appropriate measures. "
                                  "Processing of special categories shall be prohibited."},
                "extraction_error": None,
                "extraction": {
                    "chunk_summary": "s",
                    "statements": [
                        {  # true-positive statement match (same type + evidence)
                            "statement_type": "obligation",
                            "evidence_text": "The controller shall implement appropriate measures.",
                            "nodes": [],  # system emits no node -> gold node is a miss
                            "relationships": [
                                {"relationship_type": "IMPOSES_ON",
                                 "source_canonical_name": "Security Obligation",
                                 "source_node_type": "Obligation",
                                 "target_canonical_name": "controller",  # case-insensitive match
                                 "target_node_type": "Actor"},
                            ],
                        },
                        {  # same evidence as gold prohibition but wrong type
                            "statement_type": "requirement",
                            "evidence_text": "Processing of special categories shall be prohibited.",
                            "nodes": [],
                            "relationships": [],
                        },
                    ],
                },
            },
            {
                "regulation": "HIPAA",
                "chunk": {"chunk_id": "chunk-b",
                          "text": "A covered entity must provide a written denial."},
                "extraction_error": "ValueError: simulated failure",
                "extraction": None,  # failed chunk -> everything missing
            },
        ],
    }


def test_score_h1_runs_end_to_end() -> None:
    metrics = score_h1(_gold_doc(), _raw_doc())

    # structure
    for level in ("statement", "node", "relationship"):
        assert set(metrics[level]) >= {"true_positive", "system_count", "gold_count",
                                       "precision", "recall", "f1"}
    assert set(metrics["error_categories"]) == set(CATEGORIES)

    # a couple of anchored expectations that must hold on this fixture
    assert metrics["statement"]["true_positive"] == 1        # the obligation matched
    assert metrics["error_categories"]["wrong_statement_type"] == 1
    assert metrics["node"]["system_count"] == 0              # no system nodes
    assert metrics["relationship"]["true_positive"] == 1     # case-insensitive endpoint match
    assert "chunk-b" in metrics["meta"]["failed_extraction_chunks"]
