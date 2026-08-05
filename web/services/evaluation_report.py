from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)

# Locations written by evaluation/run_evaluation.py. `--mode all` writes one file
# per retrieval mode under results/<mode>/; a single-mode run writes the flat
# results/evaluation_results.json. Hybrid is the production path, so the report
# prefers it and only falls back to the flat file when no per-mode run exists.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "evaluation" / "results"
DEFAULT_RESULTS_PATH = _RESULTS_DIR / "hybrid" / "evaluation_results.json"
LEGACY_RESULTS_PATH = _RESULTS_DIR / "evaluation_results.json"

# The seven component metrics plus the headline overall_score. Order controls
# the column order in the per-question table.
COMPONENT_METRICS: list[tuple[str, str]] = [
    ("retrieval_recall", "Recall"),
    ("concept_coverage", "Concept"),
    ("evidence_keyword_coverage", "Evidence kw"),
    ("regulation_coverage", "Regulation"),
    ("citation_coverage", "Citation"),
    ("answer_relevance_score", "Relevance"),
    ("faithfulness_score", "Faithfulness"),
]

# Labels for the metrics compare_modes aggregates per mode. Mirrors
# compare_modes.ALL_METRICS; unknown keys fall back to a de-underscored name.
ABLATION_METRIC_LABELS: dict[str, str] = {
    "retrieval_recall": "Retrieval recall",
    "concept_coverage": "Concept coverage",
    "regulation_coverage": "Regulation coverage",
    "evidence_keyword_coverage": "Evidence keyword coverage",
    "overall_score": "Overall score",
    "faithfulness_score": "Faithfulness",
}

# compare_modes' marker for an answer metric under a retrieval-only mode.
_NOT_APPLICABLE = "not_applicable"

# Stable display order / labels for the known categories. Unknown categories
# are appended in their first-seen order so nothing is silently dropped.
CATEGORY_LABELS: dict[str, str] = {
    "gdpr_only": "GDPR only",
    "hipaa_only": "HIPAA only",
    "eu_ai_act_only": "EU AI Act only",
    "cross_regulation": "Cross-regulation",
    "mental_health_diagnostics": "Mental-health diagnostics",
    "mental_health_cross_regulation": "Mental-health cross-regulation",
}


def load_evaluation_report(path: Path | None = None) -> dict[str, Any]:
    """Read the pre-computed evaluation results file and shape it for the
    template. Never raises for a missing/empty/corrupt file — returns
    ``{"available": False, ...}`` so the page can show an empty state.
    """
    results_path = path or (DEFAULT_RESULTS_PATH if DEFAULT_RESULTS_PATH.exists() else LEGACY_RESULTS_PATH)

    if not results_path.exists():
        LOGGER.info("Evaluation results file not found at %s", results_path)
        return _unavailable(results_path, "No evaluation results file was found.")

    try:
        raw = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read evaluation results at %s: %s", results_path, exc)
        return _unavailable(results_path, "The evaluation results file could not be read.")

    rows = raw if isinstance(raw, list) else raw.get("results", [])
    if not isinstance(rows, list) or not rows:
        LOGGER.info("Evaluation results file at %s is empty", results_path)
        return _unavailable(results_path, "The evaluation results file is empty.")

    prepared = [_prepare_row(row) for row in rows]
    generated_at = datetime.fromtimestamp(results_path.stat().st_mtime, tz=timezone.utc)

    return {
        "available": True,
        "path": str(results_path),
        "mode": str(rows[0].get("mode") or "hybrid") if isinstance(rows[0], dict) else "hybrid",
        "generated_at": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
        "summary": _summary(prepared),
        "categories": _category_breakdown(prepared),
        "metrics": COMPONENT_METRICS,
        "rows": prepared,
    }


def load_ablation_comparison(results_dir: Path | None = None) -> dict[str, Any] | None:
    """Full semantic/graph/hybrid comparison for the evaluation page.

    Reuses evaluation/compare_modes.py's loading + aggregation so every mean is
    computed exactly as the thesis table computes it (no duplicated logic). Reads
    ``evaluation/results/<mode>/evaluation_results.json`` for each mode and
    reshapes the aggregation into: per-metric overall + per-category values with
    the winning mode flagged, abstention accuracy, and row counts. Returns
    ``None`` (never raises) when none of the three files are present/populated,
    so the template can render nothing.
    """
    # Reuse the aggregation module. Ensure the project root is importable first
    # (compare_modes lives under evaluation/, a namespace package).
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    try:
        from evaluation import compare_modes
    except ImportError as exc:  # pragma: no cover - only if evaluation/ is absent
        LOGGER.warning("Could not import evaluation.compare_modes: %s", exc)
        return None

    base = results_dir or compare_modes.DEFAULT_RESULTS_DIR
    try:
        results_by_mode = compare_modes.load_results(base)
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.warning("Could not read ablation results under %s: %s", base, exc)
        return None

    # Degrade to None when no mode produced any rows -- nothing to compare.
    if not any(results_by_mode.get(mode) for mode in compare_modes.MODES):
        LOGGER.info("No ablation results found under %s", base)
        return None

    aggregation = compare_modes.aggregate(results_by_mode)
    mode_ids = list(compare_modes.MODES)
    overall_metrics = aggregation["overall"]["metrics"]
    overall_counts = aggregation["overall"]["counts"]

    modes = [
        {
            "mode": mode,
            "label": mode.capitalize(),
            "count": overall_counts.get(mode, 0),
            "production": mode == "hybrid",
            # headline numbers for the mode cards
            "retrieval_recall": _plain(overall_metrics.get(mode, {}).get("retrieval_recall")),
            "overall_score": _plain(overall_metrics.get(mode, {}).get("overall_score")),
        }
        for mode in mode_ids
    ]

    metrics = [
        {
            "key": key,
            "label": ABLATION_METRIC_LABELS.get(key, key.replace("_", " ")),
            "hybrid_only": key in compare_modes.HYBRID_ONLY_METRICS,
            "overall": _mode_cells(overall_metrics, key, mode_ids),
            "categories": [
                {
                    "label": block["display"],
                    "counts": block["counts"],
                    "cells": _mode_cells(block["metrics"], key, mode_ids),
                }
                for block in aggregation["categories"].values()
            ],
        }
        for key in compare_modes.ALL_METRICS
    ]

    abstention_raw = aggregation.get("abstention", {})
    abstention = [
        {
            "mode": mode,
            "label": mode.capitalize(),
            "accuracy": abstention_raw.get(mode, {}).get("accuracy"),
            "correct": abstention_raw.get(mode, {}).get("correct", 0),
            "count": abstention_raw.get(mode, {}).get("count", 0),
        }
        for mode in mode_ids
    ]
    has_abstention = any(item["count"] for item in abstention)

    return {
        "modes": modes,
        "metrics": metrics,
        "abstention": abstention,
        "has_abstention": has_abstention,
        "path": str(base),
    }


def _plain(value: Any) -> float | None:
    """Numeric value or None — collapses compare_modes' ``not_applicable``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _mode_cells(metrics_by_mode: dict[str, Any], key: str, mode_ids: list[str]) -> list[dict[str, Any]]:
    """One display cell per mode for a metric, with the winning mode flagged.

    ``not_applicable`` (an answer metric under a retrieval-only mode) is carried
    through as its own state so the template never renders it as a 0 score, and
    it never competes for "best".
    """
    raw = {mode: metrics_by_mode.get(mode, {}).get(key) for mode in mode_ids}
    numeric = {m: v for m, v in raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    best = max(numeric, key=numeric.get) if len(numeric) > 1 and len(set(numeric.values())) > 1 else None
    return [
        {
            "mode": mode,
            "value": raw[mode] if mode in numeric else None,
            "not_applicable": raw[mode] == _NOT_APPLICABLE,
            "best": mode == best,
        }
        for mode in mode_ids
    ]


def _unavailable(path: Path, message: str) -> dict[str, Any]:
    return {"available": False, "path": str(path), "message": message}


def _prepare_row(row: dict[str, Any]) -> dict[str, Any]:
    overall = _num(row.get("overall_score"))
    return {
        "question_id": str(row.get("question_id") or ""),
        "category": str(row.get("category") or ""),
        "category_label": CATEGORY_LABELS.get(str(row.get("category") or ""), str(row.get("category") or "—")),
        "question": str(row.get("question") or ""),
        "overall_score": overall,
        "metrics": {key: _num(row.get(key)) for key, _ in COMPONENT_METRICS},
        "status": str(row.get("status") or "ok"),
        "error": str(row.get("error") or ""),
        "latency_seconds": _num(row.get("latency_seconds")),
        "answer": str(row.get("answer") or ""),
        "citations": _as_list(row.get("citations")),
        "notes": str(row.get("notes") or ""),
        "expected_concepts": _as_list(row.get("expected_concepts")),
        "matched_concepts": _as_list(row.get("matched_concepts")),
        "expected_evidence_keywords": _as_list(row.get("expected_evidence_keywords")),
        "matched_evidence_keywords": _as_list(row.get("matched_evidence_keywords")),
        "expected_regulations": _as_list(row.get("expected_regulations")),
        "retrieved_regulations": _as_list(row.get("retrieved_regulations")),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overalls = [r["overall_score"] for r in rows if r["overall_score"] is not None]
    latencies = [r["latency_seconds"] for r in rows if r["latency_seconds"] is not None]
    errors = sum(1 for r in rows if r["status"] != "ok")
    return {
        "total": len(rows),
        "mean_overall": _mean(overalls),
        "mean_latency": _mean(latencies),
        "error_count": errors,
    }


def _category_breakdown(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = list(CATEGORY_LABELS.keys())
    seen: list[str] = []
    for r in rows:
        if r["category"] not in seen:
            seen.append(r["category"])
    ordered = [c for c in order if c in seen] + [c for c in seen if c not in order]

    breakdown: list[dict[str, Any]] = []
    for category in ordered:
        group = [r for r in rows if r["category"] == category]
        overalls = [r["overall_score"] for r in group if r["overall_score"] is not None]
        breakdown.append(
            {
                "category": category,
                "label": CATEGORY_LABELS.get(category, category or "—"),
                "count": len(group),
                "mean_overall": _mean(overalls),
            }
        )
    return breakdown


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
