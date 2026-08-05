from __future__ import annotations

"""Three-way retrieval ablation comparison (semantic vs graph vs hybrid).

Reads the per-mode evaluation outputs produced by ``run_evaluation.py --mode all``
(``results/<mode>/evaluation_results.json``), groups rows by ``category`` and, for
each category x mode, computes the mean of the metrics every mode can produce plus
the hybrid-only answer metrics. Emits a Markdown table (for pasting into the thesis)
and a JSON sibling.

Only aggregation + rendering live here; no retrieval or scoring logic is touched.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Marker used by run_evaluation for answer metrics under retrieval-only modes.
NOT_APPLICABLE = "not_applicable"

MODES = ("semantic", "graph", "hybrid")

# Metrics all three modes can produce (retrieval-side).
SHARED_METRICS = (
    "retrieval_recall",
    "concept_coverage",
    "regulation_coverage",
    "evidence_keyword_coverage",
)
# Metrics that only exist on the hybrid (answer-generating) path.
HYBRID_ONLY_METRICS = ("overall_score", "faithfulness_score")
ALL_METRICS = (*SHARED_METRICS, *HYBRID_ONLY_METRICS)

# Human-readable category labels for the thesis table. Unknown categories fall
# back to a title-cased rendering, so new categories still appear.
CATEGORY_DISPLAY = {
    "gdpr_only": "GDPR",
    "hipaa_only": "HIPAA",
    "eu_ai_act_only": "EU AI Act",
    "cross_regulation": "Cross-regulation",
    "mental_health_cross_regulation": "MH Cross-regulation",
    "mental_health_diagnostics": "MH Diagnostics",
}
CATEGORY_ORDER = tuple(CATEGORY_DISPLAY.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a semantic/graph/hybrid retrieval ablation comparison table.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing per-mode subdirectories (results/<mode>/evaluation_results.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Where to write ablation_comparison.md / .json.",
    )
    args = parser.parse_args()

    results_by_mode = load_results(args.results_dir)
    missing = [mode for mode in MODES if not results_by_mode.get(mode)]
    if missing:
        print(
            f"Warning: no rows found for mode(s): {', '.join(missing)}. "
            f"Run `run_evaluation.py --mode all` first.",
            file=sys.stderr,
        )

    aggregation = aggregate(results_by_mode)
    md_path, json_path = write_comparison(aggregation, args.output_dir)
    print(f"Wrote comparison table to {md_path}")
    print(f"Wrote comparison JSON to {json_path}")


def load_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load results/<mode>/evaluation_results.json for each mode (missing -> [])."""
    results_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in MODES:
        path = results_dir / mode / "evaluation_results.json"
        results_by_mode[mode] = _load_rows(path)
    return results_by_mode


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        return []
    return [row for row in loaded if isinstance(row, dict)]


def aggregate(results_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Compute per-category-per-mode metric means plus an overall row.

    Hybrid-only metrics are reported as NOT_APPLICABLE for semantic/graph so the
    table never presents a missing answer metric as a 0 (failing) score.
    """
    categories = _ordered_categories(results_by_mode)

    category_block: dict[str, Any] = {}
    for category in categories:
        per_mode: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for mode in MODES:
            rows = [row for row in results_by_mode.get(mode, []) if row.get("category") == category]
            counts[mode] = len(rows)
            per_mode[mode] = _metric_means(rows, mode)
        category_block[category] = {
            "display": CATEGORY_DISPLAY.get(category, _fallback_display(category)),
            "counts": counts,
            "metrics": per_mode,
        }

    overall: dict[str, Any] = {}
    overall_counts: dict[str, int] = {}
    for mode in MODES:
        rows = list(results_by_mode.get(mode, []))
        overall_counts[mode] = len(rows)
        overall[mode] = _metric_means(rows, mode)

    return {
        "modes": list(MODES),
        "shared_metrics": list(SHARED_METRICS),
        "hybrid_only_metrics": list(HYBRID_ONLY_METRICS),
        "categories": category_block,
        "overall": {"counts": overall_counts, "metrics": overall},
        # Abstention accuracy is reported on its own, not folded into ALL_METRICS:
        # it is a pass/fail over a *subset* of questions (those annotated
        # expects_abstention), so averaging it alongside per-question metrics
        # would misrepresent both.
        "abstention": _abstention_summary(results_by_mode),
    }


def _abstention_summary(results_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-mode abstention accuracy over questions annotated expects_abstention.

    ``accuracy`` is the fraction of those questions the mode correctly abstained on
    (None when the mode has no abstention-annotated rows).
    """
    summary: dict[str, Any] = {}
    for mode in MODES:
        rows = [row for row in results_by_mode.get(mode, []) if row.get("expects_abstention")]
        correct = sum(1 for row in rows if row.get("abstention_correct") is True)
        summary[mode] = {
            "count": len(rows),
            "correct": correct,
            "accuracy": round(correct / len(rows), 3) if rows else None,
        }
    return summary


def _metric_means(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for metric in ALL_METRICS:
        if metric in HYBRID_ONLY_METRICS and mode != "hybrid":
            values[metric] = NOT_APPLICABLE
        else:
            values[metric] = _mean(row.get(metric) for row in rows)
    return values


def _mean(values: Any) -> float | None:
    nums = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 3)


def _ordered_categories(results_by_mode: dict[str, list[dict[str, Any]]]) -> list[str]:
    seen: list[str] = []
    for mode in MODES:
        for row in results_by_mode.get(mode, []):
            category = row.get("category") or "uncategorized"
            if category not in seen:
                seen.append(category)
    known = [category for category in CATEGORY_ORDER if category in seen]
    extra = [category for category in seen if category not in CATEGORY_ORDER]
    return [*known, *extra]


def _fallback_display(category: str) -> str:
    return category.replace("_", " ").title()


def render_markdown(aggregation: dict[str, Any]) -> str:
    modes = aggregation["modes"]
    lines: list[str] = []
    lines.append("# Retrieval Mode Ablation — Category × Mode")
    lines.append("")
    lines.append(
        "Mean metric per category and retrieval mode, from "
        "`results/{semantic,graph,hybrid}/evaluation_results.json`. "
        "`overall_score` and `faithfulness_score` require answer generation and are "
        "`n/a` for the retrieval-only semantic/graph modes."
    )
    lines.append("")

    categories = aggregation["categories"]
    for metric in ALL_METRICS:
        lines.append(f"## {metric}")
        lines.append("")
        lines.append("| Category | " + " | ".join(modes) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in modes) + " |")
        for category, block in categories.items():
            cells = [_fmt(block["metrics"][mode].get(metric)) for mode in modes]
            lines.append(f"| {block['display']} | " + " | ".join(cells) + " |")
        overall_metrics = aggregation["overall"]["metrics"]
        overall_cells = [_fmt(overall_metrics[mode].get(metric)) for mode in modes]
        lines.append("| **Overall** | " + " | ".join(f"**{cell}**" for cell in overall_cells) + " |")
        lines.append("")

    abstention = aggregation.get("abstention", {})
    lines.append("## Abstention accuracy")
    lines.append("")
    lines.append(
        "Fraction of questions annotated `expects_abstention: true` where the mode "
        "correctly declined a confident determination (insufficient_evidence / "
        "unconfirmed). Reported separately from the per-question metrics above."
    )
    lines.append("")
    lines.append("| Mode | Abstention accuracy | Correct / annotated |")
    lines.append("| --- | --- | --- |")
    for mode in modes:
        block = abstention.get(mode, {})
        accuracy = _fmt(block.get("accuracy"))
        lines.append(f"| {mode} | {accuracy} | {block.get('correct', 0)} / {block.get('count', 0)} |")
    lines.append("")

    counts = aggregation["overall"]["counts"]
    lines.append("## Row counts")
    lines.append("")
    lines.append("| Mode | Questions evaluated |")
    lines.append("| --- | --- |")
    for mode in modes:
        lines.append(f"| {mode} | {counts.get(mode, 0)} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if value == NOT_APPLICABLE:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_comparison(aggregation: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "ablation_comparison.md"
    json_path = output_dir / "ablation_comparison.json"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(aggregation))
        handle.write("\n")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregation, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return md_path, json_path


if __name__ == "__main__":
    main()
