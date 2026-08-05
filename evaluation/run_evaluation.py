from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.retrieval_service import retrieve as retrieve_by_mode
from web.services.rag_service import (
    NO_EVIDENCE_MESSAGE,
    UNCONFIRMED_MESSAGE,
    answer_question,
)


DEFAULT_INPUT = Path(__file__).resolve().parent / "golden_questions.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

# Retrieval strategies exposed via --mode. "hybrid" is the production path
# (answer_question, full metrics); "semantic"/"graph" are retrieval-only ablation
# modes routed through rag.retrieval_service.retrieve.
RETRIEVAL_MODES = ("semantic", "graph", "hybrid")
MODE_CHOICES = (*RETRIEVAL_MODES, "all")
DEFAULT_MODE = "hybrid"

# Verdicts that count as a (correct) abstention: the system declined to stamp a
# confident compliance determination. "insufficient_evidence" = no supporting
# evidence at all; "unconfirmed" = only semantic passages, not graph-confirmed.
ABSTENTION_VERDICTS = frozenset({"insufficient_evidence", "unconfirmed"})

# Sentinel for answer/determination-dependent fields under retrieval-only modes.
# Deliberately NOT 0 -- a 0 would read as a failing score in the comparison table,
# misrepresenting a metric that simply does not apply when no answer is generated.
NOT_APPLICABLE = "not_applicable"

# Fields that only have meaning when an answer (and determination) is generated,
# i.e. the hybrid/production path. Under semantic/graph modes they are set to
# NOT_APPLICABLE and excluded from overall_score.
ANSWER_DEPENDENT_FIELDS = (
    "answer",
    "citations",
    "citation_coverage",
    "has_citations",
    "answer_relevance_score",
    "faithfulness_score",
    "citation_accuracy",
    "evidence_grounding",
    "overall_score",
)

RESULT_COLUMNS = [
    "question_id",
    "category",
    "question",
    "expected_regulations",
    "retrieved_regulations",
    "matched_regulations",
    "missed_regulations",
    "expected_concepts",
    "matched_concepts",
    "missed_concepts",
    "expected_evidence_keywords",
    "matched_evidence_keywords",
    "missed_evidence_keywords",
    "answer",
    "citations",
    "source_documents",
    "explicit_regulations",
    "inferred_regulations",
    "target_source_documents",
    "coverage_sources_present",
    "coverage_sources_missing",
    "mental_health_query",
    "mental_health_intent_labels",
    "retrieval_recall",
    "concept_coverage",
    "evidence_keyword_coverage",
    "regulation_coverage",
    "citation_coverage",
    "has_citations",
    "answer_relevance_score",
    "faithfulness_score",
    "citation_accuracy",
    "evidence_grounding",
    "overall_score",
    "expects_abstention",
    "abstention_correct",
    "latency_seconds",
    "status",
    "error",
    "notes",
    "mode",
]

REGULATION_ALIASES = {
    "GDPR": ["gdpr", "general data protection regulation", "regulation (eu) 2016/679", "gdpr.pdf"],
    "HIPAA": ["hipaa", "protected health information", "phi", "covered entity", "45 cfr", "hipaa.pdf"],
    "EU AI Act": ["eu ai act", "ai act", "high-risk ai system", "high risk ai system", "2024/1689", "eu_ai_act.pdf"],
}

EMPTY_RESULT = {
    "answer": "",
    "evidence": [],
    "context": "",
    "debug": {},
    "metrics": {},
}

# Shared citation regexes: extract_citations() harvests Article/Art./CFR/[...] style
# citations from answers, and score_citation_accuracy() reuses the same patterns to
# canonicalize citations so surface variants ("Article 5(1)(e)" vs "Art. 5(1)(e)" vs
# "GDPR Art 5.1.e") normalize to one comparable key.
CITATION_PATTERNS = [
    r"\bArticle\s+\d+[A-Za-z0-9()\-]*",
    r"\bArt\.\s*\d+[A-Za-z0-9()\-]*",
    r"\b\d+\s*CFR\s*[\d.]+",
    r"\b45\s*CFR\s*[\d.]+",
    r"\[[^\]]+\]",
]

# overall_score weights. Each axis is orthogonal and counted exactly once, so the
# lexical-overlap coverages (already folded into retrieval_recall) are not re-added
# separately. Weights sum to 1.0: retrieval recall carries the most signal, answer
# relevance next, and the two structural axes least.
OVERALL_SCORE_WEIGHTS = {
    "retrieval_recall": 0.40,    # did retrieval surface expected regs/concepts/keywords
    "answer_relevance": 0.30,    # does the answer cover expected concepts/keywords
    "citation_coverage": 0.15,   # structural: are citations present
    "evidence_grounding": 0.15,  # structural: is the answer backed by retrieved evidence
}


def main() -> None:
    args = build_parser().parse_args()
    questions = load_questions(args.input)
    if args.category:
        selected_categories = set(args.category)
        questions = [question for question in questions if question.get("category") in selected_categories]
    if args.question_id:
        selected_ids = set(args.question_id)
        questions = [question for question in questions if question.get("id") in selected_ids]
    if args.max_questions:
        questions = questions[: args.max_questions]

    modes = RETRIEVAL_MODES if args.mode == "all" else (args.mode,)
    summaries: dict[str, float | None] = {}
    for mode in modes:
        # For a single-mode run (including the default hybrid) results go straight
        # into --output-dir. Only `--mode all` fans out into results/<mode>/
        # subdirectories.
        if args.mode == "all":
            print(f"\n=== Mode: {mode} ===")
            output_dir = args.output_dir / mode
        else:
            output_dir = args.output_dir
        results = run_mode(questions, mode, output_dir, args)
        summaries[mode] = mean_metric(results, "retrieval_recall")

    # retrieval_recall is the one axis every mode produces, so it is the comparable
    # cross-mode summary (overall_score blends answer axes only present for hybrid).
    print("\n=== Per-mode summary (mean retrieval_recall) ===")
    for mode in modes:
        print(f"{mode}: mean_retrieval_recall={summaries[mode]}")


def run_mode(
    questions: list[dict[str, Any]],
    mode: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.append:
        results = load_existing_results(output_dir)
    else:
        clear_outputs(output_dir)
        results = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        question_id = clean_text(question.get("id")) or f"question_{index}"
        print(f"[{index}/{total}] Evaluating {question_id}: {clean_text(question.get('question'))}")
        row = evaluate_question(
            question,
            limit=args.limit,
            model=args.model,
            include_context=args.include_context,
            mode=mode,
        )
        results.append(row)
        write_results(results, output_dir)
        print(
            f"[{index}/{total}] {question_id} status={row['status']} "
            f"overall_score={row['overall_score']} latency_seconds={row['latency_seconds']}"
        )
    write_results(results, output_dir)
    print(f"Wrote {len(results)} evaluation rows to {output_dir}")
    return results


def mean_metric(results: list[dict[str, Any]], key: str) -> float | None:
    values = [row.get(key) for row in results if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)]
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline golden-question evaluation for the KG-RAG QA system.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to golden questions JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/Markdown/JSON outputs.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum retrieved graph rows per question.")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=DEFAULT_MODE,
        help=(
            "Retrieval strategy: 'hybrid' (default, production answer_question path with full "
            "metrics), 'semantic' or 'graph' (retrieval-only ablation), or 'all' to run every "
            "mode over the same questions, writing each to results/<mode>/."
        ),
    )
    parser.add_argument("--model", help="Optional Ollama model override for answer generation.")
    parser.add_argument("--max-questions", type=int, help="Evaluate only the first N questions.")
    parser.add_argument("--question-id", action="append", help="Evaluate a specific question id. Can be repeated.")
    parser.add_argument("--category", action="append", help="Evaluate only questions in this category. Can be repeated.")
    parser.add_argument("--append", action="store_true", help="Append to existing evaluation output instead of overwriting it.")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Store retrieved context in JSON output. CSV/Markdown always store compact evidence fields.",
    )
    return parser


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        questions = json.load(handle)
    if not isinstance(questions, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return questions


def evaluate_question(
    question_item: dict[str, Any],
    limit: int = 30,
    model: str | None = None,
    include_context: bool = False,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    started_at = perf_counter()
    error = ""
    status = "ok"
    result: dict[str, Any] = EMPTY_RESULT

    try:
        if mode == "hybrid":
            result = answer_question(
                str(question_item.get("question") or ""),
                limit=limit,
                show_context=include_context,
                debug=True,
                model=model,
            )
        else:
            # Retrieval-only ablation (semantic / graph): no answer generation,
            # no determination. Route through the unified retrieval service and
            # adapt its rows into the result shape build_result_row consumes.
            result = _retrieval_only_result(question_item, mode, limit, include_context)
    except Exception as exc:  # pragma: no cover - exercised in real integration runs
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    latency_seconds = round(perf_counter() - started_at, 3)
    return build_result_row(question_item, result, latency_seconds, status, error, mode=mode)


def _retrieval_only_result(
    question_item: dict[str, Any],
    mode: str,
    limit: int,
    include_context: bool,
) -> dict[str, Any]:
    """Run a retrieval-only mode and shape it like an answer_question() result.

    Only the retrieval-side fields are populated (``evidence`` derived from
    ``RetrievalResult.rows`` and ``debug``); ``answer`` is empty because these
    modes generate no answer. build_result_row (called with the same ``mode``)
    marks the answer/determination-dependent fields as NOT_APPLICABLE.
    """
    retrieval = retrieve_by_mode(
        str(question_item.get("question") or ""),
        mode=mode,
        top_k=limit,
        debug=True,
    )
    debug = dict(retrieval.debug) if isinstance(retrieval.debug, dict) else {}
    debug.setdefault("top_rows", [])
    result: dict[str, Any] = {
        "answer": "",
        "evidence": _rows_to_evidence(retrieval.rows),
        "context": "",
        "debug": debug,
        "metrics": {},
    }
    if include_context:
        result["context"] = "\n\n".join(
            clean_text(item.get("evidence_text")) for item in result["evidence"] if clean_text(item.get("evidence_text"))
        )
    return result


def _rows_to_evidence(rows: Any) -> list[dict[str, Any]]:
    """Adapt RetrievalResult.rows into the evidence-item shape metrics expect.

    Coalesces ``source_chunk_text`` (semantic rows) into ``evidence_text`` and
    keeps the provenance (source_group/source_document) the retrieval metrics use.
    """
    evidence: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return evidence
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = clean_text(row.get("evidence_text")) or clean_text(row.get("source_chunk_text"))
        evidence.append(
            {
                "statement": clean_text(row.get("statement_name")) or clean_text(row.get("related_name")) or "Unknown",
                "source_group": row.get("source_group"),
                "source_document": row.get("source_document"),
                "evidence_text": text,
                "citation": clean_text(row.get("citation")),
                "score": row.get("score"),
                "semantic": bool(row.get("semantic")),
            }
        )
    return evidence


def build_result_row(
    question_item: dict[str, Any],
    result: dict[str, Any],
    latency_seconds: float,
    status: str = "ok",
    error: str = "",
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    answer = clean_text(result.get("answer"))
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    top_rows = debug.get("top_rows") if isinstance(debug.get("top_rows"), list) else []

    # Score against clean, separated text fields rather than one combined blob.
    # answer_text  -> the model's answer only.
    # evidence_text -> retrieved statement text only (no IDs, scores, field names
    #                  or json.dumps metadata, which would pollute the matches).
    answer_text = answer
    evidence_text = joined_text(
        [
            *(item.get("evidence_text") for item in evidence if isinstance(item, dict)),
            *(item.get("statement") for item in evidence if isinstance(item, dict)),
            *(item.get("statement") for item in top_rows if isinstance(item, dict)),
        ]
    )

    expected_regulations = list_values(question_item.get("expected_regulations"))
    expected_concepts = list_values(question_item.get("expected_concepts"))
    expected_keywords = list_values(question_item.get("expected_evidence_keywords"))
    expected_citations = list_values(question_item.get("expected_citations"))

    # Retrieval metrics measure what retrieval surfaced -> match against evidence_text.
    retrieved_regulations = find_retrieved_regulations(evidence, top_rows)
    matched_regulations = [reg for reg in expected_regulations if reg in retrieved_regulations]
    matched_concepts = [concept for concept in expected_concepts if concept_matches(concept, evidence_text)]
    matched_keywords = [keyword for keyword in expected_keywords if keyword_matches(keyword, evidence_text)]

    # Audit trail -- record which expected items were matched vs missed.
    missed_regulations = [reg for reg in expected_regulations if reg not in matched_regulations]
    missed_concepts = [concept for concept in expected_concepts if concept not in matched_concepts]
    missed_keywords = [keyword for keyword in expected_keywords if keyword not in matched_keywords]

    # Answer relevance measures the answer itself -> match against answer_text only.
    answer_matched_concepts = [concept for concept in expected_concepts if concept_matches(concept, answer_text)]
    answer_matched_keywords = [keyword for keyword in expected_keywords if keyword_matches(keyword, answer_text)]

    citations = extract_citations(answer, evidence, top_rows)
    source_documents = extract_source_documents(evidence, top_rows, debug)
    plan_debug = debug.get("query_plan") if isinstance(debug.get("query_plan"), dict) else {}
    selection_debug = debug.get("selection") if isinstance(debug.get("selection"), dict) else {}
    regulation_coverage = coverage(len(matched_regulations), len(expected_regulations))
    concept_coverage = coverage(len(matched_concepts), len(expected_concepts))
    keyword_coverage = coverage(len(matched_keywords), len(expected_keywords))
    citation_coverage = score_citation_coverage(answer, citations, source_documents)
    retrieval_recall = round((regulation_coverage + concept_coverage + keyword_coverage) / 3, 3)
    answer_relevance = score_answer_relevance(answer_text, answer_matched_keywords, expected_keywords, answer_matched_concepts, expected_concepts)
    citation_accuracy = score_citation_accuracy(citations, expected_citations)
    faithfulness = score_faithfulness(answer, citations, evidence, expected_citations)
    # Accurately-named structural signals. citation_coverage and faithfulness_score
    # are structural (presence checks), not semantic, so surface what they actually
    # measure under honest names. Values mirror the structural metrics above; no
    # semantic faithfulness is fabricated.
    has_citations = detect_has_citations(answer, citations)
    evidence_grounding = faithfulness
    # Single weighted average over orthogonal axes, each counted once. Lexical
    # overlap (regulation/concept/keyword coverage) lives only inside retrieval_recall
    # and is not re-added on its own.
    axis_values = {
        "retrieval_recall": retrieval_recall,
        "answer_relevance": answer_relevance,
        "citation_coverage": citation_coverage,
        "evidence_grounding": evidence_grounding,
    }
    overall_score = round(
        sum(OVERALL_SCORE_WEIGHTS[axis] * axis_values[axis] for axis in OVERALL_SCORE_WEIGHTS),
        3,
    )

    # Abstention-aware scoring. For a question deliberately designed to have no
    # answer, retrieval_recall/answer_relevance/citation_coverage are meaningless,
    # so overall_score collapses to a pass/fail on whether the system correctly
    # abstained (declined a confident determination) instead of the weighted blend.
    expects_abstention = bool(question_item.get("expects_abstention"))
    abstention_correct = detect_abstention(result, answer) if expects_abstention else False
    if expects_abstention:
        overall_score = 1.0 if abstention_correct else 0.0

    # Answer/determination-dependent axes only apply to the hybrid (production)
    # path. Under semantic/graph modes they are marked NOT_APPLICABLE so the
    # comparison table never reads them as failing (0) scores; overall_score is
    # likewise NOT_APPLICABLE because it blends those answer axes.
    answer_dependent = mode == "hybrid"

    def answer_field(value: Any) -> Any:
        return value if answer_dependent else NOT_APPLICABLE

    row = {
        "question_id": clean_text(question_item.get("id")),
        "category": clean_text(question_item.get("category")),
        "mode": mode,
        "question": clean_text(question_item.get("question")),
        "expected_regulations": expected_regulations,
        "retrieved_regulations": retrieved_regulations,
        "matched_regulations": matched_regulations,
        "missed_regulations": missed_regulations,
        "expected_concepts": expected_concepts,
        "matched_concepts": matched_concepts,
        "missed_concepts": missed_concepts,
        "expected_evidence_keywords": expected_keywords,
        "matched_evidence_keywords": matched_keywords,
        "missed_evidence_keywords": missed_keywords,
        "answer": answer_field(answer),
        "citations": answer_field(citations),
        "source_documents": source_documents,
        "explicit_regulations": list_values(plan_debug.get("explicit_regulations")),
        "inferred_regulations": list_values(plan_debug.get("inferred_regulations")),
        "target_source_documents": list_values(plan_debug.get("target_source_documents")),
        "coverage_sources_present": list_values(selection_debug.get("coverage_sources_present")),
        "coverage_sources_missing": list_values(selection_debug.get("coverage_sources_missing")),
        "mental_health_query": bool(plan_debug.get("is_mental_health_query")),
        "mental_health_intent_labels": list_values(plan_debug.get("mental_health_intent_labels")),
        "retrieval_recall": retrieval_recall,
        "concept_coverage": concept_coverage,
        "evidence_keyword_coverage": keyword_coverage,
        "regulation_coverage": regulation_coverage,
        "citation_coverage": answer_field(citation_coverage),
        "has_citations": answer_field(has_citations),
        "answer_relevance_score": answer_field(answer_relevance),
        "faithfulness_score": answer_field(faithfulness),
        "citation_accuracy": answer_field(citation_accuracy),
        "evidence_grounding": answer_field(evidence_grounding),
        "overall_score": answer_field(overall_score),
        "expects_abstention": expects_abstention,
        "abstention_correct": abstention_correct,
        "latency_seconds": latency_seconds,
        "status": status,
        "error": error,
        "notes": clean_text(question_item.get("notes")),
    }
    if result.get("context"):
        row["retrieved_context"] = result.get("context")
    row["retrieved_evidence"] = evidence
    row["retrieval_debug"] = debug
    # Consolidated audit trail (JSON-only, like retrieved_evidence/retrieval_debug).
    row["match_audit"] = {
        "regulations": {"expected": expected_regulations, "matched": matched_regulations, "missed": missed_regulations},
        "concepts": {"expected": expected_concepts, "matched": matched_concepts, "missed": missed_concepts},
        "evidence_keywords": {"expected": expected_keywords, "matched": matched_keywords, "missed": missed_keywords},
    }
    return row


def regulation_from_provenance(item: dict[str, Any]) -> str | None:
    """Map one retrieved item to a regulation using its source_group/source_document.

    Returns the regulation name when provenance is recognized, otherwise None.
    """
    source_group = normalize_text(item.get("source_group"))
    source_document = normalize_text(item.get("source_document"))
    for regulation in REGULATION_ALIASES:
        source_key = normalize_text(regulation).replace(" ", "_")
        # normalize_text turns "eu_ai_act" -> "eu ai act"; compare on that form.
        group_key = source_key.replace("_", " ")
        if source_group and source_group == group_key:
            return regulation
        if source_document and group_key in source_document:
            return regulation
    return None


def find_retrieved_regulations(
    evidence: list[Any],
    top_rows: list[Any],
) -> list[str]:
    """Derive retrieved regulations from evidence provenance (T2).

    Primary signal: each retrieved item's source_group/source_document. Alias text
    matching is only a fallback for items that carry no provenance at all, and is
    scoped to those items' text -- so a HIPAA alias (e.g. "phi") appearing inside a
    GDPR-sourced statement does NOT flag HIPAA as retrieved.
    """
    matched: set[str] = set()
    unsourced_text_parts: list[str] = []

    for item in [*evidence, *top_rows]:
        if not isinstance(item, dict):
            continue
        regulation = regulation_from_provenance(item)
        if regulation:
            matched.add(regulation)
        else:
            # No recognized provenance -> keep this item's text for a gated fallback.
            unsourced_text_parts.append(joined_text([item.get("evidence_text"), item.get("statement")]))

    # Fallback only: alias-match against text from items that lacked provenance.
    if unsourced_text_parts:
        fallback_text = normalize_text(joined_text(unsourced_text_parts))
        for regulation, aliases in REGULATION_ALIASES.items():
            if regulation in matched:
                continue
            if any(alias in fallback_text for alias in aliases):
                matched.add(regulation)

    # Stable output in REGULATION_ALIASES declaration order.
    return [regulation for regulation in REGULATION_ALIASES if regulation in matched]


def extract_citations(answer: str, evidence: list[Any], top_rows: list[Any]) -> list[str]:
    citations: list[str] = []
    for item in [*evidence, *top_rows]:
        if not isinstance(item, dict):
            continue
        citation = clean_text(item.get("citation"))
        if citation:
            citations.append(citation)

    for pattern in CITATION_PATTERNS:
        citations.extend(re.findall(pattern, answer, flags=re.IGNORECASE))
    return sorted(set(clean_text(citation) for citation in citations if clean_text(citation)))


def extract_source_documents(evidence: list[Any], top_rows: list[Any], debug: dict[str, Any] | None = None) -> list[str]:
    documents: list[str] = []
    selection = debug.get("selection") if isinstance(debug, dict) and isinstance(debug.get("selection"), dict) else {}
    context_row_ids = selection.get("context_row_ids_rendered") if isinstance(selection, dict) else []
    if isinstance(context_row_ids, list):
        documents.extend(_source_documents_from_context_ids(context_row_ids))
    if not documents:
        documents.extend(
            clean_text(item.get("source_document"))
            for item in [*evidence, *top_rows]
            if isinstance(item, dict) and clean_text(item.get("source_document"))
        )
    return sorted(set(documents))


def _source_documents_from_context_ids(row_ids: list[Any]) -> list[str]:
    documents: list[str] = []
    for row_id in row_ids:
        prefix = clean_text(row_id).split("|", 1)[0]
        for document in prefix.split(";"):
            cleaned = clean_text(document)
            if cleaned:
                documents.append(cleaned)
    return documents


def detect_has_citations(answer: str, citations: list[str]) -> bool:
    """Structural: whether the answer carries explicit citations.

    Mirrors the 1.0 branch of score_citation_coverage -- a citation list is present,
    or the answer references an article/section/cfr/etc. Names what the structural
    citation_coverage metric measures (citation *presence*), not semantic correctness.
    """
    if citations:
        return True
    return bool(re.search(r"\b(article|section|cfr|citation|source|evidence)\b", answer, flags=re.IGNORECASE))


def score_citation_coverage(answer: str, citations: list[str], source_documents: list[str]) -> float:
    if detect_has_citations(answer, citations):
        return 1.0
    if source_documents:
        return 0.5
    return 0.0


def score_answer_relevance(
    answer: str,
    matched_keywords: list[str],
    expected_keywords: list[str],
    matched_concepts: list[str],
    expected_concepts: list[str],
) -> float:
    if not answer.strip():
        return 0.0
    keyword_score = coverage(len(matched_keywords), len(expected_keywords))
    concept_score = coverage(len(matched_concepts), len(expected_concepts))
    return round((keyword_score * 0.6) + (concept_score * 0.4), 3)


def canonical_citation(citation: Any) -> str:
    """Collapse a citation to a comparable key.

    Reuses CITATION_PATTERNS to isolate the citation token, then drops the leading
    article/section words and every non-alphanumeric separator so surface variants
    normalize to one form: "Article 5(1)(e)", "Art. 5(1)(e)", and "GDPR Art 5.1.e"
    all become "51e", while "Article 8" stays "8".
    """
    text = clean_text(citation)
    if not text:
        return ""
    for pattern in CITATION_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE)
        if found:
            text = found.group(0)
            break
    text = text.lower()
    tail = re.search(r"\d.*", text)
    if tail:
        text = tail.group(0)
    return re.sub(r"[^a-z0-9]", "", text)


def score_citation_accuracy(citations: list[str], expected_citations: list[str]) -> float | None:
    """Fraction of expected citations found among the actual citations.

    Returns None when expected_citations is empty (question not yet annotated), so
    callers can distinguish "no ground truth" from a genuine 0.0 score. Matching is
    variant-insensitive via canonical_citation().
    """
    expected = list_values(expected_citations)
    if not expected:
        return None
    actual_keys = {canonical_citation(citation) for citation in citations}
    actual_keys.discard("")
    matched = sum(1 for citation in expected if canonical_citation(citation) in actual_keys)
    return round(matched / len(expected), 3)


def detect_abstention(result: dict[str, Any], answer: str) -> bool:
    """Whether the system declined to give a confident determination.

    Prefers the structured determination verdict when the result carries one
    (``insufficient_evidence`` / ``unconfirmed``). When no determination is
    present (e.g. retrieval-only modes), falls back to the answer text carrying a
    known abstention message (NO_EVIDENCE_MESSAGE / UNCONFIRMED_MESSAGE) or the
    "does not contain enough evidence" phrasing.
    """
    determination = result.get("determination") if isinstance(result, dict) else None
    if isinstance(determination, dict):
        verdict = clean_text(determination.get("verdict")).lower()
        if verdict:
            return verdict in ABSTENTION_VERDICTS
    normalized_answer = normalize_text(answer)
    if not normalized_answer:
        return False
    if "does not contain enough evidence" in normalized_answer:
        return True
    return any(
        normalize_text(message) in normalized_answer
        for message in (NO_EVIDENCE_MESSAGE, UNCONFIRMED_MESSAGE)
    )


def score_faithfulness(
    answer: str,
    citations: list[str],
    evidence: list[Any],
    expected_citations: list[str] | None = None,
) -> float:
    # When a question is annotated with expected citations, faithfulness is the
    # citation-accuracy score (a real discriminating signal). Otherwise fall back to
    # the structural presence-based 0/0.5/1.0 heuristic unchanged.
    accuracy = score_citation_accuracy(citations, expected_citations or [])
    if accuracy is not None:
        return accuracy
    normalized_answer = normalize_text(answer)
    if not normalized_answer or "does not contain enough evidence" in normalized_answer:
        return 0.0
    if citations and evidence:
        return 1.0
    if evidence:
        return 0.5
    return 0.0


def concept_matches(concept: str, text: str) -> bool:
    normalized_text = normalize_text(text)
    return any(keyword_matches(variant, normalized_text) for variant in concept_variants(concept))


def concept_variants(concept: str) -> list[str]:
    stripped = clean_text(concept)
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", stripped)
    snake_spaced = stripped.replace("_", " ").replace("-", " ")
    compact = re.sub(r"[\W_]+", "", stripped).lower()
    variants = {stripped, spaced, snake_spaced, compact}
    if stripped.endswith("Requirement"):
        variants.add(spaced.replace(" Requirement", ""))
    if stripped.endswith("Data"):
        variants.add(spaced.replace(" Data", " data"))
    return [variant for variant in variants if variant]


def keyword_matches(keyword: str, text: str) -> bool:
    normalized_keyword = normalize_text(keyword)
    normalized_text = normalize_text(text)
    if normalized_keyword in normalized_text:
        return True
    compact_keyword = re.sub(r"[\W_]+", "", normalized_keyword)
    compact_text = re.sub(r"[\W_]+", "", normalized_text)
    return bool(compact_keyword and compact_keyword in compact_text)


def coverage(matched_count: int, expected_count: int) -> float:
    if expected_count <= 0:
        return 1.0
    return round(matched_count / expected_count, 3)


def write_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    write_json(results, output_dir / "evaluation_results.json")
    write_csv(results, output_dir / "evaluation_results.csv")
    write_markdown(results, output_dir / "evaluation_results.md")


def load_existing_results(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "evaluation_results.json"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def clear_outputs(output_dir: Path) -> None:
    for name in ("evaluation_results.json", "evaluation_results.csv", "evaluation_results.md"):
        path = output_dir / name
        if path.exists():
            path.unlink()


def write_json(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow({column: cell_value(result.get(column)) for column in RESULT_COLUMNS})


def write_markdown(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(RESULT_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in RESULT_COLUMNS) + " |\n")
        for result in results:
            cells = [escape_markdown(cell_value(result.get(column), max_length=500)) for column in RESULT_COLUMNS]
            handle.write("| " + " | ".join(cells) + " |\n")


def cell_value(value: Any, max_length: int | None = None) -> str:
    if isinstance(value, list):
        rendered = "; ".join(clean_text(item) for item in value)
    elif isinstance(value, dict):
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        rendered = clean_text(value)
    rendered = " ".join(rendered.split())
    if max_length and len(rendered) > max_length:
        return rendered[: max_length - 3] + "..."
    return rendered


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def list_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [clean_text(item) for item in value if clean_text(item)]


def joined_text(values: list[Any]) -> str:
    return "\n".join(clean_text(value) for value in values if clean_text(value))


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(clean_text(item) for item in value if clean_text(item))
    return str(value).strip()


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value).replace("_", " ").replace("-", " ").lower()).strip()


if __name__ == "__main__":
    main()
