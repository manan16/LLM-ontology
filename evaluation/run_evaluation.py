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

from web.services.rag_service import answer_question


DEFAULT_INPUT = Path(__file__).resolve().parent / "golden_questions.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

RESULT_COLUMNS = [
    "question_id",
    "category",
    "question",
    "expected_regulations",
    "retrieved_regulations",
    "expected_concepts",
    "matched_concepts",
    "expected_evidence_keywords",
    "matched_evidence_keywords",
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
    "answer_relevance_score",
    "faithfulness_score",
    "overall_score",
    "latency_seconds",
    "status",
    "error",
    "notes",
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.append:
        results = load_existing_results(args.output_dir)
    else:
        clear_outputs(args.output_dir)
        results = []
    total = len(questions)
    for index, question in enumerate(questions, start=1):
        question_id = clean_text(question.get("id")) or f"question_{index}"
        print(f"[{index}/{total}] Evaluating {question_id}: {clean_text(question.get('question'))}")
        row = evaluate_question(question, limit=args.limit, model=args.model, include_context=args.include_context)
        results.append(row)
        write_results(results, args.output_dir)
        print(
            f"[{index}/{total}] {question_id} status={row['status']} "
            f"overall_score={row['overall_score']} latency_seconds={row['latency_seconds']}"
        )
    write_results(results, args.output_dir)
    print(f"Wrote {len(results)} evaluation rows to {args.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline golden-question evaluation for the KG-RAG QA system.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to golden questions JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV/Markdown/JSON outputs.")
    parser.add_argument("--limit", type=int, default=30, help="Maximum retrieved graph rows per question.")
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
) -> dict[str, Any]:
    started_at = perf_counter()
    error = ""
    status = "ok"
    result: dict[str, Any] = EMPTY_RESULT

    try:
        result = answer_question(
            str(question_item.get("question") or ""),
            limit=limit,
            show_context=include_context,
            debug=True,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - exercised in real integration runs
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    latency_seconds = round(perf_counter() - started_at, 3)
    return build_result_row(question_item, result, latency_seconds, status, error)


def build_result_row(
    question_item: dict[str, Any],
    result: dict[str, Any],
    latency_seconds: float,
    status: str = "ok",
    error: str = "",
) -> dict[str, Any]:
    answer = clean_text(result.get("answer"))
    evidence = result.get("evidence") if isinstance(result.get("evidence"), list) else []
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    top_rows = debug.get("top_rows") if isinstance(debug.get("top_rows"), list) else []

    evidence_text = joined_text(
        [
            *(item.get("evidence_text") for item in evidence if isinstance(item, dict)),
            *(item.get("statement") for item in evidence if isinstance(item, dict)),
        ]
    )
    metadata_text = joined_text([json.dumps(item, ensure_ascii=False, default=str) for item in [*evidence, *top_rows]])
    searchable_text = joined_text([answer, evidence_text, metadata_text])

    expected_regulations = list_values(question_item.get("expected_regulations"))
    expected_concepts = list_values(question_item.get("expected_concepts"))
    expected_keywords = list_values(question_item.get("expected_evidence_keywords"))

    retrieved_regulations = find_retrieved_regulations(searchable_text, evidence, top_rows)
    matched_regulations = [reg for reg in expected_regulations if reg in retrieved_regulations]
    matched_concepts = [concept for concept in expected_concepts if concept_matches(concept, searchable_text)]
    matched_keywords = [keyword for keyword in expected_keywords if keyword_matches(keyword, searchable_text)]

    citations = extract_citations(answer, evidence, top_rows)
    source_documents = extract_source_documents(evidence, top_rows, debug)
    plan_debug = debug.get("query_plan") if isinstance(debug.get("query_plan"), dict) else {}
    selection_debug = debug.get("selection") if isinstance(debug.get("selection"), dict) else {}
    regulation_coverage = coverage(len(matched_regulations), len(expected_regulations))
    concept_coverage = coverage(len(matched_concepts), len(expected_concepts))
    keyword_coverage = coverage(len(matched_keywords), len(expected_keywords))
    citation_coverage = score_citation_coverage(answer, citations, source_documents)
    retrieval_recall = round((regulation_coverage + concept_coverage + keyword_coverage) / 3, 3)
    answer_relevance = score_answer_relevance(answer, matched_keywords, expected_keywords, matched_concepts, expected_concepts)
    faithfulness = score_faithfulness(answer, citations, evidence)
    overall_score = round(
        (
            retrieval_recall
            + concept_coverage
            + keyword_coverage
            + regulation_coverage
            + citation_coverage
            + answer_relevance
            + faithfulness
        )
        / 7,
        3,
    )

    row = {
        "question_id": clean_text(question_item.get("id")),
        "category": clean_text(question_item.get("category")),
        "question": clean_text(question_item.get("question")),
        "expected_regulations": expected_regulations,
        "retrieved_regulations": retrieved_regulations,
        "expected_concepts": expected_concepts,
        "matched_concepts": matched_concepts,
        "expected_evidence_keywords": expected_keywords,
        "matched_evidence_keywords": matched_keywords,
        "answer": answer,
        "citations": citations,
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
        "citation_coverage": citation_coverage,
        "answer_relevance_score": answer_relevance,
        "faithfulness_score": faithfulness,
        "overall_score": overall_score,
        "latency_seconds": latency_seconds,
        "status": status,
        "error": error,
        "notes": clean_text(question_item.get("notes")),
    }
    if result.get("context"):
        row["retrieved_context"] = result.get("context")
    row["retrieved_evidence"] = evidence
    row["retrieval_debug"] = debug
    return row


def find_retrieved_regulations(
    searchable_text: str,
    evidence: list[Any],
    top_rows: list[Any],
) -> list[str]:
    text = normalize_text(searchable_text)
    source_groups = {
        clean_text(item.get("source_group")).lower()
        for item in [*evidence, *top_rows]
        if isinstance(item, dict)
    }
    matched: list[str] = []
    for regulation, aliases in REGULATION_ALIASES.items():
        source_key = regulation.lower().replace(" ", "_")
        if source_key in source_groups or any(alias in text for alias in aliases):
            matched.append(regulation)
    return matched


def extract_citations(answer: str, evidence: list[Any], top_rows: list[Any]) -> list[str]:
    citations: list[str] = []
    for item in [*evidence, *top_rows]:
        if not isinstance(item, dict):
            continue
        citation = clean_text(item.get("citation"))
        if citation:
            citations.append(citation)

    citation_patterns = [
        r"\bArticle\s+\d+[A-Za-z0-9()\-]*",
        r"\bArt\.\s*\d+[A-Za-z0-9()\-]*",
        r"\b\d+\s*CFR\s*[\d.]+",
        r"\b45\s*CFR\s*[\d.]+",
        r"\[[^\]]+\]",
    ]
    for pattern in citation_patterns:
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


def score_citation_coverage(answer: str, citations: list[str], source_documents: list[str]) -> float:
    if citations or re.search(r"\b(article|section|cfr|citation|source|evidence)\b", answer, flags=re.IGNORECASE):
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


def score_faithfulness(answer: str, citations: list[str], evidence: list[Any]) -> float:
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
