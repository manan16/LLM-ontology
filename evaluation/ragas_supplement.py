from __future__ import annotations

"""Supplementary RAGAS (LLM-graded) evaluation for the hybrid answer path.

Reads the deterministic evaluation output produced by ``run_evaluation.py --mode
hybrid`` (``results/hybrid/evaluation_results.json``) and computes three
*reference-free* RAGAS metrics per question -- faithfulness, answer_relevancy and
context_precision -- using Ollama's OpenAI-compatible endpoint as the judge LLM and
embedding backend. Results are written to a **separate** sibling file
(``ragas_results.json``), joined back to the deterministic rows by ``question_id``.

This is strictly supplementary: it never mutates evaluation_results.json, and the
scores are NOT folded into run_evaluation's overall_score / OVERALL_SCORE_WEIGHTS.
Each question's RAGAS call is isolated in a try/except so one failure writes null
scores for that question and the batch continues.

CLI:
    python evaluation/ragas_supplement.py --results-dir evaluation/results/hybrid
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results" / "hybrid"
INPUT_FILENAME = "evaluation_results.json"
OUTPUT_FILENAME = "ragas_results.json"

# Stable output keys -> the RAGAS metric class that produces them. Only
# reference-free metrics are used (no ground_truth required):
#   - Faithsfulness / answer_relevancy are reference-free by construction.
#   - context_precision uses the *WithoutReference* LLM variant so no ground_truth
#     column is needed.
OUTPUT_METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision")


def main() -> None:
    args = build_parser().parse_args()
    results_dir: Path = args.results_dir
    input_path = results_dir / INPUT_FILENAME
    output_path = results_dir / OUTPUT_FILENAME

    rows = load_rows(input_path)
    samples = [build_sample(row) for row in rows]
    settings = get_settings()

    llm, embeddings, metrics = build_ragas_components(
        settings,
        model=args.model,
        embedding_model=args.embedding_model,
    )

    # Fail fast: answer_relevancy needs a working embedding model. Probe it once so a
    # missing/unpulled model errors clearly here instead of silently nulling every
    # question's scores across the whole batch.
    verify_embedding_model(embeddings, args.embedding_model or settings.ragas_embedding_model)

    results: list[dict[str, Any]] = []
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        question_id = sample["question_id"]
        print(f"[{index}/{total}] RAGAS scoring {question_id or '(no id)'} ...")
        scores, error = score_sample(sample, llm, embeddings, metrics)
        if error:
            print(f"[{index}/{total}] {question_id} FAILED: {error}")
        results.append(
            {
                "question_id": question_id,
                "faithfulness": scores.get("faithfulness"),
                "answer_relevancy": scores.get("answer_relevancy"),
                "context_precision": scores.get("context_precision"),
                "error": error,
            }
        )
        write_results(results, output_path)  # incremental checkpoint

    write_results(results, output_path)
    print(f"Wrote {len(results)} RAGAS rows to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supplementary RAGAS scoring for the hybrid evaluation results (reference-free)."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory holding evaluation_results.json; ragas_results.json is written alongside it.",
    )
    parser.add_argument(
        "--model",
        help="Judge chat model served by Ollama. Defaults to Settings.ollama_model.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Embedding model served by Ollama for answer_relevancy. Defaults to Settings.ragas_embedding_model.",
    )
    return parser


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_path} not found. Run `run_evaluation.py --mode hybrid` first."
        )
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, list):
        raise ValueError(f"Expected a JSON list in {input_path}")
    return [row for row in loaded if isinstance(row, dict)]


def build_sample(row: dict[str, Any]) -> dict[str, Any]:
    """Project one deterministic result row into the RAGAS input shape.

    contexts are the per-evidence ``evidence_text`` strings (empty ones dropped).
    ground_truth is intentionally omitted -- only reference-free metrics are used.
    """
    evidence = row.get("retrieved_evidence")
    contexts: list[str] = []
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                text = item.get("evidence_text")
                if isinstance(text, str) and text.strip():
                    contexts.append(text.strip())
    return {
        "question_id": _clean(row.get("question_id")),
        "question": _clean(row.get("question")),
        "answer": _clean(row.get("answer")),
        "contexts": contexts,
    }


def build_ragas_components(
    settings: Any,
    model: str | None = None,
    embedding_model: str | None = None,
) -> tuple[Any, Any, list[Any]]:
    """Construct the Ollama-backed judge LLM, embeddings and reference-free metrics.

    Heavy optional deps (ragas, langchain-openai) are imported lazily so importing
    this module (e.g. for tests) does not require them to be installed.
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithoutReference,
            ResponseRelevancy,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise SystemExit(
            "RAGAS supplement requires optional extras. Install them with:\n"
            "    pip install 'ragas>=0.2' 'langchain-openai>=0.1'\n"
            f"(import failed: {exc})"
        )

    # Ollama exposes an OpenAI-compatible API at <base_url>/v1. Reuse the configured
    # base URL from Settings (never hardcode) and append the /v1 suffix.
    base_url = settings.ollama_base_url.rstrip("/") + "/v1"
    chat_model = model or settings.ollama_model
    embed_model = embedding_model or settings.ragas_embedding_model

    chat = ChatOpenAI(
        model=chat_model,
        base_url=base_url,
        api_key="ollama",  # placeholder; Ollama ignores the key
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        base_url=base_url,
        api_key="ollama",  # placeholder; Ollama ignores the key
        check_embedding_ctx_length=False,
    )

    llm = LangchainLLMWrapper(chat)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ]
    return llm, wrapped_embeddings, metrics


def verify_embedding_model(embeddings: Any, model: str) -> None:
    """Probe the embedding backend once before the batch loop.

    Embeds a single throwaway string; if that raises (model not pulled, endpoint
    down), print an actionable message and exit(1) rather than letting every
    question fall through to null scores.
    """
    try:
        embeddings.embed_query("ragas embedding startup check")
    except Exception as exc:  # noqa: BLE001 - startup guard, any failure is fatal
        print(
            f"RAGAS_EMBEDDING_MODEL '{model}' is not available on Ollama — "
            f"run: ollama pull {model}\n(embedding probe failed: {type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        raise SystemExit(1)


def score_sample(
    sample: dict[str, Any],
    llm: Any,
    embeddings: Any,
    metrics: list[Any],
) -> tuple[dict[str, float | None], str | None]:
    """Run RAGAS for a single question.

    Returns (scores, error). On any failure the scores map to None and the error
    message is captured so the caller can record it and continue the batch.
    """
    empty: dict[str, float | None] = {key: None for key in OUTPUT_METRIC_KEYS}
    if not sample["answer"] or not sample["contexts"]:
        return empty, "missing answer or contexts"
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate

        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=sample["question"],
                    response=sample["answer"],
                    retrieved_contexts=sample["contexts"],
                )
            ]
        )
        result = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        return _extract_scores(result, metrics), None
    except Exception as exc:  # noqa: BLE001 - isolate per-question failures
        return empty, f"{type(exc).__name__}: {exc}"


def _extract_scores(result: Any, metrics: list[Any]) -> dict[str, float | None]:
    """Map the RAGAS EvaluationResult (single sample) to our stable output keys."""
    frame = result.to_pandas()
    record = frame.iloc[0].to_dict()
    # Each metric object carries the column name RAGAS wrote (metric.name), which
    # differs from our stable output key (e.g. context precision column is
    # "llm_context_precision_without_reference").
    by_column = {
        "faithfulness": _column_for(metrics, "faithfulness"),
        "answer_relevancy": _column_for(metrics, "answer_relevancy", "response_relevancy"),
        "context_precision": _column_for(metrics, "llm_context_precision_without_reference"),
    }
    return {key: _finite(record.get(column)) for key, column in by_column.items()}


def _column_for(metrics: list[Any], *candidate_names: str) -> str | None:
    for metric in metrics:
        name = getattr(metric, "name", None)
        if name in candidate_names:
            return name
    return candidate_names[0] if candidate_names else None


def _finite(value: Any) -> float | None:
    """Coerce a RAGAS score to float, mapping NaN / non-numeric to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return round(number, 4)


def write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
