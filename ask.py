from __future__ import annotations

import argparse

from rag import AnswerGenerator, GraphRetriever, build_context
from rag.query_planner import QueryPlan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask grounded questions over the Neo4j knowledge graph.")
    parser.add_argument("question", help="Natural language compliance question to ask")
    parser.add_argument("--limit", type=int, default=30, help="Maximum retrieved graph rows to use")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved context before the answer")
    parser.add_argument("--debug-retrieval", action="store_true", help="Print query planning and retrieval diagnostics")
    parser.add_argument("--model", help="Override the Ollama model for answer generation")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    retriever = GraphRetriever()
    rows = retriever.retrieve(args.question, limit=args.limit)
    context = build_context(rows)

    if args.debug_retrieval:
        _print_debug_retrieval(retriever.last_query_plan, retriever.last_debug_rows, retriever.last_expansion_debug)
        print()

    if args.show_context:
        print("Retrieved KG context:")
        print(context or "(no context retrieved)")
        print()

    answer = AnswerGenerator(model=args.model).generate(args.question, context)
    print(answer)


def _print_debug_retrieval(
    plan: QueryPlan | None,
    rows: list[dict[str, object]],
    expansion_debug: dict[str, list[dict[str, object]]] | None = None,
) -> None:
    print("Retrieval debug:")
    if plan is None:
        print("(no query plan built)")
        return
    print(f"Category: {plan.category}")
    print(f"Intent: {plan.intent}")
    print(f"Detected phrases: {_join(plan.phrases)}")
    print(f"Detected actors: {_join(plan.detected_actors)}")
    print(f"Detected objects: {_join(plan.detected_objects)}")
    print(f"Concept groups: {plan.concept_groups or '(none)'}")
    print(f"Ambiguous terms: {_join(plan.ambiguous_terms)}")
    print(f"Expansion terms: {_join(plan.expansion_terms)}")
    print(f"Sub-questions: {_join(plan.sub_questions)}")
    if expansion_debug:
        _print_debug_bucket("Entity seeds", expansion_debug.get("seeds", []))
        _print_debug_bucket("Expanded evidence rows", expansion_debug.get("expanded", []))
        _print_debug_bucket("Excluded rows", expansion_debug.get("excluded", []))
        _print_debug_bucket("Final selected context rows", expansion_debug.get("selected", []))
    print("Top rows:")
    for index, row in enumerate(rows[:10], start=1):
        name = row.get("statement_name") or row.get("seed_name") or "(unknown)"
        route = row.get("query_route") or "(unknown route)"
        score = row.get("score") or 0
        relationship = row.get("relationship") or ""
        source_group = row.get("source_group") or "unknown"
        source_document = row.get("source_document") or ""
        matched_groups = row.get("matched_concept_groups") or []
        matched_terms = row.get("matched_terms") or []
        penalties = row.get("penalties") or []
        print(
            f"{index}. score={score} route={route} source={source_group} "
            f"source_document={source_document} statement={name} relationship={relationship} "
            f"matched={matched_groups} terms={matched_terms} penalties={penalties}"
        )


def _print_debug_bucket(label: str, rows: list[dict[str, object]]) -> None:
    print(f"{label}: {len(rows)}")
    for index, row in enumerate(rows[:5], start=1):
        name = row.get("statement") or row.get("seed") or "(unknown)"
        print(
            f"  {index}. score={row.get('score') or 0} route={row.get('route') or '(unknown route)'} "
            f"source={row.get('source_group') or 'unknown'} statement={name} "
            f"matched={row.get('matched_concept_groups') or []} penalties={row.get('penalties') or []}"
        )


def _join(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


if __name__ == "__main__":
    main()
