from __future__ import annotations

import argparse

from web.services.rag_service import answer_question


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

    result = answer_question(
        args.question,
        limit=args.limit,
        show_context=args.show_context,
        debug=args.debug_retrieval,
        model=args.model,
    )

    if args.debug_retrieval:
        _print_debug_retrieval(result.get("debug") or {})
        print()

    if args.show_context:
        print("Retrieved KG context:")
        print(result.get("context") or "(no context retrieved)")
        print()

    print(result.get("answer") or "")


def _print_debug_retrieval(debug: dict[str, object]) -> None:
    print("Retrieval debug:")
    plan = debug.get("query_plan") if isinstance(debug.get("query_plan"), dict) else {}
    if not plan:
        print("(no query plan built)")
        return
    print(f"Category: {plan.get('category')}")
    print(f"Intent: {plan.get('intent')}")
    print(f"Detected phrases: {_join(plan.get('detected_phrases'))}")
    print(f"Detected actors: {_join(plan.get('detected_actors'))}")
    print(f"Detected objects: {_join(plan.get('detected_objects'))}")
    print(f"Concept groups: {plan.get('concept_groups') or '(none)'}")
    print(f"Ambiguous terms: {_join(plan.get('ambiguous_terms'))}")
    print(f"Expansion terms: {_join(plan.get('expansion_terms'))}")
    print(f"Sub-questions: {_join(plan.get('sub_questions'))}")
    print(f"Mental-health query: {'yes' if plan.get('is_mental_health_query') else 'no'}")
    print(f"Mental-health intent labels: {_join(plan.get('mental_health_intent_labels'))}")
    print(f"Mental-health expansion terms used: {_join(plan.get('mental_health_expansion_terms'))}")
    print(f"Explicit regulations: {_join(plan.get('explicit_regulations'))}")
    print(f"Inferred regulations: {_join(plan.get('inferred_regulations'))}")
    final_regulations = plan.get("explicit_regulations") or plan.get("inferred_regulations")
    print(f"Final target regulations: {_join(final_regulations)}")
    print(f"Target source documents: {_join(plan.get('target_source_documents'))}")
    expansion_debug = debug.get("expansion") if isinstance(debug.get("expansion"), dict) else {}
    selection_debug = debug.get("selection") if isinstance(debug.get("selection"), dict) else {}
    if selection_debug:
        print(
            "Selection: "
            f"concrete_rows_selected={selection_debug.get('concrete_rows_selected', 0)} "
            f"broad_rows_excluded={selection_debug.get('broad_rows_excluded', False)} "
            f"broad_rows_excluded_count={selection_debug.get('broad_rows_excluded_count', 0)}"
        )
        print(f"Primary-source rows found: {selection_debug.get('primary_source_rows_found', 0)}")
        print(f"Fallback used: {'yes' if selection_debug.get('fallback_used') else 'no'}")
        print(f"Rows excluded due to wrong source: {selection_debug.get('wrong_source_rows_excluded', 0)}")
        print(f"Rows penalized as secondary references: {selection_debug.get('secondary_reference_rows_penalized', 0)}")
        print(f"Per-source retrieval counts: {selection_debug.get('per_source_retrieval_counts') or {}}")
        print(f"Per-source selected counts: {selection_debug.get('per_source_selected_counts') or {}}")
        print(f"Coverage sources present in final context: {_join(selection_debug.get('coverage_sources_present'))}")
        print(f"Coverage sources missing from final context: {_join(selection_debug.get('coverage_sources_missing'))}")
        print(f"Final selected row IDs: {_join(selection_debug.get('final_selected_row_ids'))}")
        print(f"Final context row IDs: {_join(selection_debug.get('context_row_ids_rendered'))}")
    if expansion_debug:
        _print_debug_bucket("Entity seeds", expansion_debug.get("seeds", []))
        _print_debug_bucket("Expanded evidence rows", expansion_debug.get("expanded", []))
        _print_debug_bucket("Excluded rows", expansion_debug.get("excluded", []))
        _print_debug_bucket("Final selected context rows", expansion_debug.get("selected", []))
    print("Top rows:")
    rows = debug.get("top_rows") if isinstance(debug.get("top_rows"), list) else []
    for index, row in enumerate(rows[:10], start=1):
        name = row.get("statement") or "(unknown)"
        route = row.get("route") or "(unknown route)"
        score = row.get("score") or 0
        relationship = row.get("relationship") or ""
        source_group = row.get("source_group") or "unknown"
        source_document = row.get("source_document") or ""
        matched_groups = row.get("matched_concept_groups") or []
        matched_terms = row.get("matched_terms") or []
        topic_groups = row.get("matched_topic_groups") or []
        boosts = row.get("boosts") or []
        penalties = row.get("penalties") or []
        print(
            f"{index}. score={score} route={route} source={source_group} "
            f"source_document={source_document} statement={name} relationship={relationship} "
            f"matched={matched_groups} topics={topic_groups} boosts={boosts} terms={matched_terms} penalties={penalties}"
        )


def _print_debug_bucket(label: str, rows: list[dict[str, object]]) -> None:
    print(f"{label}: {len(rows)}")
    for index, row in enumerate(rows[:5], start=1):
        name = row.get("statement") or row.get("seed") or "(unknown)"
        print(
            f"  {index}. score={row.get('score') or 0} route={row.get('route') or '(unknown route)'} "
            f"source={row.get('source_group') or 'unknown'} statement={name} "
            f"matched={row.get('matched_concept_groups') or []} topics={row.get('matched_topic_groups') or []} "
            f"boosts={row.get('boosts') or []} penalties={row.get('penalties') or []}"
        )


def _join(values: object) -> str:
    if not isinstance(values, list):
        return "(none)"
    return ", ".join(values) if values else "(none)"


if __name__ == "__main__":
    main()
