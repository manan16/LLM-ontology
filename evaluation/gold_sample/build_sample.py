"""Build the manually-annotated extraction evaluation sample (Task 1 / H1).

This script is measurement-only. It does NOT modify the extraction pipeline.
It reuses the exact production chunker (ingestion.chunking.build_chunks with the
configured CHUNK_MAX_CHARS / CHUNK_OVERLAP_CHARS) and the exact production
extractor (extraction.extractor.RegulationExtractor) so the sample reflects what
the pipeline actually produces.

Stages:
  select     -> choose a stratified, obligation-bearing, non-duplicate sample and
                print it for review (no model calls).
  extract    -> run the real extractor on the selected chunks, writing
                raw_extractions.json and a pre-filled draft_annotations.json.
                Every draft field carries needs_review=true: the model output is a
                STARTING POINT for human annotation, never ground truth.

Usage:
  kep/bin/python evaluation/gold_sample/build_sample.py select --per-reg 5
  kep/bin/python evaluation/gold_sample/build_sample.py extract --per-reg 5
  kep/bin/python evaluation/gold_sample/build_sample.py extract --per-reg 5 --sample-size 100 --raw-only --out-raw evaluation/gold_sample/raw_extractions_100.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.config import get_settings
from extraction.extractor import RegulationExtractor
from extraction.ollama_client import OllamaClient
from ingestion.chunking import DocumentChunk, build_chunks
from ingestion.loaders import load_document


ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = ROOT / "evaluation" / "gold_sample"

# The three source regulations, keyed by the label used for stratification and
# mapped to the canonical document name produced by ingestion.loaders.
REGULATIONS: dict[str, str] = {
    "GDPR": "data/gdpr.pdf",
    "HIPAA": "data/hipaa.pdf",
    "EU_AI_ACT": "data/eu_ai_act.pdf",
}

# Deontic cue patterns. A chunk must contain at least one obligation/prohibition/
# permission cue to be eligible for the sample.
OBLIGATION_CUES = [r"\bshall\b", r"\bmust\b", r"\brequired to\b", r"\bobligated to\b", r"\bshall ensure\b"]
PROHIBITION_CUES = [r"\bshall not\b", r"\bmust not\b", r"\bprohibited\b", r"\bforbidden\b", r"\bmay not\b"]
PERMISSION_CUES = [r"\bmay\b", r"\bpermitted\b", r"\ballowed\b", r"\bmay be\b"]

_ALL_CUES = [re.compile(p, re.IGNORECASE) for p in OBLIGATION_CUES + PROHIBITION_CUES + PERMISSION_CUES]
_OBLIG = [re.compile(p, re.IGNORECASE) for p in OBLIGATION_CUES]
_PROHIB = [re.compile(p, re.IGNORECASE) for p in PROHIBITION_CUES]
_PERMIT = [re.compile(p, re.IGNORECASE) for p in PERMISSION_CUES]

_WORD = re.compile(r"[a-z0-9]+")


def _cue_counts(text: str) -> tuple[int, int, int]:
    o = sum(len(p.findall(text)) for p in _OBLIG)
    pr = sum(len(p.findall(text)) for p in _PROHIB)
    pe = sum(len(p.findall(text)) for p in _PERMIT)
    return o, pr, pe


def _token_set(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def select_chunks(per_reg: int, dup_threshold: float = 0.6) -> dict[str, list[DocumentChunk]]:
    """Select `per_reg` obligation-bearing, non-duplicate chunks per regulation.

    Deterministic: chunks are ranked by total deontic-cue count (desc), then by
    chunk_id (asc) as a stable tiebreaker. A candidate is skipped if its token
    Jaccard similarity to any already-selected chunk in the same regulation is
    >= dup_threshold (near-duplicate guard, e.g. overlap windows or repeated
    boilerplate).
    """
    settings = get_settings()
    selection: dict[str, list[DocumentChunk]] = {}

    for label, rel_path in REGULATIONS.items():
        document = load_document(ROOT / rel_path)
        chunks = build_chunks(
            document=document,
            max_chars=settings.chunk_max_chars,
            overlap_chars=settings.chunk_overlap_chars,
        )
        # Eligible = contains at least one deontic cue.
        eligible = [c for c in chunks if any(p.search(c.text) for p in _ALL_CUES)]
        # Rank by total cue count desc, then chunk_id asc (deterministic).
        eligible.sort(key=lambda c: (-sum(_cue_counts(c.text)), c.chunk_id))

        picked: list[DocumentChunk] = []
        picked_tokens: list[set[str]] = []
        for cand in eligible:
            toks = _token_set(cand.text)
            if any(_jaccard(toks, prev) >= dup_threshold for prev in picked_tokens):
                continue
            picked.append(cand)
            picked_tokens.append(toks)
            if len(picked) >= per_reg:
                break

        if len(picked) < per_reg:
            raise RuntimeError(
                f"Only found {len(picked)} eligible non-duplicate chunks for {label}, need {per_reg}"
            )
        selection[label] = picked
    return selection


def _chunk_meta(label: str, chunk: DocumentChunk) -> dict:
    o, pr, pe = _cue_counts(chunk.text)
    return {
        "regulation": label,
        "chunk_id": chunk.chunk_id,
        "regulation_name": chunk.regulation_name,
        "section_id": chunk.section_id,
        "section_title": chunk.section_title,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "char_len": len(chunk.text),
        "cue_counts": {"obligation": o, "prohibition": pr, "permission": pe},
        "text": chunk.text,
    }


def cmd_select(args: argparse.Namespace) -> None:
    selection = select_chunks(args.per_reg, args.dup_threshold)
    for label, chunks in selection.items():
        print(f"\n===== {label} ({len(chunks)} chunks) =====")
        for c in chunks:
            o, pr, pe = _cue_counts(c.text)
            preview = re.sub(r"\s+", " ", c.text)[:180]
            print(f"[{c.chunk_id}] {c.section_title[:50]!r} len={len(c.text)} "
                  f"cues(o/pr/pe)={o}/{pr}/{pe}")
            print(f"    {preview}...")


def _draft_from_statement(stmt: dict) -> dict:
    """Build a draft Appendix B.1 annotation entry from one model statement.

    Every field is a pre-fill only; needs_review=true flags that nothing here is
    human-validated. Field names follow Appendix B.1: source_locator, evidence_text,
    statement_type, nodes[], relationships[], ambiguity_note.
    """
    def loc(d: dict) -> str:
        parts = [d.get("source_document"), d.get("article_number"), d.get("clause_number"),
                 d.get("section_id"), d.get("section_title")]
        return " | ".join(str(p) for p in parts if p)

    nodes = [
        {
            "node_type": n.get("node_type"),
            "canonical_name": n.get("canonical_name"),
            "raw_text": n.get("raw_text"),
        }
        for n in stmt.get("nodes", [])
    ]
    relationships = [
        {
            "relationship_type": r.get("relationship_type"),
            "source_canonical_name": r.get("source_canonical_name"),
            "source_node_type": r.get("source_node_type"),
            "target_canonical_name": r.get("target_canonical_name"),
            "target_node_type": r.get("target_node_type"),
        }
        for r in stmt.get("relationships", [])
    ]
    return {
        "source_locator": loc(stmt),
        "evidence_text": stmt.get("evidence_text"),
        "statement_type": stmt.get("statement_type"),
        "nodes": nodes,
        "relationships": relationships,
        "ambiguity_note": "",
        "needs_review": True,
    }


def cmd_extract(args: argparse.Namespace) -> None:
    settings = get_settings()
    selection = select_chunks(args.per_reg if not args.sample_size else args.sample_size // 3,
                              args.dup_threshold) if args.sample_size else select_chunks(
        args.per_reg, args.dup_threshold)

    # Flatten in stratified order.
    ordered: list[tuple[str, DocumentChunk]] = []
    for label, chunks in selection.items():
        for c in chunks:
            ordered.append((label, c))

    client = OllamaClient(settings)
    extractor = RegulationExtractor(client)

    raw_records: list[dict] = []
    draft_records: list[dict] = []

    for idx, (label, chunk) in enumerate(ordered, start=1):
        print(f"[{idx}/{len(ordered)}] extracting {label} {chunk.chunk_id} "
              f"(len={len(chunk.text)}) ...", flush=True)
        try:
            result = extractor.extract_chunk(chunk)
            result_dict = result.model_dump(mode="json")
            error = None
        except Exception as exc:  # noqa: BLE001 - record failure, keep going
            result_dict = None
            error = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {error}", flush=True)

        raw_records.append({
            "regulation": label,
            "chunk": _chunk_meta(label, chunk),
            "extraction": result_dict,
            "extraction_error": error,
            "model": settings.ollama_model,
        })

        if result_dict is not None:
            draft_records.append({
                "regulation": label,
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "chunk_text": chunk.text,
                "chunk_summary_draft": result_dict.get("chunk_summary"),
                "annotations": [_draft_from_statement(s) for s in result_dict.get("statements", [])],
                "needs_review": True,
                "reviewer_notes": "UNVALIDATED DRAFT - pre-filled from model output. Add missed "
                                  "statements/nodes/relationships, fix wrong types, remove "
                                  "hallucinated items, resolve ambiguity_note.",
            })
        else:
            draft_records.append({
                "regulation": label,
                "chunk_id": chunk.chunk_id,
                "section_title": chunk.section_title,
                "chunk_text": chunk.text,
                "chunk_summary_draft": None,
                "annotations": [],
                "needs_review": True,
                "reviewer_notes": f"UNVALIDATED DRAFT - extraction FAILED ({error}). Annotate "
                                  "expected statements manually.",
            })

    out_raw = Path(args.out_raw)
    out_raw.parent.mkdir(parents=True, exist_ok=True)
    out_raw.write_text(json.dumps({
        "description": "Raw extraction output from the production pipeline on the selected "
                       "sample. System output for H1/H2 scoring. NOT ground truth.",
        "model": settings.ollama_model,
        "chunk_max_chars": settings.chunk_max_chars,
        "chunk_overlap_chars": settings.chunk_overlap_chars,
        "count": len(raw_records),
        "records": raw_records,
    }, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_raw} ({len(raw_records)} records)")

    if not args.raw_only:
        out_draft = Path(args.out_draft)
        out_draft.write_text(json.dumps({
            "description": "DRAFT annotations pre-filled from model output as a starting point. "
                           "NOTHING here is human-validated. needs_review=true on every record and "
                           "field. Correct chunk-by-chunk, then save as gold_annotations.json.",
            "schema": "Appendix B.1 (source_locator / evidence_text / statement_type / nodes / "
                      "relationships / ambiguity_note)",
            "validated": False,
            "count": len(draft_records),
            "records": draft_records,
        }, indent=2, ensure_ascii=False))
        print(f"Wrote {out_draft} ({len(draft_records)} records) - UNVALIDATED")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--per-reg", type=int, default=5, help="chunks per regulation")
    common.add_argument("--dup-threshold", type=float, default=0.6,
                        help="token Jaccard >= this counts as a near-duplicate and is skipped")

    p_sel = sub.add_parser("select", parents=[common], help="select and print sample (no model)")
    p_sel.set_defaults(func=cmd_select)

    p_ext = sub.add_parser("extract", parents=[common], help="run extractor on the sample")
    p_ext.add_argument("--sample-size", type=int, default=0,
                       help="if >0, total chunks (split 3 ways) for the unannotated H2 sample")
    p_ext.add_argument("--raw-only", action="store_true", help="skip writing draft annotations")
    p_ext.add_argument("--out-raw", default=str(GOLD_DIR / "raw_extractions.json"))
    p_ext.add_argument("--out-draft", default=str(GOLD_DIR / "draft_annotations.json"))
    p_ext.set_defaults(func=cmd_extract)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
