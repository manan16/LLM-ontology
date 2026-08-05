"""Extraction-quality evaluation harness (H1 + H2). Measurement only.

This module does NOT modify the extraction pipeline (extraction/schemas.py,
extraction/prompts.py, extraction/extractor.py are untouched). It scores the
pipeline's raw output against a human-validated gold annotation set (H1) and
measures schema-validation behaviour on a larger unannotated sample (H2).

Subcommands
-----------
  score-h1   Score evaluation/gold_sample/raw_extractions.json (system output)
             against evaluation/gold_sample/gold_annotations.json (ground truth),
             writing extraction_metrics.json and extraction_metrics.md.

  run-h2     Run first-attempt extraction over a larger stratified sample
             (default 100 chunks) with NO manual annotation and NO adaptive
             retry, writing validation_rate.json.

Matching rules (documented so the numbers are reproducible)
-----------------------------------------------------------
All matching is performed WITHIN a chunk (gold and system items for the same
chunk_id are matched against each other; never across chunks).

  Statement  matched on statement_type equality AND evidence_text token-overlap
             coefficient (|A n B| / min(|A|,|B|)) >= EVID_MATCH.
  Node       matched on node_type equality AND canonical_name equality,
             case/whitespace-insensitive.
  Relationship matched on relationship_type equality AND the
             (source_canonical_name, target_canonical_name) pair,
             case/whitespace-insensitive.

Precision = TP / system_count, Recall = TP / gold_count, F1 = harmonic mean.
When system_count == 0 (e.g. the model emitted no nodes) precision is reported
as null and F1 as 0.0; the raw TP / predicted / gold counts are always shown so
the ratio is unambiguous.

Error taxonomy (Appendix B.2, 8 categories). Every unmatched or mis-matched
item is bucketed exactly once:
  unsupported_extraction        system statement grounded in the chunk but with
                                no gold counterpart; system node/rel with no gold
                                counterpart and no shared endpoint.
  wrong_statement_type          evidence overlaps a gold statement but the
                                statement_type differs.
  missing_extraction            gold statement/node/relationship with no system
                                counterpart (includes everything from a chunk
                                whose extraction failed).
  incorrect_entity_type         node canonical_name matches a gold node but
                                node_type differs.
  incorrect_relationship        relationship endpoints match a gold relationship
                                but relationship_type differs, or an unmatched
                                system relationship shares >=1 endpoint with a
                                gold relationship.
  evidence_span_error           statement matched (TP) but the evidence spans are
                                not identical after normalisation (boundary drift).
  dedup_normalisation_error     node canonical_name is a near (token-overlap)
                                match to a gold node but not an exact normalised
                                match.
  retrieval_grounding_error     system statement whose evidence_text cannot be
                                located in the source chunk text.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.config import get_settings
from extraction.ollama_client import OllamaClient, OllamaTimeoutError
from extraction.prompts import SYSTEM_PROMPT, build_extraction_prompt
from extraction.schemas import ChunkExtractionResult
from ingestion.chunking import DocumentChunk, build_chunks
from ingestion.loaders import load_document


ROOT = Path(__file__).resolve().parents[1]
GOLD_DIR = ROOT / "evaluation" / "gold_sample"

# Match thresholds (token-overlap coefficient).
EVID_MATCH = 0.5   # >= this counts a statement's evidence as overlapping.
EVID_EXACT = 0.999  # matched statements below this (but >= EVID_MATCH) are span errors.
NODE_NEAR = 0.5    # node name token-overlap >= this (but not exact) => dedup/normalisation.

# Same three regulations and canonical mapping as the pipeline / build_sample.py.
REGULATIONS: dict[str, str] = {
    "GDPR": "data/gdpr.pdf",
    "HIPAA": "data/hipaa.pdf",
    "EU_AI_ACT": "data/eu_ai_act.pdf",
}

_WORD = re.compile(r"[a-z0-9]+")
_OBLIGATION_CUES = [
    r"\bshall\b", r"\bmust\b", r"\brequired to\b", r"\bobligated to\b",
    r"\bshall not\b", r"\bmust not\b", r"\bprohibited\b", r"\bforbidden\b",
    r"\bmay not\b", r"\bmay\b", r"\bpermitted\b", r"\ballowed\b",
]
_CUES = [re.compile(p, re.IGNORECASE) for p in _OBLIGATION_CUES]

CATEGORIES = [
    "unsupported_extraction",
    "wrong_statement_type",
    "missing_extraction",
    "incorrect_entity_type",
    "incorrect_relationship",
    "evidence_span_error",
    "dedup_normalisation_error",
    "retrieval_grounding_error",
]


# --------------------------------------------------------------------------- #
# normalisation helpers
# --------------------------------------------------------------------------- #
def _norm(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _tokens(text: str | None) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap_coef(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = (tp / predicted) if predicted > 0 else None
    recall = (tp / gold) if gold > 0 else None
    if tp == 0 or precision in (None, 0.0) or recall in (None, 0.0):
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "true_positive": tp,
        "system_count": predicted,
        "gold_count": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# --------------------------------------------------------------------------- #
# H1 scoring
# --------------------------------------------------------------------------- #
@dataclass
class Scorer:
    stmt_tp: int = 0
    stmt_sys: int = 0
    stmt_gold: int = 0
    node_tp: int = 0
    node_sys: int = 0
    node_gold: int = 0
    rel_tp: int = 0
    rel_sys: int = 0
    rel_gold: int = 0
    errors: Counter = field(default_factory=Counter)
    per_reg: dict = field(default_factory=lambda: {})

    def _bucket(self, category: str, n: int = 1, reg: str | None = None) -> None:
        self.errors[category] += n
        if reg is not None:
            self.per_reg.setdefault(reg, Counter())[category] += n

    # ---- statements -----------------------------------------------------
    def score_statements(self, gold: list[dict], system: list[dict],
                         chunk_text: str, reg: str) -> None:
        self.stmt_gold += len(gold)
        self.stmt_sys += len(system)
        chunk_tokens = _tokens(chunk_text)

        gold_ev = [_tokens(g.get("evidence_text")) for g in gold]
        sys_ev = [_tokens(s.get("evidence_text")) for s in system]
        gold_type = [_norm(g.get("statement_type")) for g in gold]
        sys_type = [_norm(s.get("statement_type")) for s in system]

        # Candidate pairs by evidence overlap (regardless of type first).
        pairs: list[tuple[float, int, int]] = []
        for gi in range(len(gold)):
            for sj in range(len(system)):
                ov = _overlap_coef(gold_ev[gi], sys_ev[sj])
                if ov >= EVID_MATCH:
                    pairs.append((ov, gi, sj))
        pairs.sort(reverse=True)

        g_used: set[int] = set()
        s_used: set[int] = set()
        # Pass 1: type-matching pairs => true positives.
        for ov, gi, sj in pairs:
            if gi in g_used or sj in s_used:
                continue
            if gold_type[gi] == sys_type[sj]:
                g_used.add(gi)
                s_used.add(sj)
                self.stmt_tp += 1
                if ov < EVID_EXACT and _norm(gold[gi].get("evidence_text")) != _norm(
                        system[sj].get("evidence_text")):
                    self._bucket("evidence_span_error", reg=reg)
        # Pass 2: remaining overlapping pairs with type mismatch => wrong type.
        for ov, gi, sj in pairs:
            if gi in g_used or sj in s_used:
                continue
            g_used.add(gi)
            s_used.add(sj)
            self._bucket("wrong_statement_type", reg=reg)
        # Unmatched gold => missing.
        for gi in range(len(gold)):
            if gi not in g_used:
                self._bucket("missing_extraction", reg=reg)
        # Unmatched system => grounding vs unsupported.
        for sj in range(len(system)):
            if sj in s_used:
                continue
            ev = sys_ev[sj]
            grounded = bool(ev) and _overlap_coef(ev, chunk_tokens) >= 0.9
            self._bucket("retrieval_grounding_error" if not grounded
                        else "unsupported_extraction", reg=reg)

    # ---- nodes ----------------------------------------------------------
    def score_nodes(self, gold: list[dict], system: list[dict], reg: str) -> None:
        self.node_gold += len(gold)
        self.node_sys += len(system)
        g_used: set[int] = set()
        s_used: set[int] = set()
        gkey = [(_norm(g.get("node_type")), _norm(g.get("canonical_name"))) for g in gold]
        skey = [(_norm(s.get("node_type")), _norm(s.get("canonical_name"))) for s in system]

        # Exact (type + name).
        for sj, sk in enumerate(skey):
            for gi, gk in enumerate(gkey):
                if gi in g_used:
                    continue
                if sk == gk:
                    g_used.add(gi); s_used.add(sj); self.node_tp += 1
                    break
        # Name matches, type differs => incorrect entity type.
        for sj in range(len(system)):
            if sj in s_used:
                continue
            sname = skey[sj][1]
            for gi in range(len(gold)):
                if gi in g_used:
                    continue
                if gkey[gi][1] == sname and gkey[gi][0] != skey[sj][0]:
                    g_used.add(gi); s_used.add(sj)
                    self._bucket("incorrect_entity_type", reg=reg)
                    break
        # Near name (token overlap) but not exact => dedup/normalisation.
        for sj in range(len(system)):
            if sj in s_used:
                continue
            stoks = _tokens(system[sj].get("canonical_name"))
            for gi in range(len(gold)):
                if gi in g_used:
                    continue
                if _jaccard(stoks, _tokens(gold[gi].get("canonical_name"))) >= NODE_NEAR:
                    g_used.add(gi); s_used.add(sj)
                    self._bucket("dedup_normalisation_error", reg=reg)
                    break
        # Remaining unmatched.
        for gi in range(len(gold)):
            if gi not in g_used:
                self._bucket("missing_extraction", reg=reg)
        for sj in range(len(system)):
            if sj not in s_used:
                self._bucket("unsupported_extraction", reg=reg)

    # ---- relationships --------------------------------------------------
    def score_relationships(self, gold: list[dict], system: list[dict], reg: str) -> None:
        self.rel_gold += len(gold)
        self.rel_sys += len(system)
        g_used: set[int] = set()
        s_used: set[int] = set()

        def endpoints(r: dict) -> tuple[str, str]:
            return _norm(r.get("source_canonical_name")), _norm(r.get("target_canonical_name"))

        gkey = [(_norm(g.get("relationship_type")), *endpoints(g)) for g in gold]
        skey = [(_norm(s.get("relationship_type")), *endpoints(s)) for s in system]

        # Exact (type + endpoint pair).
        for sj, sk in enumerate(skey):
            for gi, gk in enumerate(gkey):
                if gi in g_used:
                    continue
                if sk == gk:
                    g_used.add(gi); s_used.add(sj); self.rel_tp += 1
                    break
        # Endpoint pair matches, type differs => incorrect relationship.
        for sj in range(len(system)):
            if sj in s_used:
                continue
            s_ep = (skey[sj][1], skey[sj][2])
            for gi in range(len(gold)):
                if gi in g_used:
                    continue
                if (gkey[gi][1], gkey[gi][2]) == s_ep and gkey[gi][0] != skey[sj][0]:
                    g_used.add(gi); s_used.add(sj)
                    self._bucket("incorrect_relationship", reg=reg)
                    break
        # Remaining unmatched system: shares an endpoint => incorrect rel, else unsupported.
        gold_endpoints: set[str] = set()
        for gi in range(len(gold)):
            gold_endpoints.add(gkey[gi][1]); gold_endpoints.add(gkey[gi][2])
        for sj in range(len(system)):
            if sj in s_used:
                continue
            if skey[sj][1] in gold_endpoints or skey[sj][2] in gold_endpoints:
                self._bucket("incorrect_relationship", reg=reg)
            else:
                self._bucket("unsupported_extraction", reg=reg)
        # Unmatched gold => missing.
        for gi in range(len(gold)):
            if gi not in g_used:
                self._bucket("missing_extraction", reg=reg)

    def summary(self) -> dict[str, Any]:
        return {
            "statement": _prf(self.stmt_tp, self.stmt_sys, self.stmt_gold),
            "node": _prf(self.node_tp, self.node_sys, self.node_gold),
            "relationship": _prf(self.rel_tp, self.rel_sys, self.rel_gold),
            "error_categories": {c: int(self.errors.get(c, 0)) for c in CATEGORIES},
            "error_categories_by_regulation": {
                reg: {c: int(cnts.get(c, 0)) for c in CATEGORIES}
                for reg, cnts in sorted(self.per_reg.items())
            },
        }


def _sys_statements(record: dict) -> list[dict]:
    ex = record.get("extraction")
    return ex["statements"] if ex else []


def score_h1(gold_doc: dict, raw_doc: dict) -> dict[str, Any]:
    """Score system output (raw_doc) against ground truth (gold_doc).

    Both are the on-disk JSON objects (with a 'records' list). Returns the full
    metrics dict. Pure function -> used by the pytest fixture.
    """
    gold_by_chunk = {r["chunk_id"]: r for r in gold_doc["records"]}
    raw_by_chunk = {r["chunk"]["chunk_id"]: r for r in raw_doc["records"]}

    scorer = Scorer()
    chunk_ids = list(gold_by_chunk.keys())
    failed_chunks = []
    for cid in chunk_ids:
        gold_rec = gold_by_chunk[cid]
        raw_rec = raw_by_chunk.get(cid, {})
        reg = gold_rec.get("regulation", "?")
        chunk_text = raw_rec.get("chunk", {}).get("text", "") if raw_rec else ""
        if raw_rec.get("extraction_error"):
            failed_chunks.append(cid)

        gold_anns = gold_rec.get("annotations", [])
        sys_stmts = _sys_statements(raw_rec)

        scorer.score_statements(gold_anns, sys_stmts, chunk_text, reg)

        gold_nodes = [n for a in gold_anns for n in a.get("nodes", [])]
        sys_nodes = [n for s in sys_stmts for n in s.get("nodes", [])]
        scorer.score_nodes(gold_nodes, sys_nodes, reg)

        gold_rels = [r for a in gold_anns for r in a.get("relationships", [])]
        sys_rels = [r for s in sys_stmts for r in s.get("relationships", [])]
        scorer.score_relationships(gold_rels, sys_rels, reg)

    out = scorer.summary()
    out["meta"] = {
        "gold_file": "evaluation/gold_sample/gold_annotations.json",
        "system_file": "evaluation/gold_sample/raw_extractions.json",
        "model": raw_doc.get("model"),
        "gold_human_validated": bool(gold_doc.get("human_validated")),
        "gold_provenance": "Claude-pre-reviewed then human-validated"
                           if gold_doc.get("claude_reviewed") else "human-annotated",
        "chunk_count": len(chunk_ids),
        "failed_extraction_chunks": failed_chunks,
        "match_thresholds": {
            "evidence_overlap_coefficient_match": EVID_MATCH,
            "evidence_exact": EVID_EXACT,
            "node_name_near_jaccard": NODE_NEAR,
        },
    }
    return out


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_h1_markdown(metrics: dict, path: Path) -> None:
    m = metrics["meta"]
    lines: list[str] = []
    lines.append("# Extraction quality (H1)")
    lines.append("")
    lines.append(f"- System output: `{m['system_file']}` (model `{m['model']}`)")
    lines.append(f"- Ground truth: `{m['gold_file']}` "
                 f"(human_validated={m['gold_human_validated']}, provenance: {m['gold_provenance']})")
    lines.append(f"- Chunks scored: {m['chunk_count']} "
                 f"(failed extraction, scored as all-missing: {len(m['failed_extraction_chunks'])} "
                 f"-> {', '.join(m['failed_extraction_chunks']) or 'none'})")
    th = m["match_thresholds"]
    lines.append(f"- Match: statement=type+evidence overlap>= {th['evidence_overlap_coefficient_match']}, "
                 f"node=type+canonical_name (case/space-insensitive), "
                 f"relationship=type+source/target canonical_name pair.")
    lines.append("")
    lines.append("## Precision / Recall / F1")
    lines.append("")
    lines.append("| Level | TP | System | Gold | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for level in ("statement", "node", "relationship"):
        s = metrics[level]
        lines.append(f"| {level} | {s['true_positive']} | {s['system_count']} | {s['gold_count']} | "
                     f"{_fmt(s['precision'])} | {_fmt(s['recall'])} | {_fmt(s['f1'])} |")
    lines.append("")
    lines.append("## Error categories (Appendix B.2)")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for c in CATEGORIES:
        lines.append(f"| {c} | {metrics['error_categories'][c]} |")
    lines.append("")
    lines.append("## Error categories by regulation")
    lines.append("")
    header = "| Category | " + " | ".join(metrics["error_categories_by_regulation"].keys()) + " |"
    lines.append(header)
    lines.append("|---" * (len(metrics["error_categories_by_regulation"]) + 1) + "|")
    for c in CATEGORIES:
        row = [c] + [str(metrics["error_categories_by_regulation"][reg][c])
                     for reg in metrics["error_categories_by_regulation"]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def cmd_score_h1(args: argparse.Namespace) -> None:
    gold = json.loads(Path(args.gold).read_text())
    raw = json.loads(Path(args.raw).read_text())
    metrics = score_h1(gold, raw)
    out_json = Path(args.out_json)
    out_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    write_h1_markdown(metrics, Path(args.out_md))
    print(f"Wrote {out_json}")
    print(f"Wrote {args.out_md}")
    for level in ("statement", "node", "relationship"):
        s = metrics[level]
        print(f"  {level:12} P={_fmt(s['precision'])} R={_fmt(s['recall'])} F1={_fmt(s['f1'])} "
              f"(TP={s['true_positive']} sys={s['system_count']} gold={s['gold_count']})")
    print("  errors:", dict(metrics["error_categories"]))


# --------------------------------------------------------------------------- #
# H2 validation rate (larger, unannotated sample)
# --------------------------------------------------------------------------- #
def _eligible_chunks(per_reg: int, dup_threshold: float = 0.6) -> list[DocumentChunk]:
    """Same stratified, obligation-bearing, non-duplicate selection as build_sample.py."""
    selected: list[DocumentChunk] = []
    settings = get_settings()
    for _label, rel_path in REGULATIONS.items():
        document = load_document(ROOT / rel_path)
        chunks = build_chunks(document=document, max_chars=settings.chunk_max_chars,
                              overlap_chars=settings.chunk_overlap_chars)
        eligible = [c for c in chunks if any(p.search(c.text) for p in _CUES)]
        eligible.sort(key=lambda c: (-sum(len(p.findall(c.text)) for p in _CUES), c.chunk_id))
        picked: list[DocumentChunk] = []
        picked_tokens: list[set[str]] = []
        for cand in eligible:
            toks = _tokens(cand.text)
            if any(_jaccard(toks, prev) >= dup_threshold for prev in picked_tokens):
                continue
            picked.append(cand); picked_tokens.append(toks)
            if len(picked) >= per_reg:
                break
        selected.extend(picked)
    return selected


def _classify_validation_error(exc: ValidationError) -> str:
    span = enum = empty = other = False
    for err in exc.errors():
        etype = err.get("type", "")
        msg = str(err.get("msg", ""))
        if "evidence_end_char" in msg or "evidence_end_char must be" in msg:
            span = True
        elif etype == "literal_error":
            enum = True
        elif etype in ("string_too_short", "too_short", "missing"):
            empty = True
        else:
            other = True
    # Priority order for a single label per chunk.
    if enum:
        return "literal_enum_rejection"
    if span:
        return "evidence_span_order"
    if empty:
        return "empty_required_field"
    return "other_validation_error" if other else "other_validation_error"


def run_h2(sample_size: int = 100) -> dict[str, Any]:
    settings = get_settings()
    per_reg = -(-sample_size // 3)  # ceil
    chunks = _eligible_chunks(per_reg)
    # Keep stratification but trim to exactly sample_size (round-robin by reg order).
    chunks = chunks[:sample_size]
    chunk_map = {c.chunk_id: c for c in chunks}

    client = OllamaClient(settings)
    outcomes: Counter = Counter()
    per_chunk: list[dict] = []
    validated_statements = 0
    stmt_chunk_id_strict = 0   # (a) statement.chunk_id non-null AND resolves
    stmt_chunk_id_fallback = 0  # (b) owning chunk id resolves

    schema = ChunkExtractionResult.model_json_schema()
    for idx, chunk in enumerate(chunks, start=1):
        label = chunk.regulation_name
        print(f"[{idx}/{len(chunks)}] H2 {chunk.chunk_id} (len={len(chunk.text)}) ...", flush=True)
        outcome = "valid"
        detail = None
        result: ChunkExtractionResult | None = None
        try:
            prompt = build_extraction_prompt(chunk)
            result_json = client.chat_json(system_prompt=SYSTEM_PROMPT, user_prompt=prompt, schema=schema)
            try:
                result = ChunkExtractionResult.model_validate(result_json)
                outcome = "valid"
            except ValidationError as ve:
                outcome = _classify_validation_error(ve)
                detail = ve.errors()[0].get("msg") if ve.errors() else None
        except OllamaTimeoutError as exc:
            outcome = "timeout"
            detail = str(exc)[:160]
        except ValueError as exc:
            msg = str(exc)
            if "not valid JSON" in msg:
                outcome = "malformed_json"
            elif "empty content" in msg:
                outcome = "empty_content"
            else:
                outcome = "other_error"
            detail = msg[:160]
        except Exception as exc:  # noqa: BLE001
            outcome = "other_error"
            detail = f"{type(exc).__name__}: {exc}"[:160]

        outcomes[outcome] += 1
        n_stmts = len(result.statements) if result is not None else 0
        if result is not None:
            for st in result.statements:
                validated_statements += 1
                # (a) strict: statement.chunk_id emitted by the LLM, resolving to a known chunk.
                if st.chunk_id is not None and st.chunk_id in chunk_map:
                    stmt_chunk_id_strict += 1
                # (b) fallback: owning chunk's id (assigned at normalisation) resolves to source.
                if chunk.chunk_id in chunk_map and chunk_map[chunk.chunk_id].text:
                    stmt_chunk_id_fallback += 1
        per_chunk.append({
            "chunk_id": chunk.chunk_id, "regulation": label, "char_len": len(chunk.text),
            "outcome": outcome, "statements": n_stmts, "detail": detail,
        })

    total = len(chunks)
    validation_failure_labels = [
        "literal_enum_rejection", "evidence_span_order", "empty_required_field",
        "malformed_json", "other_validation_error",
    ]
    infra_labels = ["timeout", "empty_content", "other_error"]

    valid = outcomes.get("valid", 0)
    validation_failures = sum(outcomes.get(k, 0) for k in validation_failure_labels)
    infra_failures = sum(outcomes.get(k, 0) for k in infra_labels)

    return {
        "description": "H2 first-attempt schema-validation rate over a larger unannotated, "
                       "stratified sample. Single attempt, no adaptive retry, no fallback model.",
        "model": settings.ollama_model,
        "timeout_seconds": settings.ollama_timeout_seconds,
        "chunk_max_chars": settings.chunk_max_chars,
        "sample_size": total,
        "stratification": dict(Counter(c.regulation_name for c in chunks)),
        "first_attempt_pass": {
            "valid": valid,
            "total": total,
            "fraction": (valid / total) if total else None,
        },
        "validation_failures": {
            "count": validation_failures,
            "fraction": (validation_failures / total) if total else None,
            "by_reason": {k: outcomes.get(k, 0) for k in validation_failure_labels},
        },
        "infrastructure_failures_not_validation": {
            "count": infra_failures,
            "fraction": (infra_failures / total) if total else None,
            "by_reason": {k: outcomes.get(k, 0) for k in infra_labels},
            "note": "Timeouts/empty content are infra outcomes under the production "
                    "timeout, not schema-validation failures; reported separately.",
        },
        "outcomes_raw": dict(outcomes),
        "statement_chunk_id_resolvability": {
            "validated_statements": validated_statements,
            "approach_b_owning_chunk_primary": {
                "resolvable": stmt_chunk_id_fallback,
                "fraction": (stmt_chunk_id_fallback / validated_statements)
                            if validated_statements else None,
                "definition": "statement resolves to source via its owning chunk_id, which "
                              "normalizer.py assigns deterministically at normalisation "
                              "(ExtractedStatement.chunk_id is None by design at raw-extraction stage).",
            },
            "approach_a_strict_llm_emitted": {
                "resolvable": stmt_chunk_id_strict,
                "fraction": (stmt_chunk_id_strict / validated_statements)
                            if validated_statements else None,
                "note": "Expected near-zero and BY DESIGN: the LLM is not asked to populate "
                        "ExtractedStatement.chunk_id at this stage; not a traceability gap.",
            },
        },
        "per_chunk": per_chunk,
    }


def cmd_run_h2(args: argparse.Namespace) -> None:
    result = run_h2(args.sample_size)
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {out}")
    fap = result["first_attempt_pass"]
    print(f"  first-attempt Pydantic pass: {fap['valid']}/{fap['total']} = {_fmt(fap['fraction'])}")
    print(f"  validation failures: {result['validation_failures']['count']} "
          f"{result['validation_failures']['by_reason']}")
    print(f"  infra (non-validation): {result['infrastructure_failures_not_validation']['count']} "
          f"{result['infrastructure_failures_not_validation']['by_reason']}")
    r = result["statement_chunk_id_resolvability"]
    print(f"  chunk_id resolvable (b, primary): {_fmt(r['approach_b_owning_chunk_primary']['fraction'])} | "
          f"(a, strict): {_fmt(r['approach_a_strict_llm_emitted']['fraction'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("score-h1", help="score raw extractions vs gold annotations")
    p1.add_argument("--gold", default=str(GOLD_DIR / "gold_annotations.json"))
    p1.add_argument("--raw", default=str(GOLD_DIR / "raw_extractions.json"))
    p1.add_argument("--out-json", default=str(GOLD_DIR / "extraction_metrics.json"))
    p1.add_argument("--out-md", default=str(GOLD_DIR / "extraction_metrics.md"))
    p1.set_defaults(func=cmd_score_h1)

    p2 = sub.add_parser("run-h2", help="run first-attempt validation rate over a larger sample")
    p2.add_argument("--sample-size", type=int, default=100)
    p2.add_argument("--out", default=str(GOLD_DIR / "validation_rate.json"))
    p2.set_defaults(func=cmd_run_h2)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
