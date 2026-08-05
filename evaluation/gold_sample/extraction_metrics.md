# Extraction quality (H1)

- System output: `evaluation/gold_sample/raw_extractions.json` (model `qwen2.5:7b-instruct`)
- Ground truth: `evaluation/gold_sample/gold_annotations.json` (human_validated=True, provenance: Claude-pre-reviewed then human-validated)
- Chunks scored: 15 (failed extraction, scored as all-missing: 1 -> gdpr:sec-0078:0:d46a526c26)
- Match: statement=type+evidence overlap>= 0.5, node=type+canonical_name (case/space-insensitive), relationship=type+source/target canonical_name pair.

## Precision / Recall / F1

| Level | TP | System | Gold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| statement | 45 | 67 | 73 | 0.6716 | 0.6164 | 0.6429 |
| node | 0 | 0 | 81 | n/a | 0.0000 | 0.0000 |
| relationship | 70 | 77 | 72 | 0.9091 | 0.9722 | 0.9396 |

## Error categories (Appendix B.2)

| Category | Count |
|---|---|
| unsupported_extraction | 6 |
| wrong_statement_type | 22 |
| missing_extraction | 89 |
| incorrect_entity_type | 0 |
| incorrect_relationship | 1 |
| evidence_span_error | 0 |
| dedup_normalisation_error | 0 |
| retrieval_grounding_error | 0 |

## Error categories by regulation

| Category | EU_AI_ACT | GDPR | HIPAA |
|---|---|---|---|
| unsupported_extraction | 4 | 2 | 0 |
| wrong_statement_type | 9 | 6 | 7 |
| missing_extraction | 26 | 29 | 34 |
| incorrect_entity_type | 0 | 0 | 0 |
| incorrect_relationship | 0 | 1 | 0 |
| evidence_span_error | 0 | 0 | 0 |
| dedup_normalisation_error | 0 | 0 | 0 |
| retrieval_grounding_error | 0 | 0 | 0 |
