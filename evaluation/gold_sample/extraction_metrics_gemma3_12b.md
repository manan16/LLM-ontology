# Extraction quality (H1)

- System output: `evaluation/gold_sample/raw_extractions.json` (model `gemma3:12b`)
- Ground truth: `evaluation/gold_sample/gold_annotations.json` (human_validated=True, provenance: Claude-pre-reviewed then human-validated)
- Chunks scored: 15 (failed extraction, scored as all-missing: 0 -> none)
- Match: statement=type+evidence overlap>= 0.5, node=type+canonical_name (case/space-insensitive), relationship=type+source/target canonical_name pair.

## Precision / Recall / F1

| Level | TP | System | Gold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| statement | 38 | 97 | 73 | 0.3918 | 0.5205 | 0.4471 |
| node | 0 | 0 | 81 | n/a | 0.0000 | 0.0000 |
| relationship | 0 | 20 | 72 | 0.0000 | 0.0000 | 0.0000 |

## Error categories (Appendix B.2)

| Category | Count |
|---|---|
| unsupported_extraction | 51 |
| wrong_statement_type | 24 |
| missing_extraction | 164 |
| incorrect_entity_type | 0 |
| incorrect_relationship | 1 |
| evidence_span_error | 8 |
| dedup_normalisation_error | 0 |
| retrieval_grounding_error | 3 |

## Error categories by regulation

| Category | EU_AI_ACT | GDPR | HIPAA |
|---|---|---|---|
| unsupported_extraction | 8 | 18 | 25 |
| wrong_statement_type | 8 | 7 | 9 |
| missing_extraction | 49 | 37 | 78 |
| incorrect_entity_type | 0 | 0 | 0 |
| incorrect_relationship | 1 | 0 | 0 |
| evidence_span_error | 4 | 2 | 2 |
| dedup_normalisation_error | 0 | 0 | 0 |
| retrieval_grounding_error | 3 | 0 | 0 |
