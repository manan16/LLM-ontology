# Model comparison: extraction quality (H1)

Both models scored by the identical harness (`evaluation/extraction_eval.py`, unchanged) against the same 15-chunk human-validated gold set (`gold_annotations.json`, 73 gold statements / 81 nodes / 72 relationships). Only the system-output file differs.

- qwen2.5:7b-instruct: `raw_extractions.json` (failed-extraction chunks: 1)
- gemma3:12b: `raw_extractions.gemma3_12b.json` (failed-extraction chunks: 0)

## Precision / Recall / F1

| Level | Metric | qwen2.5:7b-instruct | gemma3:12b |
|---|---|---|---|
| statement | precision | 0.6716 | 0.3918 |
| statement | recall | 0.6164 | 0.5205 |
| statement | f1 | 0.6429 | 0.4471 |
| statement | (TP/sys/gold) | 45/67/73 | 38/97/73 |
| node | precision | n/a | n/a |
| node | recall | 0.0000 | 0.0000 |
| node | f1 | 0.0000 | 0.0000 |
| node | (TP/sys/gold) | 0/0/81 | 0/0/81 |
| relationship | precision | 0.9091 | 0.0000 |
| relationship | recall | 0.9722 | 0.0000 |
| relationship | f1 | 0.9396 | 0.0000 |
| relationship | (TP/sys/gold) | 70/77/72 | 0/20/72 |

## Error categories (Appendix B.2), totals across all levels

| Category | qwen2.5:7b-instruct | gemma3:12b |
|---|---|---|
| unsupported_extraction | 6 | 51 |
| wrong_statement_type | 22 | 24 |
| missing_extraction | 89 | 164 |
| incorrect_entity_type | 0 | 0 |
| incorrect_relationship | 1 | 1 |
| evidence_span_error | 0 | 8 |
| dedup_normalisation_error | 0 | 0 |
| retrieval_grounding_error | 0 | 3 |

## Provenance caveat

The gold set was seeded from the qwen2.5:7b-instruct draft and then human-corrected, so qwen2.5 and gold share relationship/canonical-name origin. gemma3:12b did not seed the gold. This shows most starkly in the relationship row: qwen2.5 relationship F1=0.9396 vs gemma3:12b 0.0000 (0/20 matched). The node row (F1=0.0000 for both; neither model emits statement-level nodes) is provenance-neutral. Read cross-model relationship/statement F1 deltas with this bias in mind.
