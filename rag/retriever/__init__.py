"""GraphRetriever package.

Split for reviewability into cohesive modules -- routing/query construction
(routes), scoring/ranking (scoring), deduplication primitives (dedup) and the
orchestrating GraphRetriever class (core). This __init__ preserves the historical
import surface: every name previously importable as ``from rag.retriever import X``
(used across the codebase and tests) is re-exported here unchanged.
"""

from __future__ import annotations

from rag.retriever.core import (
    _COMMON_RETURN,
    GraphRetriever,
    _dedupe_rows,
    _is_evidence_expansion_seed,
    _select_final_rows,
    expand_chunk_entities,
    extract_query_terms,
)

__all__ = [
    "GraphRetriever",
    "expand_chunk_entities",
    "extract_query_terms",
    "_COMMON_RETURN",
    "_dedupe_rows",
    "_is_evidence_expansion_seed",
    "_select_final_rows",
]
