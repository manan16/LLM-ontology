from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from app.config import Settings, get_settings
from app.logger import get_logger
from rag.retriever import expand_chunk_entities


logger = get_logger(__name__)

# The default hybrid weights live in exactly one place: app/config.py's Settings
# (hybrid_semantic_weight / hybrid_graph_weight). Both runtime and tests read them
# from there. No lexical/BM25 component — that retriever is not part of this stack;
# the two signals are dense-vector similarity and graph relevance.

SemanticProvider = Callable[[], Any]


@dataclass
class HybridRetrieval:
    """Result of a hybrid retrieval pass.

    ``rows`` is the merged graph + semantic row list (graph rows first, then the
    deduplicated semantic rows) — identical to the pre-refactor merge output.
    ``graph_expansion`` is the Task-1 entity expansion of the semantic chunks.
    """

    rows: list[dict[str, Any]]
    graph_expansion: dict[str, Any] = field(default_factory=dict)
    graph_ms: int = 0
    semantic_ms: int = 0
    semantic_added: int = 0


class HybridRetriever:
    """Combines the symbolic GraphRetriever with dense semantic retrieval.

    Responsibilities:

    * merge deduplicated semantic rows onto the graph rows,
    * expand the semantic chunks into their graph neighbourhood,
    * assign a weighted ``hybrid_score`` per row so the *confirmed* state can be
      ranked instead of merely concatenated.

    State determination (confirmed / unconfirmed / no-evidence) is intentionally
    NOT handled here — it stays in ``answer_question``. This class only affects
    ranking/scoring within the confirmed state.

    Ranking rationale: graph-confirmed evidence is weighted higher than semantic
    similarity by default (graph 0.6 vs semantic 0.4) because a graph hit is a
    structurally verified connection in the compliance knowledge graph, whereas
    dense-vector similarity is supplementary context — semantically close text
    that has not been verified against the graph.
    """

    def __init__(
        self,
        graph_retriever: Any,
        semantic_provider: SemanticProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.graph_retriever = graph_retriever
        self.settings = settings or get_settings()
        self._semantic_provider: SemanticProvider = semantic_provider or (lambda: None)
        # Weights come solely from Settings — the single source of truth.
        self.semantic_weight = float(self.settings.hybrid_semantic_weight)
        self.graph_weight = float(self.settings.hybrid_graph_weight)

    # -- retrieval + merge ---------------------------------------------------
    def retrieve(self, question: str, limit: int = 30) -> HybridRetrieval:
        graph_started = perf_counter()
        graph_rows = list(self.graph_retriever.retrieve(question, limit=limit))
        graph_ms = _elapsed_ms(graph_started)

        semantic_started = perf_counter()
        rows, added = self._merge_semantic(graph_rows, question)
        semantic_ms = _elapsed_ms(semantic_started)

        expansion = self._expand_semantic_chunks(rows)
        return HybridRetrieval(
            rows=rows,
            graph_expansion=expansion,
            graph_ms=graph_ms,
            semantic_ms=semantic_ms,
            semantic_added=added,
        )

    def _merge_semantic(self, rows: list[dict[str, Any]], question: str) -> tuple[list[dict[str, Any]], int]:
        """Append dense-retrieval chunk rows that add evidence not already present.

        Graph rows keep their ranking priority; semantic rows are added afterwards
        and only when they contribute chunk text the graph retrieval did not
        surface.
        """
        semantic_retriever = self._semantic_provider()
        if semantic_retriever is None:
            return rows, 0
        try:
            semantic_rows = semantic_retriever.retrieve(question)
        except Exception:
            logger.exception("Semantic retrieval failed; returning graph rows only")
            return rows, 0
        if not semantic_rows:
            return rows, 0

        seen = {fingerprint for row in rows if (fingerprint := _evidence_fingerprint(row))}
        added = 0
        for row in semantic_rows:
            fingerprint = _evidence_fingerprint(row)
            if not fingerprint or fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append(row)
            added += 1
        logger.info("Semantic retrieval added %s new chunk row(s) to %s graph row(s)", added, len(rows) - added)
        return rows, added

    def _expand_semantic_chunks(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Enrich semantic chunks with their graph neighbourhood.

        Never raises: on any failure it returns an empty dict and the caller
        degrades gracefully.
        """
        chunk_ids = [
            _clean(row.get("chunk_id"))
            for row in rows
            if row.get("semantic") and _clean(row.get("chunk_id"))
        ]
        if not chunk_ids:
            return {}
        neo4j_client = getattr(self.graph_retriever, "neo4j_client", None)
        if neo4j_client is None:
            return {}
        try:
            return expand_chunk_entities(chunk_ids, neo4j_client=neo4j_client)
        except Exception:
            logger.exception("Chunk entity expansion failed for %s chunk(s)", len(chunk_ids))
            return {}

    # -- weighted ranking (confirmed state only) -----------------------------
    def rank_rows(
        self,
        rows: list[dict[str, Any]],
        weights: tuple[float, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Attach weighted scores to each row and return them sorted by ``hybrid_score``.

        Each row gets ``semantic_score`` (dense similarity, 0..1), a normalised
        ``graph_relevance_score`` (graph score / max graph score, 0..1) and the
        combined ``hybrid_score``. The sort is stable, so graph rows keep their
        relative GraphRetriever order (their normalised score is monotonic) while
        semantic rows are interleaved by strength instead of dumped at the end.
        """
        semantic_weight, graph_weight = self._resolve_weights(weights)
        max_graph = max(
            (_graph_raw_score(row) for row in rows if not row.get("semantic")),
            default=0.0,
        )

        indexed: list[tuple[int, dict[str, Any]]] = []
        for index, row in enumerate(rows):
            semantic_score = _semantic_component(row)
            graph_relevance = _graph_component(row, max_graph)
            hybrid_score = semantic_weight * semantic_score + graph_weight * graph_relevance
            row["semantic_score"] = round(semantic_score, 6)
            row["graph_relevance_score"] = round(graph_relevance, 6)
            row["hybrid_score"] = round(hybrid_score, 6)
            indexed.append((index, row))

        indexed.sort(key=lambda pair: (-pair[1]["hybrid_score"], pair[0]))
        return [row for _, row in indexed]

    def _resolve_weights(self, weights: tuple[float, float] | None) -> tuple[float, float]:
        if weights is not None:
            return float(weights[0]), float(weights[1])
        return self.semantic_weight, self.graph_weight


# -- module helpers ----------------------------------------------------------
def _semantic_component(row: dict[str, Any]) -> float:
    """Dense similarity in 0..1 for semantic rows, 0 for graph rows."""
    if not row.get("semantic"):
        return 0.0
    similarity = row.get("similarity")
    if similarity is None:
        score = row.get("score")
        similarity = (float(score) / 100.0) if score is not None else 0.0
    return _clamp01(_as_float(similarity))


def _graph_raw_score(row: dict[str, Any]) -> float:
    return max(0.0, _as_float(row.get("score")))


def _graph_component(row: dict[str, Any], max_graph: float) -> float:
    """Normalised graph relevance in 0..1 for graph rows, 0 for semantic rows."""
    if row.get("semantic") or max_graph <= 0:
        return 0.0
    return _clamp01(_graph_raw_score(row) / max_graph)


def _evidence_fingerprint(row: dict[str, Any]) -> str:
    text = _clean(row.get("evidence_text")) or _clean(row.get("source_chunk_text"))
    return " ".join(text.lower().split())[:300]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)
