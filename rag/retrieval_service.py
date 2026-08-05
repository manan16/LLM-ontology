from __future__ import annotations

"""Unified retrieval entry point for evaluation / comparison.

Exposes a single ``retrieve(query, mode, ...)`` function that runs one of three
retrieval strategies and returns one uniform :class:`RetrievalResult` shape, so
downstream consumers (e.g. ``context_builder.build_context``) never need
mode-specific branches.

Modes:
  * ``"semantic"`` — dense vector search only (SemanticRetriever). No graph
    traversal, no graph expansion, no HybridRetriever.
  * ``"graph"``    — graph traversal only (GraphRetriever), seeded from the
    entities/keywords the query planner extracts from the query. No vector
    search, so the embedding model is never touched.
  * ``"hybrid"``   — delegates to the existing HybridRetriever unchanged.

This module is intentionally separate from the production ``answer_question``
flow: it does NOT implement the confirmed/unconfirmed/no-evidence state logic,
which is specific to the hybrid/production path.
"""

from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Any, Callable, Literal

from app.config import Settings, get_settings
from app.logger import get_logger
from rag.hybrid_retriever import HybridRetriever
from rag.retriever import GraphRetriever
from rag.semantic_retriever import SemanticRetriever


logger = get_logger(__name__)

RetrievalMode = Literal["semantic", "graph", "hybrid"]
_VALID_MODES: tuple[str, ...] = ("semantic", "graph", "hybrid")

# Callables that lazily produce a retriever. The semantic provider may return
# None when semantic retrieval is unavailable/disabled.
GraphProvider = Callable[[], Any]
SemanticProvider = Callable[[], Any]

_UNSET = object()


@dataclass
class RetrievalResult:
    """Uniform result returned by every retrieval mode.

    ``rows`` is the shared row-dict shape all three retrievers already emit and
    is what ``context_builder.build_context`` consumes directly. ``graph_expansion``
    is populated only by the hybrid mode ({} otherwise). ``debug`` is populated
    only when ``debug=True`` was requested.
    """

    mode: str
    query: str
    rows: list[dict[str, Any]]
    top_k: int
    graph_expansion: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, int] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return len(self.rows)


class RetrievalService:
    """Runs a single retrieval strategy and returns a uniform result.

    Retrievers are created lazily and only for the mode being used, so:
      * semantic mode never constructs the graph retriever (never queries the
        graph in Neo4j), and
      * graph mode never constructs the semantic retriever (never loads/calls
        the embedding model).

    Providers can be injected for testing; in production they default to the
    real GraphRetriever / SemanticRetriever, built on first use.
    """

    def __init__(
        self,
        graph_provider: GraphProvider | None = None,
        semantic_provider: SemanticProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._graph_provider = graph_provider
        self._semantic_provider = semantic_provider
        self._graph_cache: Any = _UNSET
        self._semantic_cache: Any = _UNSET

    # -- lazy retriever accessors -------------------------------------------
    def _graph_retriever(self) -> Any:
        if self._graph_cache is _UNSET:
            if self._graph_provider is not None:
                self._graph_cache = self._graph_provider()
            else:
                self._graph_cache = GraphRetriever(settings=self.settings)
        return self._graph_cache

    def _semantic_retriever(self) -> Any:
        if self._semantic_cache is _UNSET:
            if self._semantic_provider is not None:
                self._semantic_cache = self._semantic_provider()
            else:
                self._semantic_cache = _build_default_semantic(self.settings)
        return self._semantic_cache

    # -- entry point ---------------------------------------------------------
    def retrieve(
        self,
        query: str,
        mode: RetrievalMode,
        top_k: int = 10,
        debug: bool = False,
    ) -> RetrievalResult:
        if mode not in _VALID_MODES:
            raise ValueError(f"Unknown retrieval mode: {mode!r}. Expected one of {_VALID_MODES}.")
        cleaned = (query or "").strip()
        safe_top_k = max(1, int(top_k))
        logger.debug("Retrieval mode=%s top_k=%s query=%r", mode, safe_top_k, cleaned)

        if mode == "semantic":
            return self._retrieve_semantic(cleaned, safe_top_k, debug)
        if mode == "graph":
            return self._retrieve_graph(cleaned, safe_top_k, debug)
        return self._retrieve_hybrid(cleaned, safe_top_k, debug)

    # -- per-mode implementations -------------------------------------------
    def _retrieve_semantic(self, query: str, top_k: int, debug: bool) -> RetrievalResult:
        started = perf_counter()
        semantic = self._semantic_retriever()
        rows = list(semantic.retrieve(query, top_k=top_k)) if semantic is not None else []
        elapsed = _elapsed_ms(started)
        return RetrievalResult(
            mode="semantic",
            query=query,
            rows=rows,
            top_k=top_k,
            graph_expansion={},
            timings_ms={"semantic_ms": elapsed},
            debug=self._debug(debug, "semantic", rows, extra={"semantic_available": semantic is not None}),
        )

    def _retrieve_graph(self, query: str, top_k: int, debug: bool) -> RetrievalResult:
        started = perf_counter()
        graph = self._graph_retriever()
        rows = list(graph.retrieve(query, limit=top_k))
        elapsed = _elapsed_ms(started)
        return RetrievalResult(
            mode="graph",
            query=query,
            rows=rows,
            top_k=top_k,
            graph_expansion={},
            timings_ms={"graph_ms": elapsed},
            debug=self._debug(debug, "graph", rows, retriever=graph),
        )

    def _retrieve_hybrid(self, query: str, top_k: int, debug: bool) -> RetrievalResult:
        graph = self._graph_retriever()
        hybrid = HybridRetriever(
            graph_retriever=graph,
            semantic_provider=self._semantic_retriever,
            settings=self.settings,
        )
        outcome = hybrid.retrieve(query, limit=top_k)
        # Apply the weighted hybrid ranking (0.4 semantic / 0.6 graph, sourced from
        # Settings) so hybrid mode matches the production ordering in
        # rag_service.answer_question, instead of returning the raw graph-first /
        # semantic-appended merge order. rank_rows() attaches semantic_score,
        # graph_relevance_score and hybrid_score and sorts by hybrid_score.
        ranked_rows = hybrid.rank_rows(outcome.rows)
        return RetrievalResult(
            mode="hybrid",
            query=query,
            rows=ranked_rows,
            top_k=top_k,
            graph_expansion=outcome.graph_expansion,
            timings_ms={
                "graph_ms": outcome.graph_ms,
                "semantic_ms": outcome.semantic_ms,
                "semantic_added": outcome.semantic_added,
            },
            debug=self._debug(
                debug,
                "hybrid",
                outcome.rows,
                retriever=graph,
                extra={"semantic_added": outcome.semantic_added, "graph_expansion_chunks": len(outcome.graph_expansion)},
            ),
        )

    @staticmethod
    def _debug(
        enabled: bool,
        mode: str,
        rows: list[dict[str, Any]],
        retriever: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not enabled:
            return {}
        payload: dict[str, Any] = {"mode": mode, "row_count": len(rows)}
        plan = getattr(retriever, "last_query_plan", None)
        if plan is not None:
            payload["query_plan"] = {
                "category": getattr(plan, "category", None),
                "intent": getattr(plan, "intent", None),
                "terms": getattr(plan, "terms", None),
                "phrases": getattr(plan, "phrases", None),
            }
        if extra:
            payload.update(extra)
        return payload


def _build_default_semantic(settings: Settings) -> SemanticRetriever | None:
    if not settings.semantic_retrieval_enabled:
        logger.info("Semantic retrieval disabled by settings; semantic mode will return no rows")
        return None
    try:
        return SemanticRetriever(settings=settings)
    except Exception:
        logger.exception("SemanticRetriever unavailable; semantic mode will return no rows")
        return None


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


# -- module-level entry point ------------------------------------------------
_default_service: RetrievalService | None = None
_service_lock = Lock()


def _get_default_service() -> RetrievalService:
    global _default_service
    if _default_service is None:
        with _service_lock:
            if _default_service is None:
                _default_service = RetrievalService()
    return _default_service


def retrieve(
    query: str,
    mode: RetrievalMode,
    top_k: int = 10,
    debug: bool = False,
) -> RetrievalResult:
    """Run a single retrieval strategy and return a uniform :class:`RetrievalResult`.

    See the module docstring for mode semantics. Uses a process-wide default
    :class:`RetrievalService` whose retrievers are built lazily on first use.
    """
    return _get_default_service().retrieve(query, mode, top_k=top_k, debug=debug)
