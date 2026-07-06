from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, render_template, request
from neo4j import GraphDatabase

from app.config import get_settings
from rag.retrieval_service import retrieve as retrieval_retrieve
from web.services.evaluation_report import load_evaluation_report
from web.services.rag_service import answer_question


app = Flask(__name__)

# Retrieval modes for the ablation/comparison view. "hybrid" is the production
# path; "semantic"/"graph" only populate the debug comparison section — they never
# replace the hybrid-gated answer/determination.
VALID_RETRIEVAL_MODES = ("semantic", "graph", "hybrid")


@app.get("/")
def index() -> str:
    return render_template("index.html")


@app.get("/evaluation")
def evaluation() -> str:
    return render_template("evaluation.html", report=load_evaluation_report())


@app.get("/health")
def health() -> tuple[Any, int]:
    settings = get_settings()
    neo4j = _neo4j_health(settings)
    status = "ok" if neo4j["status"] == "ok" else "degraded"
    return jsonify(
        {
            "status": status,
            "neo4j": neo4j,
            "ollama": {
                "base_url": settings.ollama_base_url,
                "status": "configured",
            },
        }
    ), 200


@app.post("/ask")
def ask() -> tuple[Any, int]:
    payload = request.get_json(silent=True) or {}

    mode = payload.get("mode") or "hybrid"
    if not isinstance(mode, str) or mode not in VALID_RETRIEVAL_MODES:
        return jsonify(
            _error_response(
                f"Invalid mode {mode!r}. Expected one of {list(VALID_RETRIEVAL_MODES)}."
            )
        ), 400

    question = str(payload.get("question") or "")
    limit = payload.get("limit", 30)
    try:
        # Production answer/determination ALWAYS run on the hybrid-gated path,
        # regardless of the requested mode.
        result = answer_question(
            question=question,
            limit=limit,
            show_context=bool(payload.get("show_context", True)),
            debug=bool(payload.get("debug", False)),
            model=payload.get("model") or None,
        )
    except ValueError as exc:
        return jsonify(_error_response(str(exc))), 400
    except Exception:
        return jsonify(
            _error_response(
                "The system could not retrieve enough graph evidence or generate an answer right now."
            )
        ), 500

    # Echo the mode and, for the ablation modes, attach the mode-specific
    # retrieval as a *comparison-only* view. This never feeds the top-level
    # answer/determination, which stay hybrid-gated above.
    response_debug = dict(result.get("debug") or {})
    response_debug["mode"] = mode
    if mode != "hybrid":
        response_debug["comparison"] = _mode_comparison(question, mode, limit)

    return jsonify(
        {
            "answer": result["answer"],
            "evidence": result["evidence"],
            "graph": result.get("graph", {"nodes": [], "edges": []}),
            "context": result["context"],
            "debug": response_debug,
            "determination": result.get("determination", _unavailable_determination()),
            "metrics": result["metrics"],
            "response_source": result.get("response_source", "live"),
            "mode": mode,
            "error": None,
        }
    ), 200


def _mode_comparison(question: str, mode: str, limit: Any) -> dict[str, Any]:
    """Run the requested ablation retrieval for the debug/comparison section.

    Failures degrade to an empty comparison — this view is for demoing the
    ablation and must never break the production response.
    """
    try:
        top_k = int(limit)
    except (TypeError, ValueError):
        top_k = 10
    try:
        outcome = retrieval_retrieve(question, mode=mode, top_k=max(1, top_k), debug=True)
    except Exception:
        return {"mode": mode, "error": "retrieval_unavailable", "rows": [], "row_count": 0}
    return {
        "mode": outcome.mode,
        "row_count": outcome.row_count,
        "top_k": outcome.top_k,
        "timings_ms": outcome.timings_ms,
        "graph_expansion_chunks": len(outcome.graph_expansion),
        "rows": [_trim_comparison_row(row) for row in outcome.rows[:20]],
        "detail": outcome.debug,
    }


def _trim_comparison_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "statement": row.get("statement_name") or row.get("seed_name"),
        "source_document": row.get("source_document"),
        "text": row.get("evidence_text") or row.get("source_chunk_text"),
        "score": row.get("score"),
        "similarity": row.get("similarity"),
        "query_route": row.get("query_route"),
        "semantic": bool(row.get("semantic")),
    }


def _error_response(message: str) -> dict[str, Any]:
    return {
        "answer": "",
        "evidence": [],
        "graph": {"nodes": [], "edges": []},
        "context": "",
        "debug": {},
        "determination": _unavailable_determination(),
        "metrics": {},
        "response_source": "error",
        "error": message,
    }


def _unavailable_determination() -> dict[str, Any]:
    return {
        "verdict": "unavailable",
        "obligations": [],
        "summary": "No determination is available for this response.",
    }


def _neo4j_health(settings: Any) -> dict[str, Any]:
    driver = None
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        with driver.session(database=settings.neo4j_database) as session:
            session.run("RETURN 1 AS ok").single()
            counts = {
                "regulations": _node_count(session, "Regulation"),
                "statements": _node_count(session, "Statement"),
                "source_chunks": _node_count(session, "SourceChunk"),
            }
        return {
            "status": "ok",
            "uri": settings.neo4j_uri,
            "database": settings.neo4j_database,
            "counts": counts,
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "uri": settings.neo4j_uri,
            "database": settings.neo4j_database,
            "error": exc.__class__.__name__,
        }
    finally:
        if driver is not None:
            driver.close()


def _node_count(session: Any, label: str) -> int:
    record = session.run(f"MATCH (n:{label}) RETURN count(n) AS count").single()
    return int(record["count"] if record else 0)


if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "5001"))
    app.run(host=host, port=port, debug=True)
