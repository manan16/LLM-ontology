from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, render_template, request
from neo4j import GraphDatabase

from app.config import get_settings
from web.services.rag_service import answer_question


app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template("index.html")


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
    try:
        result = answer_question(
            question=str(payload.get("question") or ""),
            limit=payload.get("limit", 30),
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

    return jsonify(
        {
            "answer": result["answer"],
            "evidence": result["evidence"],
            "graph": result.get("graph", {"nodes": [], "edges": []}),
            "context": result["context"],
            "debug": result["debug"],
            "determination": result.get("determination", _unavailable_determination()),
            "metrics": result["metrics"],
            "response_source": result.get("response_source", "live"),
            "error": None,
        }
    ), 200


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
