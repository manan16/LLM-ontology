from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, render_template, request

from web.services.rag_service import answer_question


app = Flask(__name__)


@app.get("/")
def index() -> str:
    return render_template("index.html")


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
            "metrics": result["metrics"],
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
        "metrics": {},
        "error": message,
    }


if __name__ == "__main__":
    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "5001"))
    app.run(host=host, port=port, debug=True)
