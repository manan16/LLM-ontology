from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, get_settings
from extraction.ollama_client import OllamaClient
from rag.prompts import DETERMINATION_SYSTEM_PROMPT, build_determination_prompt


LOGGER = logging.getLogger(__name__)

DEFAULT_RAG_MODEL = "qwen3:8b"

# Controlled set of evidence-supported verdicts the model may choose from.
# "insufficient_evidence" is the honest fallback when the retrieved evidence
# does not support any of the substantive verdicts.
SUPPORTED_VERDICTS = (
    "obligations_apply",
    "permitted_with_conditions",
    "prohibited",
    "insufficient_evidence",
)

# Returned only by this module (never by the model) when the determination
# step itself fails — kept distinct from "insufficient_evidence", which is a
# judgement about the evidence rather than a system failure.
VERDICT_UNAVAILABLE = "unavailable"
VERDICT_INSUFFICIENT = "insufficient_evidence"

# Ollama structured-output JSON schema. The enum constrains the verdict to the
# supported set so parsing is reliable and the model cannot invent a verdict.
DETERMINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(SUPPORTED_VERDICTS)},
        "obligations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["verdict", "obligations", "summary"],
}


def insufficient_determination(summary: str) -> dict[str, Any]:
    """Deterministic determination for the no/insufficient-evidence path."""
    return {
        "verdict": VERDICT_INSUFFICIENT,
        "obligations": [],
        "summary": summary,
    }


def unavailable_determination(summary: str) -> dict[str, Any]:
    """Determination used when the synthesis step itself fails."""
    return {
        "verdict": VERDICT_UNAVAILABLE,
        "obligations": [],
        "summary": summary,
    }


class DeterminationGenerator:
    """Second structured LLM pass that derives a determination
    ({verdict, obligations[], summary}) strictly from the retrieved KG evidence
    and the grounded answer already produced for the same question.

    It never inspects the raw question for keywords to pick a verdict — the
    verdict is grounded in the supplied evidence/answer only.
    """

    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.ollama_client = ollama_client or OllamaClient(settings or get_settings())
        self.ollama_client.model = model or DEFAULT_RAG_MODEL

    def generate(self, question: str, context: str, answer: str) -> dict[str, Any]:
        if not context.strip():
            # No retrieved evidence -> cannot support a verdict. Report it.
            return insufficient_determination(
                "The graph did not return evidence sufficient to support a determination."
            )

        try:
            raw = self.ollama_client.chat_json(
                system_prompt=DETERMINATION_SYSTEM_PROMPT,
                user_prompt=build_determination_prompt(question, context, answer),
                schema=DETERMINATION_SCHEMA,
            )
        except Exception:
            # Do not swallow: log with stack trace, then degrade honestly.
            LOGGER.exception("Determination synthesis failed for question=%r", question)
            return unavailable_determination(
                "The determination could not be generated for this question."
            )

        return _normalize(raw)


def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce the model output into the contract shape defensively."""
    verdict = str(raw.get("verdict") or "").strip()
    if verdict not in SUPPORTED_VERDICTS:
        LOGGER.warning("Determination returned out-of-enum verdict=%r; treating as insufficient", verdict)
        verdict = VERDICT_INSUFFICIENT

    raw_obligations = raw.get("obligations") or []
    obligations = [str(item).strip() for item in raw_obligations if str(item).strip()]
    # An evidence-insufficient verdict must not carry obligations.
    if verdict == VERDICT_INSUFFICIENT:
        obligations = []

    summary = str(raw.get("summary") or "").strip()
    return {"verdict": verdict, "obligations": obligations, "summary": summary}
