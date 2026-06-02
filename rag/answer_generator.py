from __future__ import annotations

from app.config import Settings, get_settings
from extraction.ollama_client import OllamaClient
from rag.prompts import SYSTEM_PROMPT, build_rag_prompt


DEFAULT_RAG_MODEL = "qwen3:8b"


class AnswerGenerator:
    def __init__(
        self,
        ollama_client: OllamaClient | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.ollama_client = ollama_client or OllamaClient(settings or get_settings())
        self.ollama_client.model = model or DEFAULT_RAG_MODEL

    def generate(self, question: str, context: str) -> str:
        if not context.strip():
            return (
                "The graph does not contain enough evidence to answer this question. "
                "No retrieved knowledge graph context was available."
            )
        return self.ollama_client.chat_text(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_rag_prompt(question, context),
        )
