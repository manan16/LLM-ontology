from __future__ import annotations

import json
from typing import Any

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.logger import get_logger


logger = get_logger(__name__)


class OllamaTimeoutError(ValueError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.fallback_models = [item.strip() for item in settings.ollama_fallback_models.split(",") if item.strip()]
        self.temperature = settings.ollama_temperature
        self.timeout_seconds = settings.ollama_timeout_seconds
        self.keep_alive = settings.ollama_keep_alive

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.ConnectionError, requests.HTTPError)),
    )
    def _chat_json_once(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        effective_system_prompt = system_prompt
        if model_name.startswith("qwen3") and "/no_think" not in system_prompt:
            effective_system_prompt = f"/no_think\n{system_prompt}"

        payload = {
            "model": model_name,
            "stream": False,
            "format": schema,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        logger.debug(
            "Sending Ollama request model=%s timeout=%ss keep_alive=%s prompt_chars=%s schema_keys=%s",
            model_name,
            self.timeout_seconds,
            self.keep_alive,
            len(user_prompt),
            len(schema.keys()),
        )
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
        except requests.ReadTimeout as exc:
            raise OllamaTimeoutError(
                f"Ollama timed out after {self.timeout_seconds}s for model {model_name}. "
                "Increase OLLAMA_TIMEOUT_SECONDS or reduce CHUNK_MAX_CHARS for local execution."
            ) from exc
        response.raise_for_status()
        body = response.json()
        content = body.get("message", {}).get("content", "")
        logger.debug(
            "Received Ollama response model=%s content_chars=%s eval_count=%s eval_duration=%s",
            model_name,
            len(content),
            body.get("eval_count"),
            body.get("eval_duration"),
        )
        if not content:
            raise ValueError("Ollama returned empty content")
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Ollama returned non-JSON content: %s", content[:500])
            raise ValueError("Ollama content was not valid JSON") from exc

    def chat_json(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        model_chain = [self.model] + [model for model in self.fallback_models if model != self.model]
        last_error: Exception | None = None

        for model_name in model_chain:
            try:
                return self._chat_json_once(
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                )
            except OllamaTimeoutError as exc:
                last_error = exc
                logger.warning("Model timeout for model=%s, trying next fallback if available", model_name)
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Ollama models were available for extraction")

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((requests.ConnectionError, requests.HTTPError)),
    )
    def _chat_text_once(self, model_name: str, system_prompt: str, user_prompt: str) -> str:
        effective_system_prompt = system_prompt
        if model_name.startswith("qwen3") and "/no_think" not in system_prompt:
            effective_system_prompt = f"/no_think\n{system_prompt}"

        payload = {
            "model": model_name,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
            },
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        logger.debug(
            "Sending Ollama text request model=%s timeout=%ss keep_alive=%s prompt_chars=%s",
            model_name,
            self.timeout_seconds,
            self.keep_alive,
            len(user_prompt),
        )
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=(10, self.timeout_seconds),
            )
        except requests.ReadTimeout as exc:
            raise OllamaTimeoutError(
                f"Ollama timed out after {self.timeout_seconds}s for model {model_name}. "
                "Increase OLLAMA_TIMEOUT_SECONDS or reduce retrieved context size for local execution."
            ) from exc
        response.raise_for_status()
        body = response.json()
        content = body.get("message", {}).get("content", "")
        logger.debug(
            "Received Ollama text response model=%s content_chars=%s eval_count=%s eval_duration=%s",
            model_name,
            len(content),
            body.get("eval_count"),
            body.get("eval_duration"),
        )
        if not content:
            raise ValueError("Ollama returned empty content")
        return content

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        model_chain = [self.model] + [model for model in self.fallback_models if model != self.model]
        last_error: Exception | None = None

        for model_name in model_chain:
            try:
                return self._chat_text_once(model_name, system_prompt, user_prompt)
            except OllamaTimeoutError as exc:
                last_error = exc
                logger.warning("Model timeout for model=%s, trying next fallback if available", model_name)
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError("No Ollama models were available for generation")
