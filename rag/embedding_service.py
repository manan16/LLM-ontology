from __future__ import annotations

from collections.abc import Sequence
from threading import Lock

from app.config import Settings, get_settings
from app.logger import get_logger


logger = get_logger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


def _resolve_device(configured_device: str) -> str:
    """Return an explicit torch device string, auto-selecting cuda when available."""
    device = (configured_device or "").strip().lower()
    if device:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch always ships with sentence-transformers
        logger.debug("torch not available for device probe; defaulting to cpu")
    return "cpu"


class EmbeddingService:
    """Loads a sentence-transformers model once and reuses it for all embeddings.

    Queries and passages are embedded with L2-normalized vectors so that a
    Neo4j cosine vector index returns scores in the intuitive 0..1 range. BGE
    models additionally expect an instruction prefix on *query* text only, which
    is applied by :meth:`embed_query`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._device = _resolve_device(self._settings.embedding_device)
        self._model_name = self._settings.embedding_model
        self._batch_size = max(1, int(self._settings.embedding_batch_size))
        self._query_prefix = self._settings.embedding_query_prefix or ""
        self._model = self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise EmbeddingServiceError(
                "sentence-transformers is not installed. Install project dependencies "
                "with `pip install -r requirements.txt` to enable semantic retrieval."
            ) from exc

        logger.info(
            "Loading embedding model name=%s device=%s batch_size=%s",
            self._model_name,
            self._device,
            self._batch_size,
        )
        try:
            model = SentenceTransformer(self._model_name, device=self._device)
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Failed to load embedding model '{self._model_name}' on device '{self._device}'."
            ) from exc
        logger.info("Embedding model loaded dimension=%s", model.get_sentence_embedding_dimension())
        return model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
        """Embed passage/chunk text in batches. Returns one vector per input text."""
        if not texts:
            return []
        effective_batch = max(1, int(batch_size or self._batch_size))
        try:
            vectors = self._model.encode(
                list(texts),
                batch_size=effective_batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingServiceError(f"Failed to embed {len(texts)} documents.") from exc
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query, applying the model's query instruction prefix."""
        cleaned = (text or "").strip()
        if not cleaned:
            raise EmbeddingServiceError("Cannot embed an empty query.")
        prefixed = f"{self._query_prefix}{cleaned}"
        try:
            vector = self._model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingServiceError("Failed to embed query text.") from exc
        return vector.tolist()


_service: EmbeddingService | None = None
_service_lock = Lock()


def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Return a process-wide shared EmbeddingService, loading the model on first use."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = EmbeddingService(settings)
    return _service
