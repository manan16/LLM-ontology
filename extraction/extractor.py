from __future__ import annotations

from app.logger import get_logger
from extraction.ollama_client import OllamaClient, OllamaTimeoutError
from extraction.prompts import SYSTEM_PROMPT, build_extraction_prompt
from extraction.schemas import ChunkExtractionResult
from ingestion.chunking import DocumentChunk, split_chunk_for_retry


logger = get_logger(__name__)


class RegulationExtractor:
    def __init__(self, ollama_client: OllamaClient) -> None:
        self.ollama_client = ollama_client

    def extract_chunk(self, chunk: DocumentChunk) -> ChunkExtractionResult:
        logger.debug(
            "Starting extraction for chunk=%s section=%s chars=%s",
            chunk.chunk_id,
            chunk.section_title,
            len(chunk.text),
        )
        prompt = build_extraction_prompt(chunk)
        schema = ChunkExtractionResult.model_json_schema()
        result_json = self.ollama_client.chat_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            schema=schema,
        )
        result = ChunkExtractionResult.model_validate(result_json)
        logger.info(
            "Extracted %s statements from chunk=%s section=%s",
            len(result.statements),
            chunk.chunk_id,
            chunk.section_title,
        )
        logger.debug(
            "Validated extraction for chunk=%s summary_chars=%s",
            chunk.chunk_id,
            len(result.chunk_summary),
        )
        return result

    def extract_chunk_adaptive(
        self,
        chunk: DocumentChunk,
        min_chunk_chars: int = 500,
        max_split_depth: int = 3,
        split_depth: int = 0,
    ) -> list[tuple[DocumentChunk, ChunkExtractionResult]]:
        try:
            return [(chunk, self.extract_chunk(chunk))]
        except OllamaTimeoutError as exc:

            if split_depth >= max_split_depth or len(chunk.text) <= min_chunk_chars:
                logger.error(
                    "Extraction timeout could not be recovered for chunk=%s chars=%s depth=%s",
                    chunk.chunk_id,
                    len(chunk.text),
                    split_depth,
                )
                raise

            retry_max_chars = max(min_chunk_chars, len(chunk.text) // 2)
            logger.warning(
                "Retrying timed-out chunk=%s by splitting chars=%s target=%s depth=%s",
                chunk.chunk_id,
                len(chunk.text),
                retry_max_chars,
                split_depth + 1,
            )
            results: list[tuple[DocumentChunk, ChunkExtractionResult]] = []
            for child in split_chunk_for_retry(chunk, retry_max_chars):
                results.extend(
                    self.extract_chunk_adaptive(
                        child,
                        min_chunk_chars=min_chunk_chars,
                        max_split_depth=max_split_depth,
                        split_depth=split_depth + 1,
                    )
                )
            return results
