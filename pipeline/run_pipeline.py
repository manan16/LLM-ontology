from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

from app.config import get_settings
from app.logger import configure_logging, get_logger
from extraction.extractor import RegulationExtractor
from extraction.normalizer import normalize_chunk_results
from extraction.ollama_client import OllamaClient
from graph.embeddings import ensure_vector_index, store_chunk_embeddings
from graph.neo4j_client import Neo4jClient
from graph.writer import GraphWriter
from ingestion.chunking import DocumentChunk, build_chunks
from ingestion.loaders import load_document
from rag.embedding_service import get_embedding_service
from tqdm import tqdm


logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    document_path: str
    chunk_count: int
    failed_chunk_count: int
    statement_count: int
    entity_count: int
    relationship_count: int
    elapsed_seconds: float


def run_pipeline(document_paths: str | Path | Iterable[str | Path]) -> list[PipelineResult]:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.debug(
        "Pipeline settings model=%s timeout=%s chunk_max=%s overlap=%s neo4j_db=%s",
        settings.ollama_model,
        settings.ollama_timeout_seconds,
        settings.chunk_max_chars,
        settings.chunk_overlap_chars,
        settings.neo4j_database,
    )

    paths = _coerce_document_paths(document_paths)
    document_progress = tqdm(
        paths,
        desc="Building graph",
        unit="doc",
        disable=len(paths) <= 1,
    )
    results: list[PipelineResult] = []
    for document_path in document_progress:
        document = load_document(document_path)
        document_progress.set_postfix_str(document.name)
        results.append(_run_document_pipeline(document, settings))
    return results


def _run_document_pipeline(document, settings) -> PipelineResult:
    started_at = perf_counter()
    chunks = build_chunks(
        document=document,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    logger.info("Loaded document=%s and created %s chunks", document.name, len(chunks))

    ollama_client = OllamaClient(settings)
    extractor = RegulationExtractor(ollama_client)

    chunk_results = []
    failed_chunks = 0
    chunk_progress = tqdm(chunks, desc=f"Extracting {document.name}", unit="chunk")
    for chunk in chunk_progress:
        chunk_progress.set_postfix_str(chunk.section_title[:40])
        try:
            extracted = extractor.extract_chunk_adaptive(chunk)
        except Exception as exc:
            failed_chunks += 1
            tqdm.write(f"{document.name}: failed chunk {chunk.chunk_id}: {exc}")
            logger.exception("Skipping failed chunk=%s document=%s", chunk.chunk_id, document.name)
            continue
        if len(extracted) > 1:
            tqdm.write(
                f"{document.name}: chunk {chunk.chunk_id} was split after timeout retry; "
                f"processed {len(extracted)} child chunks."
            )
        chunk_results.extend(extracted)

    nodes, relationships, statements = normalize_chunk_results(chunk_results)
    logger.debug(
        "Post-normalization counts statements=%s nodes=%s relationships=%s",
        len(statements),
        len(nodes),
        len(relationships),
    )

    neo4j_client = Neo4jClient(settings)
    try:
        writer = GraphWriter(neo4j_client)
        writer.ensure_schema()
        writer.write_document_graph(document, chunks, nodes, relationships, statements)
        if getattr(settings, "semantic_retrieval_enabled", False):
            _embed_new_chunks(neo4j_client, chunks, settings)
    finally:
        neo4j_client.close()

    elapsed_seconds = perf_counter() - started_at
    tqdm.write(
        f"Completed: {document.name}\n"
        f"Chunks processed: {len(chunks) - failed_chunks}/{len(chunks)}\n"
        f"Failed chunks: {failed_chunks}\n"
        f"Statements extracted: {len(statements)}\n"
        f"Entities extracted: {len(nodes)}\n"
        f"Relationships extracted: {len(relationships)}\n"
        f"Elapsed time: {elapsed_seconds:.2f}s"
    )
    logger.info("Pipeline complete for document=%s", document.path)
    return PipelineResult(
        document_path=document.path,
        chunk_count=len(chunks),
        failed_chunk_count=failed_chunks,
        statement_count=len(statements),
        entity_count=len(nodes),
        relationship_count=len(relationships),
        elapsed_seconds=elapsed_seconds,
    )


def _embed_new_chunks(neo4j_client: Neo4jClient, chunks: list[DocumentChunk], settings) -> None:
    """Generate and store embeddings for freshly written chunks during ingestion.

    Failures here are logged but never abort ingestion: the graph is already
    written and embeddings can be produced later via ``backfill_embeddings.py``.
    """
    if not chunks:
        return
    try:
        ensure_vector_index(neo4j_client, settings)
        embedding_service = get_embedding_service(settings)
        vectors = embedding_service.embed_documents([chunk.text for chunk in chunks])
        payload = [
            {"chunk_id": chunk.chunk_id, "embedding": vector}
            for chunk, vector in zip(chunks, vectors)
        ]
        updated = store_chunk_embeddings(neo4j_client, payload)
        logger.info("Embedded %s/%s new chunks during ingestion", updated, len(chunks))
    except Exception:
        logger.exception(
            "Failed to embed new chunks during ingestion; run backfill_embeddings.py to recover"
        )


def _coerce_document_paths(document_paths: str | Path | Iterable[str | Path]) -> list[str | Path]:
    if isinstance(document_paths, (str, Path)):
        return [document_paths]
    return list(document_paths)
