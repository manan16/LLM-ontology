from __future__ import annotations

from pathlib import Path

from app.config import get_settings
from app.logger import configure_logging, get_logger
from extraction.extractor import RegulationExtractor
from extraction.normalizer import normalize_chunk_results
from extraction.ollama_client import OllamaClient
from graph.neo4j_client import Neo4jClient
from graph.writer import GraphWriter
from ingestion.chunking import build_chunks
from ingestion.loaders import load_document


logger = get_logger(__name__)


def run_pipeline(document_path: str | Path) -> None:
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

    document = load_document(document_path)
    chunks = build_chunks(
        document=document,
        max_chars=settings.chunk_max_chars,
        overlap_chars=settings.chunk_overlap_chars,
    )
    logger.info("Loaded document=%s and created %s chunks", document.name, len(chunks))

    ollama_client = OllamaClient(settings)
    extractor = RegulationExtractor(ollama_client)

    chunk_results = []
    for chunk in chunks:
        chunk_results.extend(extractor.extract_chunk_adaptive(chunk))

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
    finally:
        neo4j_client.close()

    logger.info("Pipeline complete for document=%s", document.path)
