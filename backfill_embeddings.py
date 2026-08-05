from __future__ import annotations

"""One-time backfill that adds embeddings to existing SourceChunk nodes.

This does NOT touch entities, relationships, statements, or graph topology. It
only reads existing chunk text, generates embeddings in batches, and writes an
``embedding`` vector property back onto each chunk. Chunks that already have an
embedding are skipped, so the script is idempotent and safe to re-run.

Usage:
    python backfill_embeddings.py            # embed all chunks missing embeddings
    python backfill_embeddings.py --batch 64 # override embedding batch size
"""

import argparse
from time import perf_counter

from app.config import get_settings
from app.logger import configure_logging, get_logger
from graph.embeddings import (
    count_chunks_missing_embeddings,
    ensure_vector_index,
    iter_chunks_missing_embeddings,
    store_chunk_embeddings,
)
from graph.neo4j_client import Neo4jClient
from rag.embedding_service import get_embedding_service
from tqdm import tqdm


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill embeddings on existing Chunk nodes.")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Embedding/query batch size (defaults to EMBEDDING_BATCH_SIZE).",
    )
    return parser


def run_backfill(batch_size: int | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    effective_batch = batch_size or settings.embedding_batch_size

    neo4j_client = Neo4jClient(settings)
    total_updated = 0
    try:
        ensure_vector_index(neo4j_client, settings)

        missing = count_chunks_missing_embeddings(neo4j_client)
        if missing == 0:
            logger.info("No chunks require embeddings; backfill is up to date.")
            return 0
        logger.info("Backfilling embeddings for %s chunk(s) batch_size=%s", missing, effective_batch)

        embedding_service = get_embedding_service(settings)
        started_at = perf_counter()
        progress = tqdm(total=missing, desc="Embedding chunks", unit="chunk")
        for batch in iter_chunks_missing_embeddings(neo4j_client, batch_size=effective_batch):
            texts = [row["text"] for row in batch]
            vectors = embedding_service.embed_documents(texts, batch_size=effective_batch)
            payload = [
                {"chunk_id": row["chunk_id"], "embedding": vector}
                for row, vector in zip(batch, vectors)
            ]
            updated = store_chunk_embeddings(neo4j_client, payload)
            total_updated += updated
            progress.update(len(batch))
        progress.close()

        elapsed = perf_counter() - started_at
        logger.info("Backfill complete: embedded %s chunk(s) in %.2fs", total_updated, elapsed)
    finally:
        neo4j_client.close()
    return total_updated


def main() -> None:
    args = build_parser().parse_args()
    run_backfill(batch_size=args.batch)


if __name__ == "__main__":
    main()
