# Semantic Retrieval

This document describes the dense/semantic retrieval layer that was added **on
top of** the existing knowledge graph. Entity extraction, relationship
generation, and graph topology are unchanged — semantic retrieval only adds an
`embedding` property to existing `SourceChunk` nodes plus a Neo4j vector index.

## Design at a glance

```
                    ┌─────────────────────────────────────────────┐
   ingestion  ───▶  │ Regulation ─HAS_SECTION▶ Section ─CONTAINS▶  │
   (unchanged)      │                            SourceChunk       │
                    │                              .text           │
                    │                              .embedding  ◀── added
                    └─────────────────────────────────────────────┘
                                     ▲
                                     │  Neo4j VECTOR INDEX (cosine, 768-dim)
                                     │  name: source_chunk_embedding
                                     │
   query ──▶ EmbeddingService.embed_query ──▶ db.index.vector.queryNodes ──▶ top-k chunks
```

The embeddings are produced by `BAAI/bge-base-en-v1.5` via `sentence-transformers`,
stored **in Neo4j** (no separate vector database), and searched with Neo4j's
native vector index.

## Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `EmbeddingService` | `rag/embedding_service.py` | Loads the model **once** (cached singleton), batched passage encoding, query encoding with the BGE instruction prefix, L2-normalized vectors. |
| Vector index + storage | `graph/embeddings.py` | `ensure_vector_index` (idempotent `CREATE VECTOR INDEX ... IF NOT EXISTS`), `store_chunk_embeddings` (`db.create.setNodeVectorProperty`), and helpers to find chunks missing embeddings. |
| Backfill script | `backfill_embeddings.py` | One-time / re-runnable migration: reads existing chunks, **skips those already embedded**, embeds in batches, stores results. |
| `SemanticRetriever` | `rag/semantic_retriever.py` | Embeds the query, runs the vector search, returns top-k chunks with similarity scores as pipeline-shaped rows. |
| Pipeline integration | `web/services/rag_service.py` | Merges semantic chunk rows into the existing graph rows (deduped) before context building. |
| Ingestion hook | `pipeline/run_pipeline.py` | Embeds **new** chunks automatically after the graph is written. |

## Configuration

Set via environment / `.env` (see `.env.example`):

| Variable | Default | Meaning |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | sentence-transformers model id. |
| `EMBEDDING_DEVICE` | *(empty = auto)* | `cpu`, `cuda`, or empty to auto-select cuda when available. |
| `EMBEDDING_BATCH_SIZE` | `32` | Batch size for encoding. |
| `VECTOR_INDEX_NAME` | `source_chunk_embedding` | Neo4j vector index name. |
| `VECTOR_DIMENSIONS` | `768` | Embedding dimension (bge-base is 768). |
| `VECTOR_SIMILARITY` | `cosine` | Similarity function for the index. |
| `SEMANTIC_TOP_K` | `10` | Chunks returned per query. |
| `SEMANTIC_RETRIEVAL_ENABLED` | `true` | Master toggle; when false the pipeline is graph-only. |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | — | Existing Neo4j connection. |

## One-time setup (existing graph)

```bash
pip install -r requirements.txt        # pulls in sentence-transformers
python backfill_embeddings.py          # creates the index + embeds all chunks
```

The backfill is idempotent: chunks that already carry an `embedding` are
skipped, so it is safe to re-run after adding documents or if it is interrupted.

## Retrieval flow (per query)

1. `web/services/rag_service.answer_question` runs the existing `GraphRetriever`
   to get symbolic graph rows (unchanged behaviour).
2. `_augment_with_semantic` calls `SemanticRetriever.retrieve(question)`:
   - `EmbeddingService.embed_query` prefixes the query with the BGE instruction
     and produces a normalized 768-dim vector.
   - `db.index.vector.queryNodes(source_chunk_embedding, top_k, embedding)`
     returns the most similar `SourceChunk` nodes with cosine scores.
   - Each hit is joined back to its `Section`/`Regulation` and shaped into a row
     with `source_chunk_text`, `source_document`, `section_title`, `similarity`,
     and `score` (`similarity * 100`, so it ranks comparably to graph rows).
3. Semantic rows are appended **after** graph rows and only when they contribute
   chunk text not already present (dedup by normalized evidence fingerprint), so
   graph precision is preserved while dense recall fills gaps.
4. The merged rows flow through the unchanged `build_context` / `_extract_evidence`
   path into answer + determination generation.

## New documents

`pipeline/run_pipeline.py` embeds new chunks automatically during ingestion
(after the graph write, gated on `SEMANTIC_RETRIEVAL_ENABLED`). Embedding
failures are logged but never abort ingestion — the graph is already persisted
and `backfill_embeddings.py` can fill in any missing embeddings later.

## Failure modes / graceful degradation

- If `sentence-transformers` is not installed, Neo4j is unreachable, or the
  vector index is missing, `SemanticRetriever` construction/queries fail softly
  and the pipeline falls back to **graph-only** retrieval.
- The embedding model is loaded once per process (`get_embedding_service`) and
  reused across queries and ingestion.
