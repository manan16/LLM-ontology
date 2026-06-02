# LLM-ontology

Local-first Python MVP for turning regulation documents into an ontology-style Neo4j knowledge graph using Ollama and `qwen3:14b`.

## What It Does

- Loads `.pdf`, `.txt`, and `.md` regulation documents
- Extracts section-aware chunks with provenance
- Calls a local Ollama model with deterministic structured JSON extraction
- Normalizes and deduplicates extracted concepts
- Upserts ontology-style nodes and relationships into Neo4j with provenance

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and update credentials:

```bash
cp .env.example .env
```

4. Run the pipeline on the sample document:

```bash
python main.py data/sample_regulation.md
```

If Ollama is slow on larger chunks, increase `OLLAMA_TIMEOUT_SECONDS` and/or lower `CHUNK_MAX_CHARS` in `.env`.

## Notes

- Ollama must be running locally with `qwen3:14b` available.
- Neo4j must be reachable using the values in `.env`.
- The first MVP assumes PDFs are machine-readable and does not perform OCR.
