# KEP Compliance Graph-RAG

Local-first Python project for building a compliance-aware Neo4j knowledge graph from regulation documents and asking grounded questions over that graph.

The project has two separate flows:

- KG extraction: load a `.pdf`, `.txt`, or `.md` regulation document, extract compliance concepts with Ollama, normalize them, and upsert them into Neo4j.
- Graph-RAG answering: plan a user question, retrieve relevant graph evidence, build a grounded context, and generate a compliance-style answer with Ollama.

## Features

- Section-aware document chunking with provenance
- Structured extraction of entities, statements, requirements, obligations, permissions, prohibitions, exceptions, citations, risks, and controls
- Neo4j graph writer with schema-safe upserts
- Agentic Graph-RAG retrieval flow:
  - query categorization
  - query transformation and expansion
  - follow-up rewrite support
  - simple decomposition for multi-part questions
  - graph retrieval routes for entities, statements, actors, evidence, permissions, and exceptions
  - scoring, deduplication, and evidence-focused context building
- Compliance-focused answer prompt that avoids inventing legal requirements

## Requirements

- Python 3.11+
- Neo4j reachable over Bolt
- Ollama running locally
- An Ollama model available for extraction and answering

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Copy the environment template and update Neo4j/Ollama settings:

```bash
cp .env.example .env
```

Important settings:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
OLLAMA_FALLBACK_MODELS=qwen2.5:7b-instruct,qwen3:14b
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=please-change-me
NEO4J_DATABASE=neo4j
```

The RAG answer generator defaults to `qwen3:8b`, and can be overridden with `--model`.

## Build The Knowledge Graph

Run extraction on a regulation document:

```bash
python main.py sample_regulation.md
```

For your own documents:

```bash
python main.py path/to/regulation.pdf
```

The extraction pipeline writes graph nodes and relationships into Neo4j. It does not need to be rerun when you are only changing RAG retrieval logic.

## Ask Questions

Ask a grounded compliance question over the existing Neo4j graph:

```bash
python ask.py "When can a covered entity disclose PHI without authorization?"
```

Show the retrieved graph context:

```bash
python ask.py "When can a covered entity disclose PHI without authorization?" --show-context
```

Debug the query plan and top retrieval rows:

```bash
python ask.py "When can a covered entity disclose PHI without authorization?" --debug-retrieval --show-context --limit 50
```

Override the answer model:

```bash
python ask.py "What obligations apply to providers?" --model qwen3:14b
```

## Web Interface

Run the lightweight Flask UI for the same Graph-RAG pipeline:

```bash
export FLASK_APP=web.app
flask run --port 5001
```

You can also run it as a module:

```bash
WEB_PORT=5001 python -m web.app
```

Then open:

```text
http://127.0.0.1:5001
```

The web app wraps the existing RAG components used by `ask.py`; it does not modify KG extraction, rebuild the graph, or change the Neo4j schema.

## RAG Architecture

The current retrieval flow is:

```text
User Query
-> Query Categorization Agent
-> Query Transformation Agent
-> Query Rewrite / Decomposition
-> Graph Query Planner
-> Neo4j Graph Retrieval Tool
-> Graph Expansion + Evidence Collection
-> Context Builder
-> Compliance Reporter / Answer Generator
-> Final Grounded Response
```

Key RAG files:

- `ask.py`: CLI entry point for asking questions
- `rag/query_planner.py`: categorization, normalization, rewrite, decomposition, phrase detection, expansion
- `rag/retriever.py`: Neo4j graph retrieval routes, scoring, deduplication
- `rag/context_builder.py`: evidence grouping and context formatting
- `rag/prompts.py`: answer-generation prompt
- `rag/answer_generator.py`: Ollama-backed answer generation
- `web/`: Flask web UI and reusable RAG service wrapper

## Retrieval Notes

The retriever includes special handling for permission/exception questions such as:

```text
When can a covered entity disclose PHI without authorization?
```

For negated authorization queries, retrieval prioritizes Permission and Exception statements, down-ranks plain authorization requirements, and keeps treatment/payment/health care operations evidence grouped only when the evidence actually belongs to that semantic category.

## Tests

Run the test suite:

```bash
python -m pytest
```

If `pytest` is not installed in your current interpreter:

```bash
pip install pytest
```

Basic syntax check:

```bash
python -m compileall ask.py app extraction graph ingestion pipeline rag tests main.py
```

## Project Layout

```text
app/          Configuration and logging
extraction/   Ollama extraction schemas, prompts, normalizer, extractor
graph/        Neo4j client, schema helpers, writer
ingestion/    Document loading and chunking
pipeline/     End-to-end KG construction pipeline
rag/          Graph-RAG planner, retriever, context builder, prompt, answer generator
web/          Flask UI and service wrapper for Graph-RAG questions
tests/        Unit tests
```

## Data And Local Artifacts

Large local inputs and generated artifacts are intentionally ignored by git:

- `data/`
- `*.png`
- `*.csv`
- `Neo4j-*.txt`
- `.env`
- virtual environments and Python caches

Keep credentials in `.env`, not in source files.
