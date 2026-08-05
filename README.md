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
- Three retrieval modes for ablation — `semantic` (vector similarity only), `graph` (ontology traversal only), `hybrid` (production; graph-gated with semantic recall) — selectable in the web console and the evaluation runner (`ask.py` always uses the hybrid path)
- Compliance-focused answer prompt that avoids inventing legal requirements
- Backend-derived determination (verdict + obligations) kept separate from the generated prose
- Batch evaluation over a 33-question golden set with a per-mode comparison report in the web UI

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
http://127.0.0.1:5001    Console — ask questions, trace the pipeline
http://127.0.0.1:5001/evaluation    Evaluation report
```

The web app wraps the existing RAG components used by `ask.py`; it does not modify KG extraction, rebuild the graph, or change the Neo4j schema.

### Console

The console traces one question through the pipeline: stage-by-stage timings, the grounded answer with clickable `[E#]` citations, the retrieved subgraph, the evidence list, and a right-hand inspector for the query plan, ranked rows, and retrieval stats.

- **Retrieval mode** selector (hybrid / semantic / graph) — the answer and determination are *always* the hybrid-gated production result; semantic and graph attach their retrieval as a comparison-only view in the inspector, so an ablation can never be mistaken for a production answer.
- **Session history** persists in `localStorage` across page navigation, including queries still in flight.
- **Reset** (in the session-history header) clears the session. It arms on first click and wipes on the second, so a stray click during a demo does nothing.

### Demo and presentation mode

Every empty session — a fresh load or one just reset — is seeded with a preloaded question and its complete response (answer, determination, graph, evidence, timings), drawn from the bundled sample payloads in `web/static/js/kep-demo.js`. It sits in session history one click away, so a recording can open a finished result immediately instead of waiting on a live query. Seeded and offline-fallback results carry a **sample data** badge; they are never presented as live retrieval.

To preload more than one, raise `SEED_COUNT` in `web/static/js/kep-app.js`.

The same bundled payloads back the offline fallback: when the Flask `/ask` endpoint is unreachable (or `index.html` is opened as a static file), the UI degrades to sample responses rather than erroring.

## Docker Demo Setup

For a reproducible local demo, you can run the Flask app and Neo4j with Docker Compose. Ollama is intentionally not containerized; keep it running on the host machine.

Start Ollama on the host:

```bash
ollama serve
```

Then start the app and Neo4j:

```bash
docker compose up --build
```

Open:

```text
http://localhost:5001
http://localhost:7474
```

Useful commands:

```bash
docker compose logs -f app
docker compose logs -f neo4j
docker compose down
curl -I http://localhost:5001/
```

Inside Docker, the app uses:

```bash
NEO4J_URI=bolt://neo4j:7687
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Set local credentials in `.env` before starting Compose if you do not want the demo defaults:

```bash
NEO4J_PASSWORD=please-change-me
```

The Docker Neo4j service uses the local `neo4j` user and `neo4j` database. This keeps the Compose demo independent from any Aura or external Neo4j username/database you may use in `.env` for non-Docker runs.

The Compose setup does not rebuild or seed the knowledge graph on container start. Build or import the graph separately, then use the web app for the RAG/KG demo.

The Flask debugger is **off by default**. It is gated behind `FLASK_DEBUG=1`, which enables Werkzeug's interactive debugger — that allows arbitrary code execution and must **never** be set in the Compose demo or anywhere the app is reachable over a network (e.g. an exposed Cloudflare tunnel URL). Treat `FLASK_DEBUG=1` as opt-in for local development only.

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
- `rag/retriever/`: Neo4j graph retrieval — `routes.py` (query construction and routing), `scoring.py` (scoring, ranking, classification), `dedup.py` (deduplication primitives), `core.py` (orchestration)
- `rag/semantic_retriever.py`: vector-similarity retrieval
- `rag/hybrid_retriever.py`: graph-gated combination of the two
- `rag/retrieval_service.py`: single entry point that dispatches on retrieval mode
- `rag/context_builder.py`: evidence grouping and context formatting
- `rag/determination.py`: backend-derived verdict and obligations
- `rag/prompts.py`: answer-generation prompt
- `rag/answer_generator.py`: Ollama-backed answer generation
- `web/`: Flask web UI and reusable RAG service wrapper

## Retrieval Notes

The retriever includes special handling for permission/exception questions such as:

```text
When can a covered entity disclose PHI without authorization?
```

For negated authorization queries, retrieval prioritizes Permission and Exception statements, down-ranks plain authorization requirements, and keeps treatment/payment/health care operations evidence grouped only when the evidence actually belongs to that semantic category.

## Evaluation

The evaluation suite scores the RAG pipeline against a hand-written golden set. It is measurement only — it never modifies retrieval, extraction, or the graph.

Question sets:

- `evaluation/golden_questions.json` — 33 questions across `gdpr_only`, `hipaa_only`, `eu_ai_act_only`, `cross_regulation`, `mental_health_diagnostics`, and `mental_health_cross_regulation`
- `evaluation/stress_questions.json` — 25 robustness questions: `paraphrase`, `negative_control`, `adversarial`, `cross_regulation_conflict`

Run the batch evaluation (hybrid is the default production path):

```bash
python evaluation/run_evaluation.py
```

Run all three retrieval modes over the same questions, writing `results/<mode>/`:

```bash
python evaluation/run_evaluation.py --mode all
```

Useful flags: `--input` (question set), `--output-dir`, `--model`, `--limit`, `--max-questions`, `--question-id`, `--category`.

Each run writes `evaluation_results.json`, `.csv`, and `.md` to its output directory.

Build the semantic/graph/hybrid ablation table for the thesis:

```bash
python evaluation/compare_modes.py
```

This aggregates `results/{semantic,graph,hybrid}/evaluation_results.json` by category × mode and writes `ablation_comparison.md` / `.json`. `overall_score` and `faithfulness_score` require answer generation, so they are reported as `n/a` for the retrieval-only modes rather than as a failing 0.

Supplementary reference-free RAGAS scoring over the hybrid results:

```bash
python evaluation/ragas_supplement.py
```

Extraction-quality harness (H1 scores the pipeline against the human-validated gold annotations in `evaluation/gold_sample/`; H2 measures schema-validation behaviour on a larger unannotated sample):

```bash
python -m evaluation.extraction_eval score-h1
python -m evaluation.extraction_eval run-h2
```

### Evaluation page

`/evaluation` renders a read-only report over the pre-computed results — no evaluation runs on page load. It reads `evaluation/results/hybrid/evaluation_results.json` (the production run), falling back to a flat `evaluation/results/evaluation_results.json` from a single-mode run. It shows:

- summary tiles: questions evaluated, mean overall score, mean latency, failed rows
- **retrieval-mode comparison**: mean recall per mode, then a metric-by-metric table of category × mode with the winning mode flagged, plus abstention accuracy when the run has `expects_abstention` questions
- category breakdown and a sortable per-question table; click a row for the answer, citations, and matched/missed concepts and keywords

The comparison section only appears once per-mode results exist — run `--mode all` first.

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
rag/          Graph-RAG planner, retrievers, context builder, determination, answer generator
web/          Flask UI (console + evaluation report) and service wrapper
evaluation/   Golden/stress question sets, batch runner, mode comparison, results
docs/         Design and thesis notes
tests/        Unit tests
```

## Data And Local Artifacts

Large local inputs and generated artifacts are intentionally ignored by git:

- `data/`
- `*.png`
- `*.csv`
- `Neo4j-*.txt`
- `.env`
- `test-results/`
- virtual environments and Python caches

Evaluation `.json` and `.md` results **are** tracked — the web report and the ablation table read them, so a checkout renders the report without rerunning a multi-hour evaluation. The `.csv` siblings are regenerable and ignored.

Keep credentials in `.env`, not in source files.
