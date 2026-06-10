from __future__ import annotations

import main as main_module
from types import SimpleNamespace
from typing import Any

from ingestion.chunking import DocumentChunk
from ingestion.loaders import LoadedDocument
from main import build_parser, resolve_input_paths
from pipeline import run_pipeline as pipeline_module


class FakeExtractor:
    def __init__(self, ollama_client: Any) -> None:
        self.ollama_client = ollama_client

    def extract_chunk_adaptive(self, chunk: DocumentChunk) -> list[tuple[DocumentChunk, object]]:
        if chunk.chunk_id.endswith(":fail"):
            raise RuntimeError("boom")
        if chunk.chunk_id.endswith(":split"):
            left = _chunk(f"{chunk.chunk_id}:left")
            right = _chunk(f"{chunk.chunk_id}:right")
            return [(left, object()), (right, object())]
        return [(chunk, object())]


class FakeNeo4jClient:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeGraphWriter:
    written_documents: list[str] = []

    def __init__(self, neo4j_client: FakeNeo4jClient) -> None:
        self.neo4j_client = neo4j_client

    def ensure_schema(self) -> None:
        return None

    def write_document_graph(
        self,
        document: LoadedDocument,
        chunks: list[DocumentChunk],
        nodes: list[object],
        relationships: list[object],
        statements: list[object],
    ) -> None:
        self.written_documents.append(document.name)


def test_main_accepts_single_pdf_file(tmp_path: Any) -> None:
    gdpr = tmp_path / "gdpr.pdf"
    gdpr.write_text("pdf placeholder")

    args = build_parser().parse_args([str(gdpr)])
    paths = resolve_input_paths(args.document_path)

    assert paths == [gdpr]


def test_main_accepts_directory_with_multiple_pdfs(tmp_path: Any) -> None:
    (tmp_path / "hipaa.pdf").write_text("hipaa")
    (tmp_path / "gdpr.pdf").write_text("gdpr")

    paths = resolve_input_paths([str(tmp_path)])

    assert paths == [tmp_path / "hipaa.pdf", tmp_path / "gdpr.pdf"]


def test_directory_ingestion_processes_files_sequentially(monkeypatch: Any, tmp_path: Any) -> None:
    hipaa = tmp_path / "hipaa.pdf"
    gdpr = tmp_path / "gdpr.pdf"
    hipaa.write_text("hipaa")
    gdpr.write_text("gdpr")
    processed: list[str] = []

    monkeypatch.setattr("sys.argv", ["main.py", str(tmp_path)])
    monkeypatch.setattr(main_module, "run_pipeline", lambda paths: processed.extend(path.name for path in paths))

    main_module.main()

    assert processed == ["hipaa.pdf", "gdpr.pdf"]


def test_directory_ingestion_sorts_files_deterministically(tmp_path: Any) -> None:
    for name in ["zeta.pdf", "alpha.pdf", "notes.md"]:
        (tmp_path / name).write_text(name)

    paths = resolve_input_paths([str(tmp_path)])

    assert [path.name for path in paths] == ["alpha.pdf", "notes.md", "zeta.pdf"]


def test_directory_ingestion_prefers_known_regulation_order(tmp_path: Any) -> None:
    for name in ["zeta.pdf", "gdpr.pdf", "eu_ai_act.pdf", "hipaa.pdf", "alpha.pdf"]:
        (tmp_path / name).write_text(name)

    paths = resolve_input_paths([str(tmp_path)])

    assert [path.name for path in paths] == ["hipaa.pdf", "eu_ai_act.pdf", "gdpr.pdf", "alpha.pdf", "zeta.pdf"]


def test_directory_ingestion_rejects_empty_directory(tmp_path: Any) -> None:
    try:
        resolve_input_paths([str(tmp_path)])
    except ValueError as exc:
        assert "No supported input files found" in str(exc)
        assert ".pdf" in str(exc)
    else:
        raise AssertionError("empty directory should be rejected")


def test_directory_ingestion_rejects_unsupported_files(tmp_path: Any) -> None:
    (tmp_path / "image.png").write_text("png")

    try:
        resolve_input_paths([str(tmp_path)])
    except ValueError as exc:
        assert "No supported input files found" in str(exc)
        assert ".pdf" in str(exc)
    else:
        raise AssertionError("directory with unsupported files should be rejected")


def test_main_parser_accepts_multiple_document_paths() -> None:
    args = build_parser().parse_args(["data/gdpr.pdf", "data/hipaa.pdf"])

    assert args.document_path == ["data/gdpr.pdf", "data/hipaa.pdf"]


def test_pipeline_reports_summary_split_chunks_and_failures(monkeypatch: Any, capsys: Any) -> None:
    FakeGraphWriter.written_documents = []
    document = LoadedDocument(
        name="sample.md",
        path="/tmp/sample.md",
        file_type=".md",
        text="Sample text",
    )
    chunks = [_chunk("sample:ok"), _chunk("sample:split"), _chunk("sample:fail")]

    monkeypatch.setattr(pipeline_module, "get_settings", lambda: SimpleNamespace(
        log_level="CRITICAL",
        ollama_model="fake",
        ollama_timeout_seconds=1,
        chunk_max_chars=100,
        chunk_overlap_chars=10,
        neo4j_database="neo4j",
    ))
    monkeypatch.setattr(pipeline_module, "load_document", lambda path: document)
    monkeypatch.setattr(pipeline_module, "build_chunks", lambda document, max_chars, overlap_chars: chunks)
    monkeypatch.setattr(pipeline_module, "OllamaClient", lambda settings: object())
    monkeypatch.setattr(pipeline_module, "RegulationExtractor", FakeExtractor)
    monkeypatch.setattr(pipeline_module, "normalize_chunk_results", lambda results: ([object(), object()], [object()], [object(), object(), object()]))
    monkeypatch.setattr(pipeline_module, "Neo4jClient", FakeNeo4jClient)
    monkeypatch.setattr(pipeline_module, "GraphWriter", FakeGraphWriter)

    results = pipeline_module.run_pipeline(["sample.md"])

    captured = capsys.readouterr()
    output = f"{captured.out}\n{captured.err}"
    assert "chunk sample:split was split after timeout retry; processed 2 child chunks" in output
    assert "failed chunk sample:fail: boom" in output
    assert "Completed: sample.md" in output
    assert "Chunks processed: 2/3" in output
    assert "Failed chunks: 1" in output
    assert "Statements extracted: 3" in output
    assert "Entities extracted: 2" in output
    assert "Relationships extracted: 1" in output
    assert FakeGraphWriter.written_documents == ["sample.md"]
    assert results[0].document_path == "/tmp/sample.md"
    assert results[0].chunk_count == 3
    assert results[0].failed_chunk_count == 1
    assert results[0].statement_count == 3
    assert results[0].entity_count == 2
    assert results[0].relationship_count == 1


def _chunk(chunk_id: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        regulation_name="sample.md",
        section_id="sec-0000",
        section_title="Section 1",
        chunk_index=0,
        text="Text",
        start_char=0,
        end_char=4,
    )
