from __future__ import annotations

from typing import Any

from extraction.schemas import NormalizedNode, NormalizedRelationship, NormalizedStatement
from graph.writer import GraphWriter
from ingestion.chunking import DocumentChunk
from ingestion.loaders import LoadedDocument


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((query, parameters or {}))
        return []


def test_writer_keeps_regulation_from_becoming_statement_hub() -> None:
    document = LoadedDocument(
        name="sample.md",
        path="/tmp/sample.md",
        file_type=".md",
        text="Controllers must protect data.",
    )
    chunk = DocumentChunk(
        chunk_id="sample.md:sec-0000:0:abc",
        regulation_name="sample.md",
        section_id="sec-0000",
        section_title="Section 1",
        chunk_index=0,
        text="Controllers must protect data.",
        start_char=0,
        end_char=30,
    )
    statement = NormalizedStatement(
        statement_id="sample.md:sec-0000:0:abc:stmt-1",
        statement_type="obligation",
        canonical_name="Protect data",
        raw_text="Controllers must protect data.",
        actor="Controller",
        action="protect",
        object="data",
        confidence=0.95,
        evidence_text="Controllers must protect data.",
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
    )

    client = FakeNeo4jClient()
    writer = GraphWriter(client)  # type: ignore[arg-type]
    writer.write_document_graph(document, [chunk], [], [], [statement])

    combined_queries = "\n".join(query for query, _ in client.queries)
    assert "DETACH DELETE node" in combined_queries
    assert "MERGE (r)-[:HAS_REQUIREMENT]->(s)" not in combined_queries
    assert "MERGE (sec)-[:HAS_REQUIREMENT]->(s)" in combined_queries
    assert "IMPOSES_OBLIGATION" not in combined_queries


def test_writer_handles_compliance_metadata_fields() -> None:
    document = LoadedDocument(
        name="eu-ai-act.txt",
        path="/tmp/eu-ai-act.txt",
        file_type=".txt",
        text="Providers shall keep technical documentation.",
    )
    chunk = DocumentChunk(
        chunk_id="eu-ai-act.txt:sec-0000:0:abc",
        regulation_name="eu-ai-act.txt",
        section_id="sec-0000",
        section_title="Article 11",
        chunk_index=0,
        text="Providers shall keep technical documentation.",
        start_char=0,
        end_char=45,
    )
    nodes = [
        NormalizedNode(
            node_type="Provider",
            canonical_name="Provider",
            aliases=["Providers"],
            source_documents=["eu-ai-act.txt"],
            evidence_chunks=[chunk.chunk_id],
            evidence_texts=["Providers"],
        ),
        NormalizedNode(
            node_type="TechnicalDocumentation",
            canonical_name="Technical Documentation",
            aliases=["technical documentation"],
            source_documents=["eu-ai-act.txt"],
            evidence_chunks=[chunk.chunk_id],
            evidence_texts=["technical documentation"],
        ),
    ]
    relationships = [
        NormalizedRelationship(
            relationship_type="REQUIRES_DOCUMENTATION",
            source_key="Provider:provider",
            target_key="TechnicalDocumentation:technical documentation",
            evidence_chunks=[chunk.chunk_id],
            source_documents=["eu-ai-act.txt"],
            section_ids=["sec-0000"],
            section_titles=["Article 11"],
            evidence_texts=["Providers shall keep technical documentation."],
            article_numbers=["11"],
            confidence=0.93,
        )
    ]
    statement = NormalizedStatement(
        statement_id="eu-ai-act.txt:sec-0000:0:abc:stmt-1",
        statement_type="requirement",
        canonical_name="Keep technical documentation",
        raw_text="Providers shall keep technical documentation.",
        actor="Provider",
        action="keep",
        object="technical documentation",
        confidence=0.95,
        evidence_text="Providers shall keep technical documentation.",
        source_document="eu-ai-act.txt",
        article_number="11",
        chunk_id=chunk.chunk_id,
        section_id=chunk.section_id,
        section_title=chunk.section_title,
    )

    client = FakeNeo4jClient()
    writer = GraphWriter(client)  # type: ignore[arg-type]
    writer.write_document_graph(document, [chunk], nodes, relationships, [statement])

    combined_queries = "\n".join(query for query, _ in client.queries)
    params = [parameters for _, parameters in client.queries]

    assert "MERGE (e:Entity:Provider" in combined_queries
    assert "REQUIRES_DOCUMENTATION" in combined_queries
    assert any(parameters.get("source_document") == "eu-ai-act.txt" for parameters in params)
    assert any(parameters.get("article_number") == "11" for parameters in params)
    assert any(parameters.get("source_documents") == ["eu-ai-act.txt"] for parameters in params)
