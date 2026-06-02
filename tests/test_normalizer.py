from extraction.normalizer import canonical_key, canonicalize_compliance_term, normalize_chunk_results
from extraction.schemas import ChunkExtractionResult, ExtractedNode, ExtractedRelationship, ExtractedStatement
from ingestion.chunking import DocumentChunk


def test_canonical_key_normalizes_spacing_and_case() -> None:
    assert canonical_key("Role", " Data Protection Officer ") == "Role:data protection officer"


def test_canonicalize_compliance_term_maps_aliases() -> None:
    assert canonicalize_compliance_term("DataCategory", "PHI") == (
        "ProtectedHealthInformation",
        "Protected Health Information",
    )
    assert canonicalize_compliance_term("Actor", "high risk ai system") == (
        "HighRiskAISystem",
        "High-Risk AI System",
    )


def test_normalize_chunk_results_deduplicates_nodes() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        regulation_name="sample.md",
        section_id="sec-0000",
        section_title="Section 1",
        chunk_index=0,
        text="Controllers must protect personal data.",
        start_char=0,
        end_char=40,
    )
    result = ChunkExtractionResult(
        chunk_summary="summary",
        statements=[
            ExtractedStatement(
                statement_id="stmt-1",
                statement_type="obligation",
                canonical_name="Protect personal data",
                raw_text="must protect personal data",
                actor="Controller",
                action="protect",
                object="personal data",
                confidence=0.95,
                evidence_text="Controllers must protect personal data.",
                nodes=[
                    ExtractedNode(
                        node_type="Actor",
                        canonical_name="Controller",
                        raw_text="Controllers",
                        confidence=0.95,
                    )
                ],
                relationships=[
                    ExtractedRelationship(
                        relationship_type="RELATED_TO",
                        source_canonical_name="Controller",
                        source_node_type="Actor",
                        target_canonical_name="Personal Data",
                        target_node_type="DataCategory",
                        raw_text="protect personal data",
                        confidence=0.8,
                    )
                ],
            ),
            ExtractedStatement(
                statement_id="stmt-2",
                statement_type="obligation",
                canonical_name="Secure personal data",
                raw_text="shall secure personal data",
                actor="controller",
                action="secure",
                object="personal data",
                confidence=0.92,
                evidence_text="A controller shall secure personal data.",
                nodes=[
                    ExtractedNode(
                        node_type="Actor",
                        canonical_name="controller",
                        raw_text="controller",
                        confidence=0.92,
                    )
                ],
            ),
        ],
    )
    nodes, relationships, statements = normalize_chunk_results([(chunk, result)])
    actor_nodes = [node for node in nodes if node.node_type == "Actor"]
    assert len(actor_nodes) == 1
    assert relationships
    assert len(statements) == 2


def test_normalize_chunk_results_applies_compliance_aliases() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        regulation_name="hipaa.txt",
        section_id="sec-0000",
        section_title="Section 164.508",
        chunk_index=0,
        text="A covered entity must obtain an authorization for uses and disclosures of PHI.",
        start_char=0,
        end_char=76,
    )
    result = ChunkExtractionResult(
        chunk_summary="summary",
        statements=[
            ExtractedStatement(
                statement_id="stmt-1",
                statement_type="obligation",
                canonical_name="Authorization for use and disclosure",
                raw_text="must obtain an authorization",
                actor="covered entity",
                action="obtain",
                object="authorization",
                confidence=0.95,
                evidence_text="A covered entity must obtain an authorization for uses and disclosures of PHI.",
                nodes=[
                    ExtractedNode(
                        node_type="Actor",
                        canonical_name="covered entity",
                        raw_text="covered entity",
                        confidence=0.95,
                    ),
                    ExtractedNode(
                        node_type="DataCategory",
                        canonical_name="PHI",
                        raw_text="PHI",
                        confidence=0.95,
                    ),
                ],
                relationships=[
                    ExtractedRelationship(
                        relationship_type="REQUIRES_AUTHORIZATION",
                        source_canonical_name="covered entity",
                        source_node_type="Actor",
                        target_canonical_name="uses and disclosures",
                        target_node_type="ProcessingActivity",
                        raw_text="authorization for uses and disclosures",
                        confidence=0.9,
                    )
                ],
            )
        ],
    )

    nodes, relationships, statements = normalize_chunk_results([(chunk, result)])

    assert any(node.node_type == "CoveredEntity" and node.canonical_name == "Covered Entity" for node in nodes)
    assert any(
        node.node_type == "ProtectedHealthInformation" and node.canonical_name == "Protected Health Information"
        for node in nodes
    )
    assert relationships[0].source_key == "CoveredEntity:covered entity"
    assert relationships[0].target_key == "UseDisclosure:use and disclosure"
    assert relationships[0].source_documents == ["hipaa.txt"]
    assert statements[0].source_document == "hipaa.txt"


def test_chunk_extraction_result_normalizes_percentage_confidence() -> None:
    result = ChunkExtractionResult.model_validate(
        {
            "chunk_summary": "summary",
            "statements": [
                {
                    "statement_id": "stmt-1",
                    "statement_type": "obligation",
                    "canonical_name": "Protect personal data",
                    "raw_text": "must protect personal data",
                    "confidence": 100,
                    "evidence_text": "Controllers must protect personal data.",
                    "nodes": [
                        {
                            "node_type": "Actor",
                            "canonical_name": "Controller",
                            "raw_text": "Controllers",
                            "confidence": 85,
                        }
                    ],
                    "relationships": [
                        {
                            "relationship_type": "RELATED_TO",
                            "source_canonical_name": "Controller",
                            "source_node_type": "Actor",
                            "target_canonical_name": "Personal Data",
                            "target_node_type": "DataCategory",
                            "raw_text": "protect personal data",
                            "confidence": "75",
                        }
                    ],
                }
            ],
        }
    )

    statement = result.statements[0]
    assert statement.confidence == 1.0
    assert statement.nodes[0].confidence == 0.85
    assert statement.relationships[0].confidence == 0.75
