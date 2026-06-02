from extraction.prompts import SYSTEM_PROMPT, build_extraction_prompt
from extraction.schemas import ChunkExtractionResult
from ingestion.chunking import DocumentChunk


def test_schema_accepts_compliance_entity_and_relationship_types() -> None:
    result = ChunkExtractionResult.model_validate(
        {
            "chunk_summary": "summary",
            "statements": [
                {
                    "statement_id": "stmt-1",
                    "statement_type": "requirement",
                    "canonical_name": "Deployer assessment",
                    "raw_text": "Deployers shall perform a fundamental rights impact assessment.",
                    "confidence": 0.96,
                    "evidence_text": "Deployers shall perform a fundamental rights impact assessment.",
                    "article_number": "27",
                    "chunk_id": "chunk-1",
                    "nodes": [
                        {
                            "node_type": "Deployer",
                            "canonical_name": "Deployer",
                            "raw_text": "Deployers",
                            "confidence": 0.95,
                            "evidence_text": "Deployers",
                        },
                        {
                            "node_type": "ImpactAssessment",
                            "canonical_name": "Fundamental Rights Impact Assessment",
                            "raw_text": "fundamental rights impact assessment",
                            "confidence": 0.95,
                        },
                    ],
                    "relationships": [
                        {
                            "relationship_type": "REQUIRES_ASSESSMENT",
                            "source_canonical_name": "Deployer",
                            "source_node_type": "Deployer",
                            "target_canonical_name": "Fundamental Rights Impact Assessment",
                            "target_node_type": "ImpactAssessment",
                            "raw_text": "shall perform a fundamental rights impact assessment",
                            "confidence": 0.94,
                            "article_number": "27",
                            "evidence_text": "Deployers shall perform a fundamental rights impact assessment.",
                        }
                    ],
                }
            ],
        }
    )

    statement = result.statements[0]
    assert statement.article_number == "27"
    assert statement.nodes[0].node_type == "Deployer"
    assert statement.relationships[0].relationship_type == "REQUIRES_ASSESSMENT"


def test_extraction_prompt_contains_compliance_guidance() -> None:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        regulation_name="eu-ai-act.txt",
        section_id="sec-0000",
        section_title="Article 14",
        chunk_index=0,
        text="High-risk AI systems shall be designed for human oversight.",
        start_char=0,
        end_char=62,
    )

    prompt = build_extraction_prompt(chunk)

    assert "who has the duty" in prompt
    assert '"must not", "shall not", "prohibited", "forbidden"' in prompt
    assert "Covered Entity, Business Associate, Protected Health Information" in prompt
    assert "Provider, Deployer, AI System, High-Risk AI System" in prompt
    assert "REQUIRES_HUMAN_OVERSIGHT" in prompt
    assert "compliance-aware ontology extraction engine" in SYSTEM_PROMPT
