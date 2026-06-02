from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


StatementType = Literal[
    "obligation",
    "prohibition",
    "permission",
    "definition",
    "requirement",
    "control",
    "risk",
    "exception",
    "citation",
    "other",
]

NodeType = Literal[
    "Regulation",
    "Section",
    "Requirement",
    "Obligation",
    "Prohibition",
    "Permission",
    "Role",
    "Actor",
    "DataCategory",
    "SensitiveDataCategory",
    "ProcessingActivity",
    "LegalBasis",
    "Control",
    "Risk",
    "Exception",
    "Jurisdiction",
    "Citation",
    "Definition",
    "Term",
    "AISystem",
    "HighRiskAISystem",
    "Provider",
    "Deployer",
    "CoveredEntity",
    "BusinessAssociate",
    "ProtectedHealthInformation",
    "Authorization",
    "IndividualRight",
    "Safeguard",
    "Policy",
    "Procedure",
    "Assessment",
    "ImpactAssessment",
    "TechnicalDocumentation",
    "RecordKeeping",
    "Log",
    "MonitoringActivity",
    "Incident",
    "Deadline",
    "Regulator",
    "UseDisclosure",
    "PrivacyNotice",
    "TrainingRequirement",
    "ReportingRequirement",
    "DocumentationRequirement",
    "AuditRequirement",
]

RelationshipType = Literal[
    "CONTAINS",
    "HAS_SECTION",
    "HAS_REQUIREMENT",
    "IMPOSES_OBLIGATION",
    "IMPOSES_PROHIBITION",
    "GRANTS_PERMISSION",
    "APPLIES_TO",
    "INVOLVES_DATA",
    "REQUIRES_CONTROL",
    "HAS_RISK",
    "HAS_EXCEPTION",
    "DEFINES",
    "REFERENCES",
    "CITES",
    "RELATED_TO",
    "IMPOSES_ON",
    "REQUIRES_AUTHORIZATION",
    "REQUIRES_DOCUMENTATION",
    "REQUIRES_ASSESSMENT",
    "REQUIRES_MONITORING",
    "REQUIRES_HUMAN_OVERSIGHT",
    "HAS_CONDITION",
    "HAS_DEADLINE",
    "HAS_SAFEGUARD",
    "HAS_CONTROL",
    "PROTECTS_RIGHT",
    "MITIGATES_RISK",
    "REPORTS_TO",
    "NOTIFIES",
    "DOCUMENTED_BY",
    "LIMITED_BY",
    "DISCLOSES_TO",
    "USES_FOR",
    "HAS_PURPOSE",
    "DERIVED_FROM",
]


class ExtractedNode(BaseModel):
    node_type: NodeType
    canonical_name: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    description: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source_document: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    article_number: str | None = None
    clause_number: str | None = None
    evidence_text: str | None = None
    chunk_id: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: float | int | str) -> float | int | str:
        return _normalize_confidence_value(value)


class ExtractedRelationship(BaseModel):
    relationship_type: RelationshipType
    source_canonical_name: str = Field(min_length=1)
    source_node_type: NodeType
    target_canonical_name: str = Field(min_length=1)
    target_node_type: NodeType
    raw_text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    source_document: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    article_number: str | None = None
    clause_number: str | None = None
    evidence_text: str | None = None
    chunk_id: str | None = None

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: float | int | str) -> float | int | str:
        return _normalize_confidence_value(value)


class CitationReference(BaseModel):
    citation_text: str = Field(min_length=1)
    citation_type: str | None = None


class ExtractedStatement(BaseModel):
    statement_id: str = Field(min_length=1)
    statement_type: StatementType
    canonical_name: str = Field(min_length=1)
    raw_text: str = Field(min_length=1)
    subject: str | None = None
    actor: str | None = None
    role: str | None = None
    action: str | None = None
    object: str | None = None
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    citations: list[CitationReference] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_text: str = Field(min_length=1)
    evidence_start_char: int | None = Field(default=None, ge=0)
    evidence_end_char: int | None = Field(default=None, ge=0)
    source_document: str | None = None
    section_id: str | None = None
    section_title: str | None = None
    page_number: int | None = None
    article_number: str | None = None
    clause_number: str | None = None
    chunk_id: str | None = None
    nodes: list[ExtractedNode] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: float | int | str) -> float | int | str:
        return _normalize_confidence_value(value)

    @model_validator(mode="after")
    def validate_span(self) -> "ExtractedStatement":
        if self.evidence_start_char is not None and self.evidence_end_char is not None:
            if self.evidence_end_char < self.evidence_start_char:
                raise ValueError("evidence_end_char must be >= evidence_start_char")
        return self


class ChunkExtractionResult(BaseModel):
    chunk_summary: str
    statements: list[ExtractedStatement] = Field(default_factory=list)


class NormalizedNode(BaseModel):
    node_type: NodeType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    descriptions: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    evidence_chunks: list[str] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)


class NormalizedRelationship(BaseModel):
    relationship_type: RelationshipType
    source_key: str
    target_key: str
    evidence_chunks: list[str] = Field(default_factory=list)
    source_documents: list[str] = Field(default_factory=list)
    section_ids: list[str] = Field(default_factory=list)
    section_titles: list[str] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    article_numbers: list[str] = Field(default_factory=list)
    clause_numbers: list[str] = Field(default_factory=list)
    confidence: float | None = None


class NormalizedStatement(BaseModel):
    statement_id: str
    statement_type: StatementType
    canonical_name: str
    raw_text: str
    subject: str | None = None
    actor: str | None = None
    role: str | None = None
    action: str | None = None
    object: str | None = None
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    jurisdiction: str | None = None
    citations: list[CitationReference] = Field(default_factory=list)
    confidence: float
    evidence_text: str
    evidence_start_char: int | None = None
    evidence_end_char: int | None = None
    source_document: str | None = None
    page_number: int | None = None
    article_number: str | None = None
    clause_number: str | None = None
    chunk_id: str
    section_id: str
    section_title: str


def _normalize_confidence_value(value: float | int | str) -> float | int | str:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            numeric_value = float(stripped)
        except ValueError:
            return value
    elif isinstance(value, (int, float)):
        numeric_value = float(value)
    else:
        return value

    if numeric_value > 1.0 and numeric_value <= 100.0:
        return numeric_value / 100.0
    return numeric_value
