from __future__ import annotations

import re
from collections import defaultdict

from app.logger import get_logger
from extraction.schemas import (
    ChunkExtractionResult,
    NormalizedNode,
    NormalizedRelationship,
    NormalizedStatement,
)
from ingestion.chunking import DocumentChunk

logger = get_logger(__name__)

COMPLIANCE_ALIASES: dict[str, tuple[str, str]] = {
    "phi": ("ProtectedHealthInformation", "Protected Health Information"),
    "protected health information": ("ProtectedHealthInformation", "Protected Health Information"),
    "covered entity": ("CoveredEntity", "Covered Entity"),
    "business associate": ("BusinessAssociate", "Business Associate"),
    "ai system": ("AISystem", "AI System"),
    "high risk ai system": ("HighRiskAISystem", "High-Risk AI System"),
    "high-risk ai system": ("HighRiskAISystem", "High-Risk AI System"),
    "provider": ("Provider", "Provider"),
    "deployer": ("Deployer", "Deployer"),
    "fundamental rights impact assessment": (
        "ImpactAssessment",
        "Fundamental Rights Impact Assessment",
    ),
    "technical documentation": ("TechnicalDocumentation", "Technical Documentation"),
    "human oversight": ("Safeguard", "Human Oversight"),
    "minimum necessary": ("Requirement", "Minimum Necessary Rule"),
    "minimum necessary rule": ("Requirement", "Minimum Necessary Rule"),
    "authorization": ("Authorization", "Authorization"),
    "use and disclosure": ("UseDisclosure", "Use and Disclosure"),
    "uses and disclosures": ("UseDisclosure", "Use and Disclosure"),
    "controller": ("Actor", "Controller"),
    "data controller": ("Actor", "Controller"),
    "processor": ("Actor", "Processor"),
    "data processor": ("Actor", "Processor"),
    "data subject": ("Actor", "Data Subject"),
    "supervisory authority": ("Regulator", "Supervisory Authority"),
    "data protection authority": ("Regulator", "Supervisory Authority"),
    "dpa": ("Regulator", "Supervisory Authority"),
    "data protection officer": ("Role", "Data Protection Officer"),
    "dpo": ("Role", "Data Protection Officer"),
    "personal data": ("DataCategory", "Personal Data"),
    "special category data": ("SensitiveDataCategory", "Special Category Data"),
    "special categories of personal data": ("SensitiveDataCategory", "Special Category Data"),
    "health data": ("SensitiveDataCategory", "Health Data"),
    "data concerning health": ("SensitiveDataCategory", "Health Data"),
    "lawful basis": ("LegalBasis", "Lawful Basis"),
    "legal basis": ("LegalBasis", "Lawful Basis"),
    "explicit consent": ("LegalBasis", "Explicit Consent"),
    "consent": ("LegalBasis", "Consent"),
    "data protection impact assessment": ("ImpactAssessment", "Data Protection Impact Assessment"),
    "dpia": ("ImpactAssessment", "Data Protection Impact Assessment"),
    "personal data breach": ("Incident", "Personal Data Breach"),
    "data breach": ("Incident", "Personal Data Breach"),
    "right to erasure": ("IndividualRight", "Right to Erasure"),
    "right to be forgotten": ("IndividualRight", "Right to Erasure"),
}


def canonical_key(node_type: str, name: str) -> str:
    simplified = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    simplified = re.sub(r"\s+", " ", simplified)
    return f"{node_type}:{simplified}"


def canonicalize_compliance_term(node_type: str, name: str) -> tuple[str, str]:
    simplified = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    simplified = re.sub(r"\s+", " ", simplified)
    return COMPLIANCE_ALIASES.get(simplified, (node_type, name.strip()))


def normalize_chunk_results(
    chunk_results: list[tuple[DocumentChunk, ChunkExtractionResult]]
) -> tuple[list[NormalizedNode], list[NormalizedRelationship], list[NormalizedStatement]]:
    logger.debug("Normalizing chunk results chunk_count=%s", len(chunk_results))
    node_index: dict[str, NormalizedNode] = {}
    relationship_index: dict[str, NormalizedRelationship] = {}
    statements: list[NormalizedStatement] = []

    for chunk, result in chunk_results:
        for statement in result.statements:
            statements.append(
                NormalizedStatement(
                    statement_id=f"{chunk.chunk_id}:{statement.statement_id}",
                    statement_type=statement.statement_type,
                    canonical_name=statement.canonical_name.strip(),
                    raw_text=statement.raw_text.strip(),
                    subject=statement.subject,
                    actor=statement.actor,
                    role=statement.role,
                    action=statement.action,
                    object=statement.object,
                    conditions=[item.strip() for item in statement.conditions if item.strip()],
                    exceptions=[item.strip() for item in statement.exceptions if item.strip()],
                    jurisdiction=statement.jurisdiction,
                    citations=statement.citations,
                    confidence=statement.confidence,
                    evidence_text=statement.evidence_text.strip(),
                    evidence_start_char=statement.evidence_start_char,
                    evidence_end_char=statement.evidence_end_char,
                    source_document=_canonical_source_document(statement.source_document, chunk.regulation_name),
                    page_number=statement.page_number,
                    article_number=statement.article_number,
                    clause_number=statement.clause_number,
                    chunk_id=chunk.chunk_id,
                    section_id=chunk.section_id,
                    section_title=chunk.section_title,
                )
            )
            for node in statement.nodes:
                node_type, canonical_name = canonicalize_compliance_term(node.node_type, node.canonical_name)
                key = canonical_key(node_type, canonical_name)
                entry = node_index.get(key)
                if entry is None:
                    entry = NormalizedNode(
                        node_type=node_type,  # type: ignore[arg-type]
                        canonical_name=canonical_name,
                        aliases=[],
                        descriptions=[],
                    )
                    node_index[key] = entry
                _append_unique(entry.aliases, node.raw_text.strip())
                if node.canonical_name.strip() != canonical_name:
                    _append_unique(entry.aliases, node.canonical_name.strip())
                if node.description and node.description.strip() and node.description.strip() not in entry.descriptions:
                    entry.descriptions.append(node.description.strip())
                _append_unique(entry.source_documents, _canonical_source_document(node.source_document, chunk.regulation_name))
                _append_unique(entry.evidence_chunks, node.chunk_id or chunk.chunk_id)
                _append_unique(entry.evidence_texts, node.evidence_text or statement.evidence_text)

            for relationship in statement.relationships:
                source_node_type, source_name = canonicalize_compliance_term(
                    relationship.source_node_type, relationship.source_canonical_name
                )
                target_node_type, target_name = canonicalize_compliance_term(
                    relationship.target_node_type, relationship.target_canonical_name
                )
                source_key = canonical_key(source_node_type, source_name)
                target_key = canonical_key(target_node_type, target_name)
                key = f"{relationship.relationship_type}|{source_key}|{target_key}"
                entry = relationship_index.get(key)
                if entry is None:
                    entry = NormalizedRelationship(
                        relationship_type=relationship.relationship_type,
                        source_key=source_key,
                        target_key=target_key,
                        evidence_chunks=[],
                    )
                    relationship_index[key] = entry
                _append_unique(entry.evidence_chunks, relationship.chunk_id or chunk.chunk_id)
                _append_unique(
                    entry.source_documents,
                    _canonical_source_document(relationship.source_document, chunk.regulation_name),
                )
                _append_unique(entry.section_ids, relationship.section_id or chunk.section_id)
                _append_unique(entry.section_titles, relationship.section_title or chunk.section_title)
                _append_unique(entry.evidence_texts, relationship.evidence_text or relationship.raw_text)
                if relationship.page_number is not None and relationship.page_number not in entry.page_numbers:
                    entry.page_numbers.append(relationship.page_number)
                _append_unique(entry.article_numbers, relationship.article_number)
                _append_unique(entry.clause_numbers, relationship.clause_number)
                entry.confidence = max(entry.confidence or 0.0, relationship.confidence)

    _enrich_nodes_from_statements(statements, node_index)
    logger.debug(
        "Normalization complete statements=%s nodes=%s relationships=%s",
        len(statements),
        len(node_index),
        len(relationship_index),
    )
    return list(node_index.values()), list(relationship_index.values()), statements


def _enrich_nodes_from_statements(
    statements: list[NormalizedStatement], node_index: dict[str, NormalizedNode]
) -> None:
    inferred_values = defaultdict(set)
    for statement in statements:
        statement_node_type = _statement_to_node_type(statement.statement_type)
        statement_node_type, canonical_name = canonicalize_compliance_term(statement_node_type, statement.canonical_name)
        inferred_values[canonical_key(statement_node_type, canonical_name)].add(statement.raw_text)
        if statement.actor:
            node_type, canonical_name = canonicalize_compliance_term("Actor", statement.actor)
            inferred_values[canonical_key(node_type, canonical_name)].add(statement.actor)
        if statement.role:
            node_type, canonical_name = canonicalize_compliance_term("Role", statement.role)
            inferred_values[canonical_key(node_type, canonical_name)].add(statement.role)
        if statement.jurisdiction:
            inferred_values[canonical_key("Jurisdiction", statement.jurisdiction)].add(statement.jurisdiction)
        for citation in statement.citations:
            inferred_values[canonical_key("Citation", citation.citation_text)].add(citation.citation_text)
        for item in statement.exceptions:
            inferred_values[canonical_key("Exception", item)].add(item)

    for key, aliases in inferred_values.items():
        if key in node_index:
            node = node_index[key]
        else:
            node_type, canonical_name = key.split(":", 1)
            node = NormalizedNode(
                node_type=node_type,  # type: ignore[arg-type]
                canonical_name=canonical_name.title(),
                aliases=[],
                descriptions=[],
            )
            node_index[key] = node
        for alias in sorted(aliases):
            if alias not in node.aliases:
                node.aliases.append(alias)


def _statement_to_node_type(statement_type: str) -> str:
    mapping = {
        "obligation": "Obligation",
        "prohibition": "Prohibition",
        "permission": "Permission",
        "definition": "Definition",
        "requirement": "Requirement",
        "control": "Control",
        "risk": "Risk",
        "exception": "Exception",
        "citation": "Citation",
    }
    return mapping.get(statement_type, "Requirement")


def _append_unique(items: list, value: object) -> None:
    if value is None:
        return
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return
    if value not in items:
        items.append(value)


def _canonical_source_document(source_document: str | None, fallback: str) -> str:
    value = (source_document or fallback).strip()
    if value.lower() == "gdpr.pdf":
        return "GDPR"
    return value
