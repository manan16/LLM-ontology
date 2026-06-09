from __future__ import annotations

from typing import Any

from app.logger import get_logger
from extraction.normalizer import canonical_key, canonicalize_compliance_term
from extraction.schemas import NormalizedNode, NormalizedRelationship, NormalizedStatement
from graph.cypher import CONSTRAINT_STATEMENTS
from graph.neo4j_client import Neo4jClient
from graph.ontology_seed import ontology_class_for_entity, seed_ontology
from graph.mental_health_seed import seed_mental_health_use_case
from ingestion.chunking import DocumentChunk
from ingestion.loaders import LoadedDocument


logger = get_logger(__name__)

ALLOWED_NODE_LABELS = {
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
}

ALLOWED_REL_TYPES = {
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
}


class GraphWriter:
    def __init__(self, neo4j_client: Neo4jClient) -> None:
        self.neo4j_client = neo4j_client

    def ensure_schema(self) -> None:
        logger.debug("Ensuring Neo4j schema")
        self.neo4j_client.run_statements(CONSTRAINT_STATEMENTS)
        seed_ontology(self.neo4j_client)
        seed_mental_health_use_case(self.neo4j_client)

    def write_document_graph(
        self,
        document: LoadedDocument,
        chunks: list[DocumentChunk],
        nodes: list[NormalizedNode],
        relationships: list[NormalizedRelationship],
        statements: list[NormalizedStatement],
    ) -> None:
        self._clear_document_graph(document)
        self._merge_regulation(document)
        self._merge_sections_and_chunks(document, chunks)
        self._merge_nodes(nodes)
        self._merge_statements(document, statements)
        self._merge_relationships(relationships)
        logger.info(
            "Wrote graph for regulation=%s sections=%s chunks=%s statements=%s nodes=%s relationships=%s",
            document.name,
            len({chunk.section_id for chunk in chunks}),
            len(chunks),
            len(statements),
            len(nodes),
            len(relationships),
        )

    def _clear_document_graph(self, document: LoadedDocument) -> None:
        logger.debug("Clearing existing graph for regulation id=%s", document.name)
        query = """
        MATCH (r:Regulation {id: $regulation_id})
        OPTIONAL MATCH (r)-[:HAS_SECTION]->(sec:Section)
        OPTIONAL MATCH (sec)-[:CONTAINS]->(contained)
        OPTIONAL MATCH (r)-[:HAS_REQUIREMENT]->(direct_statement:Statement)
        WITH collect(DISTINCT r)
            + collect(DISTINCT sec)
            + collect(DISTINCT contained)
            + collect(DISTINCT direct_statement) AS nodes
        UNWIND nodes AS node
        WITH DISTINCT node
        WHERE node IS NOT NULL
        DETACH DELETE node
        """
        self.neo4j_client.run_query(query, {"regulation_id": document.name})

    def _merge_regulation(self, document: LoadedDocument) -> None:
        logger.debug("Merging regulation node id=%s", document.name)
        query = """
        MERGE (r:Regulation {id: $id})
        SET r.name = $name,
            r.path = $path,
            r.file_type = $file_type
        """
        self.neo4j_client.run_query(
            query,
            {
                "id": document.name,
                "name": document.name,
                "path": document.path,
                "file_type": document.file_type,
            },
        )

    def _merge_sections_and_chunks(self, document: LoadedDocument, chunks: list[DocumentChunk]) -> None:
        seen_sections: set[str] = set()
        for chunk in chunks:
            section_graph_id = f"{document.name}:{chunk.section_id}"
            if chunk.section_id not in seen_sections:
                logger.debug("Merging section id=%s title=%s", section_graph_id, chunk.section_title)
                section_query = """
                MERGE (r:Regulation {id: $regulation_id})
                MERGE (s:Section {id: $section_id})
                SET s.title = $title,
                    s.source_section_id = $source_section_id
                MERGE (r)-[:HAS_SECTION]->(s)
                """
                self.neo4j_client.run_query(
                    section_query,
                    {
                        "regulation_id": document.name,
                        "section_id": section_graph_id,
                        "title": chunk.section_title,
                        "source_section_id": chunk.section_id,
                    },
                )
                seen_sections.add(chunk.section_id)

            logger.debug("Merging source chunk id=%s section=%s", chunk.chunk_id, section_graph_id)
            chunk_query = """
            MERGE (s:Section {id: $section_id})
            MERGE (c:SourceChunk {id: $chunk_id})
            SET c.chunk_index = $chunk_index,
                c.text = $text,
                c.start_char = $start_char,
                c.end_char = $end_char,
                c.section_title = $section_title
            MERGE (s)-[:CONTAINS]->(c)
            """
            self.neo4j_client.run_query(
                chunk_query,
                {
                    "section_id": section_graph_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "section_title": chunk.section_title,
                },
            )

    def _merge_nodes(self, nodes: list[NormalizedNode]) -> None:
        for node in nodes:
            if node.node_type not in ALLOWED_NODE_LABELS:
                continue
            logger.debug("Merging entity key=%s type=%s", canonical_key(node.node_type, node.canonical_name), node.node_type)
            query = f"""
            MERGE (e:Entity:{node.node_type} {{key: $key}})
            SET e.canonical_name = $canonical_name,
                e.aliases = $aliases,
                e.descriptions = $descriptions,
                e.node_type = $node_type,
                e.source_documents = $source_documents,
                e.evidence_chunks = $evidence_chunks,
                e.evidence_texts = $evidence_texts
            """
            self.neo4j_client.run_query(
                query,
                {
                    "key": canonical_key(node.node_type, node.canonical_name),
                    "canonical_name": node.canonical_name,
                    "aliases": node.aliases,
                    "descriptions": node.descriptions,
                    "node_type": node.node_type,
                    "source_documents": node.source_documents,
                    "evidence_chunks": node.evidence_chunks,
                    "evidence_texts": node.evidence_texts,
                },
            )
            self._link_entity_to_ontology(node.node_type, node.canonical_name, node.aliases)

    def _merge_statements(self, document: LoadedDocument, statements: list[NormalizedStatement]) -> None:
        for statement in statements:
            statement_node_type = _statement_node_type(statement.statement_type)
            logger.debug(
                "Merging statement id=%s type=%s section=%s chunk=%s",
                statement.statement_id,
                statement_node_type,
                statement.section_id,
                statement.chunk_id,
            )
            query = f"""
            MERGE (s:Statement:{statement_node_type} {{id: $statement_id}})
            SET s.statement_type = $statement_type,
                s.canonical_name = $canonical_name,
                s.raw_text = $raw_text,
                s.subject = $subject,
                s.actor = $actor,
                s.role = $role,
                s.action = $action,
                s.object = $object,
                s.conditions = $conditions,
                s.exceptions = $exceptions,
                s.jurisdiction = $jurisdiction,
                s.confidence = $confidence,
                s.evidence_text = $evidence_text,
                s.evidence_start_char = $evidence_start_char,
                s.evidence_end_char = $evidence_end_char,
                s.source_document = $source_document,
                s.section_id = $section_id,
                s.section_title = $section_title,
                s.page_number = $page_number,
                s.article_number = $article_number,
                s.clause_number = $clause_number,
                s.chunk_id = $chunk_id,
                s.citations = $citations
            WITH s
            MATCH (c:SourceChunk {{id: $chunk_id}})
            MATCH (sec:Section {{id: $section_graph_id}})
            MERGE (sec)-[:HAS_REQUIREMENT]->(s)
            MERGE (s)-[:REFERENCES]->(c)
            """
            self.neo4j_client.run_query(
                query,
                {
                    "statement_id": statement.statement_id,
                    "statement_type": statement.statement_type,
                    "canonical_name": statement.canonical_name,
                    "raw_text": statement.raw_text,
                    "subject": statement.subject,
                    "actor": statement.actor,
                    "role": statement.role,
                    "action": statement.action,
                    "object": statement.object,
                    "conditions": statement.conditions,
                    "exceptions": statement.exceptions,
                    "jurisdiction": statement.jurisdiction,
                    "confidence": statement.confidence,
                    "evidence_text": statement.evidence_text,
                    "evidence_start_char": statement.evidence_start_char,
                    "evidence_end_char": statement.evidence_end_char,
                    "source_document": statement.source_document or document.name,
                    "section_id": statement.section_id,
                    "section_title": statement.section_title,
                    "page_number": statement.page_number,
                    "article_number": statement.article_number,
                    "clause_number": statement.clause_number,
                    "citations": [citation.citation_text for citation in statement.citations],
                    "chunk_id": statement.chunk_id,
                    "section_graph_id": f"{document.name}:{statement.section_id}",
                },
            )
            self._link_statement_entities(statement)

    def _link_statement_entities(self, statement: NormalizedStatement) -> None:
        target_specs: list[dict[str, Any]] = []
        if statement.actor:
            target_specs.append({"node_type": "Actor", "canonical_name": statement.actor, "rel": "APPLIES_TO"})
        if statement.role:
            target_specs.append({"node_type": "Role", "canonical_name": statement.role, "rel": "APPLIES_TO"})
        if statement.object:
            target_specs.append({"node_type": "Term", "canonical_name": statement.object, "rel": "RELATED_TO"})
        if statement.jurisdiction:
            target_specs.append(
                {"node_type": "Jurisdiction", "canonical_name": statement.jurisdiction, "rel": "APPLIES_TO"}
            )
        for item in statement.exceptions:
            target_specs.append({"node_type": "Exception", "canonical_name": item, "rel": "HAS_EXCEPTION"})
        for citation in statement.citations:
            target_specs.append({"node_type": "Citation", "canonical_name": citation.citation_text, "rel": "CITES"})

        for spec in target_specs:
            node_type, canonical_name = canonicalize_compliance_term(spec["node_type"], spec["canonical_name"])
            node_key = canonical_key(node_type, canonical_name)
            logger.debug(
                "Linking statement id=%s to entity=%s rel=%s",
                statement.statement_id,
                node_key,
                spec["rel"],
            )
            merge_entity = f"""
            MERGE (e:Entity:{node_type} {{key: $node_key}})
            SET e.canonical_name = $canonical_name,
                e.node_type = $node_type
            """
            self.neo4j_client.run_query(
                merge_entity,
                {
                    "node_key": node_key,
                    "canonical_name": canonical_name,
                    "node_type": node_type,
                },
            )
            self._link_entity_to_ontology(node_type, canonical_name)

            rel_type = spec["rel"] if spec["rel"] in ALLOWED_REL_TYPES else "RELATED_TO"
            primary_query = f"""
            MATCH (s:Statement {{id: $statement_id}})
            MATCH (e:Entity {{key: $node_key}})
            MERGE (s)-[:{rel_type}]->(e)
            """
            self.neo4j_client.run_query(primary_query, {"statement_id": statement.statement_id, "node_key": node_key})

    def _link_entity_to_ontology(
        self,
        node_type: str,
        canonical_name: str,
        aliases: list[str] | None = None,
    ) -> None:
        ontology_class = ontology_class_for_entity(node_type, canonical_name, aliases)
        if not ontology_class:
            return
        query = """
        MATCH (e:Entity {key: $entity_key})
        MERGE (c:OntologyClass {name: $class_name})
        MERGE (e)-[:INSTANCE_OF]->(c)
        """
        self.neo4j_client.run_query(
            query,
            {
                "entity_key": canonical_key(node_type, canonical_name),
                "class_name": ontology_class,
            },
        )

    def _merge_relationships(self, relationships: list[NormalizedRelationship]) -> None:
        for relationship in relationships:
            if relationship.relationship_type not in ALLOWED_REL_TYPES:
                continue
            logger.debug(
                "Merging entity relationship type=%s source=%s target=%s",
                relationship.relationship_type,
                relationship.source_key,
                relationship.target_key,
            )
            query = f"""
            MATCH (a:Entity {{key: $source_key}})
            MATCH (b:Entity {{key: $target_key}})
            MERGE (a)-[r:{relationship.relationship_type}]->(b)
            SET r.evidence_chunks = $evidence_chunks,
                r.source_documents = $source_documents,
                r.section_ids = $section_ids,
                r.section_titles = $section_titles,
                r.evidence_texts = $evidence_texts,
                r.page_numbers = $page_numbers,
                r.article_numbers = $article_numbers,
                r.clause_numbers = $clause_numbers,
                r.confidence = $confidence
            """
            self.neo4j_client.run_query(
                query,
                {
                    "source_key": relationship.source_key,
                    "target_key": relationship.target_key,
                    "evidence_chunks": relationship.evidence_chunks,
                    "source_documents": relationship.source_documents,
                    "section_ids": relationship.section_ids,
                    "section_titles": relationship.section_titles,
                    "evidence_texts": relationship.evidence_texts,
                    "page_numbers": relationship.page_numbers,
                    "article_numbers": relationship.article_numbers,
                    "clause_numbers": relationship.clause_numbers,
                    "confidence": relationship.confidence,
                },
            )


def _statement_node_type(statement_type: str) -> str:
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
