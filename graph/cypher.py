from __future__ import annotations

CONSTRAINT_STATEMENTS = [
    "CREATE CONSTRAINT regulation_id IF NOT EXISTS FOR (n:Regulation) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT source_chunk_id IF NOT EXISTS FOR (n:SourceChunk) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT statement_id IF NOT EXISTS FOR (n:Statement) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE",
    "CREATE CONSTRAINT ontology_class_name IF NOT EXISTS FOR (c:OntologyClass) REQUIRE c.name IS UNIQUE",
    "CREATE INDEX citation_name IF NOT EXISTS FOR (n:Citation) ON (n.canonical_name)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.node_type)",
    "CREATE INDEX statement_source_document IF NOT EXISTS FOR (n:Statement) ON (n.source_document)",
    "CREATE INDEX ontology_class_category IF NOT EXISTS FOR (c:OntologyClass) ON (c.category)",
]
