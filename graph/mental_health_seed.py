"""
Seed a mental-health diagnostics use-case layer into Neo4j.

This module does NOT create regulation-derived obligations.
It creates a domain-context layer derived from APMS 2023/4 so that
the KG can connect mental-health diagnostic AI questions to the
existing ontology and regulation-grounded evidence.

Expected graph pattern:

(:UseCaseSource {id: "APMS_2023_24"})
(:Entity:MentalHealthDiagnosticSystem {canonical_name: "Mental Health Diagnostic AI System"})
    -[:DERIVED_FROM]->(:UseCaseSource)
    -[:INSTANCE_OF]->(:OntologyClass {name: "MentalHealthDiagnosticSystem"})
    -[:USES_DATA]->(:OntologyClass {name: "MentalHealthData"})
    -[:GENERATES_OUTPUT]->(:OntologyClass {name: "DiagnosticPrediction"})
    -[:HAS_RISK]->(:OntologyClass {name: "PatientSafetyRisk"})
    -[:REQUIRES_CONTROL]->(:OntologyClass {name: "HumanOversightRequirement"})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class Neo4jRunnable(Protocol):
    """
    Minimal protocol expected from the project's Neo4j client.

    Your existing Neo4jClient likely exposes a `run_query`, `query`,
    or `execute_query` method. The helper `_run` below supports common
    variants so this seed can be adapted with minimal changes.
    """

    def run_query(self, query: str, parameters: dict | None = None): ...


@dataclass(frozen=True)
class UseCaseSource:
    id: str
    name: str
    source_type: str
    jurisdiction: str
    purpose: str
    citation: str
    source_document: str


@dataclass(frozen=True)
class UseCaseEntity:
    key: str
    canonical_name: str
    description: str


APMS_SOURCE = UseCaseSource(
    id="APMS_2023_24",
    name="Adult Psychiatric Morbidity Survey 2023/4",
    source_type="official_statistics",
    jurisdiction="England",
    purpose="mental_health_use_case_context",
    citation=(
        "Morris, S., Hill, S., Brugha, T., McManus, S. (Eds.). "
        "Adult Psychiatric Morbidity Survey: Survey of Mental Health "
        "and Wellbeing, England, 2023/4. NHS England."
    ),
    source_document="Adult Psychiatric Morbidity Survey 2023/4 - NHS England Digital",
)

MENTAL_HEALTH_SYSTEM = UseCaseEntity(
    key="use_case:mental_health_diagnostic_ai_system",
    canonical_name="Mental Health Diagnostic AI System",
    description=(
        "A clinical decision-support AI system that assists qualified "
        "healthcare professionals in assessing, screening, monitoring, "
        "and triaging adult mental-health conditions. The system does "
        "not replace clinician judgement."
    ),
)


# These classes should already exist from graph/ontology_seed.py.
# `seed_mental_health_use_case` MERGEs them anyway to remain idempotent.
SYSTEM_CLASS = "MentalHealthDiagnosticSystem"

DATA_CLASSES = [
    "MentalHealthData",
    "HealthData",
    "PersonalData",
    "SpecialCategoryData",
    "SensitiveData",
    "ProtectedHealthInformation",
]

OUTPUT_CLASSES = [
    "DiagnosticPrediction",
    "PatientRiskScore",
    "ClinicalDecisionSupportOutput",
    "TreatmentMonitoringIndicator",
    "SymptomSeverityScore",
]

RISK_CLASSES = [
    "PatientSafetyRisk",
    "BiasRisk",
    "SafetyRisk",
]

CONTROL_CLASSES = [
    "HumanOversightRequirement",
    "ExplainabilityRequirement",
    "TransparencyRequirement",
    "DataProtectionImpactAssessment",
]

ACTOR_CLASSES = [
    "HealthcareProvider",
    "Provider",
    "Deployer",
    "Controller",
    "Processor",
    "CoveredEntity",
    "BusinessAssociate",
    "DataSubject",
    "Patient",
    "Clinician",
]

CONDITION_CLASSES = [
    "CommonMentalHealthCondition",
    "Depression",
    "GeneralisedAnxietyDisorder",
    "PanicDisorder",
    "Phobia",
    "ObsessiveCompulsiveDisorder",
    "PostTraumaticStressDisorder",
    "AttentionDeficitHyperactivityDisorder",
    "BipolarDisorder",
    "PsychoticDisorder",
    "EatingDisorder",
    "SubstanceDependence",
    "SelfHarmRisk",
    "SuicideRisk",
]


# Optional subclass edges for classes that may not yet be in the ontology seed.
# These are conservative and useful for query expansion.
ADDITIONAL_SUBCLASS_RELATIONS = [
    ("DiagnosticPrediction", "ClinicalDecisionSupportOutput"),
    ("PatientRiskScore", "ClinicalDecisionSupportOutput"),
    ("TreatmentMonitoringIndicator", "ClinicalDecisionSupportOutput"),
    ("SymptomSeverityScore", "ClinicalDecisionSupportOutput"),
    ("ClinicalDecisionSupportOutput", "ComplianceConcept"),
    ("Patient", "DataSubjectRole"),
    ("Clinician", "HealthcareActor"),
    ("CommonMentalHealthCondition", "MentalHealthData"),
    ("Depression", "CommonMentalHealthCondition"),
    ("GeneralisedAnxietyDisorder", "CommonMentalHealthCondition"),
    ("PanicDisorder", "CommonMentalHealthCondition"),
    ("Phobia", "CommonMentalHealthCondition"),
    ("ObsessiveCompulsiveDisorder", "CommonMentalHealthCondition"),
    ("PostTraumaticStressDisorder", "MentalHealthData"),
    ("AttentionDeficitHyperactivityDisorder", "MentalHealthData"),
    ("BipolarDisorder", "MentalHealthData"),
    ("PsychoticDisorder", "MentalHealthData"),
    ("EatingDisorder", "MentalHealthData"),
    ("SubstanceDependence", "MentalHealthData"),
    ("SelfHarmRisk", "PatientSafetyRisk"),
    ("SuicideRisk", "PatientSafetyRisk"),
]


def _run(client, query: str, parameters: dict | None = None):
    """
    Compatibility wrapper for common project-specific Neo4j client APIs.

    If your Neo4jClient has only one known method, you can simplify this.
    """
    parameters = parameters or {}

    if hasattr(client, "run_query"):
        return client.run_query(query, parameters)

    if hasattr(client, "query"):
        return client.query(query, parameters)

    if hasattr(client, "execute_query"):
        return client.execute_query(query, parameters)

    if hasattr(client, "execute"):
        return client.execute(query, parameters)

    raise TypeError(
        "Unsupported Neo4j client. Expected one of: "
        "run_query(), query(), execute_query(), execute()."
    )


def _merge_ontology_classes(client, class_names: Iterable[str]) -> None:
    _run(
        client,
        """
        UNWIND $class_names AS class_name
        MERGE (c:OntologyClass {name: class_name})
        ON CREATE SET
            c.created_by = "mental_health_seed",
            c.source = "curated_use_case_seed"
        SET
            c.updated_by = "mental_health_seed"
        """,
        {"class_names": sorted(set(class_names))},
    )


def _merge_subclass_relations(client, relations: Iterable[tuple[str, str]]) -> None:
    rows = [{"child": child, "parent": parent} for child, parent in relations]
    if not rows:
        return

    _run(
        client,
        """
        UNWIND $rows AS row
        MERGE (child:OntologyClass {name: row.child})
        MERGE (parent:OntologyClass {name: row.parent})
        MERGE (child)-[:SUBCLASS_OF]->(parent)
        """,
        {"rows": rows},
    )


def _merge_use_case_source(client, source: UseCaseSource) -> None:
    _run(
        client,
        """
        MERGE (src:UseCaseSource {id: $id})
        SET
            src.name = $name,
            src.source_type = $source_type,
            src.jurisdiction = $jurisdiction,
            src.purpose = $purpose,
            src.citation = $citation,
            src.source_document = $source_document
        """,
        {
            "id": source.id,
            "name": source.name,
            "source_type": source.source_type,
            "jurisdiction": source.jurisdiction,
            "purpose": source.purpose,
            "citation": source.citation,
            "source_document": source.source_document,
        },
    )


def _merge_system_entity(client, entity: UseCaseEntity, source: UseCaseSource) -> None:
    _run(
        client,
        """
        MERGE (system:Entity:MentalHealthDiagnosticSystem {key: $key})
        SET
            system.canonical_name = $canonical_name,
            system.name = $canonical_name,
            system.description = $description,
            system.source_type = "use_case",
            system.source_document = $source_document,
            system.regulation_id = null,
            system.created_by = coalesce(system.created_by, "mental_health_seed"),
            system.updated_by = "mental_health_seed"

        MERGE (src:UseCaseSource {id: $source_id})
        MERGE (system)-[:DERIVED_FROM]->(src)

        MERGE (cls:OntologyClass {name: $system_class})
        MERGE (system)-[:INSTANCE_OF]->(cls)
        """,
        {
            "key": entity.key,
            "canonical_name": entity.canonical_name,
            "description": entity.description,
            "source_id": source.id,
            "source_document": source.source_document,
            "system_class": SYSTEM_CLASS,
        },
    )


def _link_system_to_ontology_classes(
    client,
    relationship_type: str,
    class_names: Iterable[str],
) -> None:
    """
    Link the Mental Health Diagnostic AI System entity to ontology classes.

    relationship_type is controlled by constants in this file, so this does
    not expose arbitrary user input to Cypher.
    """
    allowed_relationships = {
        "USES_DATA",
        "GENERATES_OUTPUT",
        "HAS_RISK",
        "REQUIRES_CONTROL",
        "INVOLVES_ACTOR",
        "ASSESSES_CONDITION",
    }
    if relationship_type not in allowed_relationships:
        raise ValueError(f"Unsupported relationship type: {relationship_type}")

    query = f"""
    MATCH (system:Entity:MentalHealthDiagnosticSystem {{key: $system_key}})
    UNWIND $class_names AS class_name
    MERGE (c:OntologyClass {{name: class_name}})
    MERGE (system)-[:{relationship_type}]->(c)
    """

    _run(
        client,
        query,
        {
            "system_key": MENTAL_HEALTH_SYSTEM.key,
            "class_names": sorted(set(class_names)),
        },
    )


def seed_mental_health_use_case(client) -> None:
    """
    Seed the mental-health diagnostics use-case layer.

    This function is idempotent and safe to run multiple times.
    It does not delete or rewrite existing regulation-derived nodes.
    """

    all_classes = {
        SYSTEM_CLASS,
        *DATA_CLASSES,
        *OUTPUT_CLASSES,
        *RISK_CLASSES,
        *CONTROL_CLASSES,
        *ACTOR_CLASSES,
        *CONDITION_CLASSES,
    }

    # Add parent classes referenced by ADDITIONAL_SUBCLASS_RELATIONS.
    for child, parent in ADDITIONAL_SUBCLASS_RELATIONS:
        all_classes.add(child)
        all_classes.add(parent)

    _merge_ontology_classes(client, all_classes)
    _merge_subclass_relations(client, ADDITIONAL_SUBCLASS_RELATIONS)
    _merge_use_case_source(client, APMS_SOURCE)
    _merge_system_entity(client, MENTAL_HEALTH_SYSTEM, APMS_SOURCE)

    _link_system_to_ontology_classes(client, "USES_DATA", DATA_CLASSES)
    _link_system_to_ontology_classes(client, "GENERATES_OUTPUT", OUTPUT_CLASSES)
    _link_system_to_ontology_classes(client, "HAS_RISK", RISK_CLASSES)
    _link_system_to_ontology_classes(client, "REQUIRES_CONTROL", CONTROL_CLASSES)
    _link_system_to_ontology_classes(client, "INVOLVES_ACTOR", ACTOR_CLASSES)
    _link_system_to_ontology_classes(client, "ASSESSES_CONDITION", CONDITION_CLASSES)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from app.config import get_settings
    from graph.neo4j_client import Neo4jClient

    client = Neo4jClient(get_settings())
    try:
        seed_mental_health_use_case(client)
        print("Seeded mental-health diagnostic AI use-case layer.")
    finally:
        client.close()
