from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from graph.neo4j_client import Neo4jClient


@dataclass(frozen=True)
class OntologyClass:
    name: str
    category: str
    description: str = ""


ONTOLOGY_CLASSES: tuple[OntologyClass, ...] = (
    OntologyClass("RegulatedActor", "actor", "An actor with compliance duties or responsibilities."),
    OntologyClass("HealthcareActor", "actor", "A regulated actor in healthcare operations."),
    OntologyClass("AIActor", "actor", "A regulated actor in the AI system lifecycle."),
    OntologyClass("DataActor", "actor", "A regulated actor in personal data processing."),
    OntologyClass("Authority", "actor", "A regulatory or oversight authority."),
    OntologyClass("DataSubjectRole", "actor", "A person role whose data or rights are regulated."),
    OntologyClass("AISystem", "system", "An artificial intelligence system."),
    OntologyClass("HighRiskAISystem", "system", "A high-risk AI system."),
    OntologyClass("ClinicalAISystem", "system", "An AI system used in clinical contexts."),
    OntologyClass("MentalHealthDiagnosticSystem", "system", "A clinical AI system for mental health diagnostics."),
    OntologyClass("Data", "data", "Data handled by regulated systems or actors."),
    OntologyClass("PersonalData", "data", "Data relating to an identified or identifiable person."),
    OntologyClass("HealthData", "data", "Personal data concerning health."),
    OntologyClass("SensitiveData", "data", "Data requiring heightened protection."),
    OntologyClass("SpecialCategoryData", "data", "GDPR special categories of personal data."),
    OntologyClass("MentalHealthData", "data", "Health data about mental health or mental health diagnostics."),
    OntologyClass("ProtectedHealthInformation", "data", "HIPAA protected health information."),
    OntologyClass("ComplianceConcept", "compliance", "A compliance concept, duty, rule, risk, or permission."),
    OntologyClass("Requirement", "compliance", "A required condition or rule."),
    OntologyClass("Obligation", "compliance", "A duty imposed on an actor."),
    OntologyClass("Prohibition", "compliance", "A forbidden action or practice."),
    OntologyClass("Permission", "compliance", "An allowed action or processing basis."),
    OntologyClass("LegalBasis", "compliance", "A legal basis for processing or action."),
    OntologyClass("Consent", "compliance", "Consent as a legal or permission concept."),
    OntologyClass("ExplicitConsent", "compliance", "Explicit consent."),
    OntologyClass("Control", "compliance", "A safeguard, measure, or control."),
    OntologyClass("Risk", "compliance", "A risk considered by compliance requirements."),
    OntologyClass("SafetyRisk", "compliance", "A safety-related risk."),
    OntologyClass("BiasRisk", "compliance", "A bias or discrimination risk."),
    OntologyClass("PatientSafetyRisk", "compliance", "A patient safety risk."),
    OntologyClass("TransparencyRequirement", "compliance", "A transparency requirement."),
    OntologyClass("ExplainabilityRequirement", "compliance", "An explainability requirement."),
    OntologyClass("HumanOversightRequirement", "compliance", "A human oversight requirement."),
    OntologyClass("DataProtectionImpactAssessment", "compliance", "A GDPR data protection impact assessment."),
    OntologyClass("BreachNotification", "compliance", "A breach reporting or notification obligation."),
    OntologyClass("TechnicalDocumentation", "compliance", "Technical documentation duties."),
    OntologyClass("PostMarketMonitoring", "compliance", "Post-market monitoring duties."),
    OntologyClass("Provider", "actor", "An EU AI Act provider."),
    OntologyClass("Deployer", "actor", "An EU AI Act deployer."),
    OntologyClass("Importer", "actor", "An EU AI Act importer."),
    OntologyClass("Distributor", "actor", "An EU AI Act distributor."),
    OntologyClass("ProductManufacturer", "actor", "An EU AI Act product manufacturer."),
    OntologyClass("Controller", "actor", "A GDPR controller."),
    OntologyClass("Processor", "actor", "A GDPR processor."),
    OntologyClass("JointController", "actor", "GDPR joint controllers."),
    OntologyClass("DataSubject", "actor", "A GDPR data subject."),
    OntologyClass("DataProtectionOfficer", "actor", "A GDPR data protection officer."),
    OntologyClass("SupervisoryAuthority", "actor", "A GDPR supervisory authority."),
    OntologyClass("CoveredEntity", "actor", "A HIPAA covered entity."),
    OntologyClass("BusinessAssociate", "actor", "A HIPAA business associate."),
    OntologyClass("HealthcareProvider", "actor", "A healthcare provider."),
)

SUBCLASS_RELATIONSHIPS: tuple[tuple[str, str], ...] = (
    ("Provider", "AIActor"),
    ("Deployer", "AIActor"),
    ("Importer", "AIActor"),
    ("Distributor", "AIActor"),
    ("ProductManufacturer", "AIActor"),
    ("AIActor", "RegulatedActor"),
    ("Controller", "DataActor"),
    ("Processor", "DataActor"),
    ("JointController", "DataActor"),
    ("DataProtectionOfficer", "DataActor"),
    ("DataActor", "RegulatedActor"),
    ("CoveredEntity", "HealthcareActor"),
    ("BusinessAssociate", "HealthcareActor"),
    ("HealthcareProvider", "HealthcareActor"),
    ("HealthcareActor", "RegulatedActor"),
    ("SupervisoryAuthority", "Authority"),
    ("Authority", "RegulatedActor"),
    ("DataSubject", "DataSubjectRole"),
    ("HighRiskAISystem", "AISystem"),
    ("ClinicalAISystem", "HighRiskAISystem"),
    ("MentalHealthDiagnosticSystem", "ClinicalAISystem"),
    ("PersonalData", "Data"),
    ("HealthData", "PersonalData"),
    ("SensitiveData", "PersonalData"),
    ("SpecialCategoryData", "SensitiveData"),
    ("MentalHealthData", "HealthData"),
    ("MentalHealthData", "SpecialCategoryData"),
    ("ProtectedHealthInformation", "HealthData"),
    ("Obligation", "ComplianceConcept"),
    ("Requirement", "ComplianceConcept"),
    ("Prohibition", "ComplianceConcept"),
    ("Permission", "ComplianceConcept"),
    ("LegalBasis", "ComplianceConcept"),
    ("Consent", "LegalBasis"),
    ("ExplicitConsent", "Consent"),
    ("Control", "ComplianceConcept"),
    ("Risk", "ComplianceConcept"),
    ("SafetyRisk", "Risk"),
    ("BiasRisk", "Risk"),
    ("PatientSafetyRisk", "SafetyRisk"),
    ("TransparencyRequirement", "Requirement"),
    ("ExplainabilityRequirement", "Requirement"),
    ("HumanOversightRequirement", "Requirement"),
    ("DataProtectionImpactAssessment", "Requirement"),
    ("BreachNotification", "Obligation"),
    ("TechnicalDocumentation", "Obligation"),
    ("PostMarketMonitoring", "Obligation"),
)

ALIGNMENT_RELATIONSHIPS: tuple[tuple[str, str, str], ...] = (
    ("CoveredEntity", "OVERLAPS_WITH", "Controller"),
    ("BusinessAssociate", "OVERLAPS_WITH", "Processor"),
    ("ProtectedHealthInformation", "OVERLAPS_WITH", "HealthData"),
    ("ProtectedHealthInformation", "OVERLAPS_WITH", "PersonalData"),
    ("HighRiskAISystem", "RELATED_CONCEPT", "ClinicalAISystem"),
    ("MentalHealthDiagnosticSystem", "RELATED_CONCEPT", "HighRiskAISystem"),
)

ENTITY_ONTOLOGY_MAPPINGS: dict[tuple[str | None, str], str] = {
    ("Provider", "provider"): "Provider",
    ("Actor", "provider"): "Provider",
    ("Deployer", "deployer"): "Deployer",
    ("Actor", "deployer"): "Deployer",
    ("Actor", "importer"): "Importer",
    ("Actor", "distributor"): "Distributor",
    ("Actor", "product manufacturer"): "ProductManufacturer",
    ("Actor", "controller"): "Controller",
    ("Actor", "data controller"): "Controller",
    ("Actor", "processor"): "Processor",
    ("Actor", "data processor"): "Processor",
    ("Actor", "joint controller"): "JointController",
    ("Actor", "data subject"): "DataSubject",
    ("Role", "data subject"): "DataSubject",
    ("Role", "data protection officer"): "DataProtectionOfficer",
    ("Role", "dpo"): "DataProtectionOfficer",
    ("Regulator", "supervisory authority"): "SupervisoryAuthority",
    ("Actor", "supervisory authority"): "SupervisoryAuthority",
    ("Actor", "data protection authority"): "SupervisoryAuthority",
    ("CoveredEntity", "covered entity"): "CoveredEntity",
    ("Actor", "covered entity"): "CoveredEntity",
    ("BusinessAssociate", "business associate"): "BusinessAssociate",
    ("Actor", "business associate"): "BusinessAssociate",
    ("Actor", "healthcare provider"): "HealthcareProvider",
    ("Actor", "health care provider"): "HealthcareProvider",
    ("ProtectedHealthInformation", "protected health information"): "ProtectedHealthInformation",
    ("ProtectedHealthInformation", "phi"): "ProtectedHealthInformation",
    ("Term", "protected health information"): "ProtectedHealthInformation",
    ("Term", "phi"): "ProtectedHealthInformation",
    ("DataCategory", "protected health information"): "ProtectedHealthInformation",
    ("DataCategory", "phi"): "ProtectedHealthInformation",
    ("DataCategory", "personal data"): "PersonalData",
    ("Term", "personal data"): "PersonalData",
    ("DataCategory", "health data"): "HealthData",
    ("Term", "health data"): "HealthData",
    ("SensitiveDataCategory", "health data"): "HealthData",
    ("DataCategory", "special category data"): "SpecialCategoryData",
    ("Term", "special category data"): "SpecialCategoryData",
    ("SensitiveDataCategory", "special category data"): "SpecialCategoryData",
    ("DataCategory", "mental health data"): "MentalHealthData",
    ("Term", "mental health data"): "MentalHealthData",
    ("SensitiveDataCategory", "mental health data"): "MentalHealthData",
    ("HighRiskAISystem", "high-risk ai system"): "HighRiskAISystem",
    ("HighRiskAISystem", "high risk ai system"): "HighRiskAISystem",
    ("Term", "high-risk ai system"): "HighRiskAISystem",
    ("Term", "high risk ai system"): "HighRiskAISystem",
    ("AISystem", "ai system"): "AISystem",
    ("Term", "ai system"): "AISystem",
    ("Term", "mental health diagnostic system"): "MentalHealthDiagnosticSystem",
    ("ImpactAssessment", "data protection impact assessment"): "DataProtectionImpactAssessment",
    ("ImpactAssessment", "dpia"): "DataProtectionImpactAssessment",
    ("Term", "data protection impact assessment"): "DataProtectionImpactAssessment",
    ("Term", "dpia"): "DataProtectionImpactAssessment",
    ("Control", "human oversight"): "HumanOversightRequirement",
    ("Safeguard", "human oversight"): "HumanOversightRequirement",
    ("Term", "human oversight"): "HumanOversightRequirement",
    ("TechnicalDocumentation", "technical documentation"): "TechnicalDocumentation",
    ("Term", "technical documentation"): "TechnicalDocumentation",
    ("MonitoringActivity", "post-market monitoring"): "PostMarketMonitoring",
    ("MonitoringActivity", "post market monitoring"): "PostMarketMonitoring",
    ("Term", "post-market monitoring"): "PostMarketMonitoring",
    ("Term", "post market monitoring"): "PostMarketMonitoring",
}

ONTOLOGY_QUERY_EXPANSIONS: dict[str, list[str]] = {
    "regulated actor": ["Provider", "Deployer", "Controller", "Processor", "CoveredEntity", "BusinessAssociate"],
    "regulated actors": ["Provider", "Deployer", "Controller", "Processor", "CoveredEntity", "BusinessAssociate"],
    "ai actor": ["Provider", "Deployer", "Importer", "Distributor", "ProductManufacturer"],
    "ai actors": ["Provider", "Deployer", "Importer", "Distributor", "ProductManufacturer"],
    "data actor": ["Controller", "Processor", "JointController", "DataProtectionOfficer"],
    "data actors": ["Controller", "Processor", "JointController", "DataProtectionOfficer"],
    "health data": ["HealthData", "MentalHealthData", "ProtectedHealthInformation"],
    "sensitive data": ["SensitiveData", "SpecialCategoryData", "MentalHealthData"],
    "mental health data": ["MentalHealthData", "HealthData", "SpecialCategoryData", "ProtectedHealthInformation"],
    "ai system": ["AISystem", "HighRiskAISystem", "ClinicalAISystem", "MentalHealthDiagnosticSystem"],
    "ai systems": ["AISystem", "HighRiskAISystem", "ClinicalAISystem", "MentalHealthDiagnosticSystem"],
    "high-risk ai system": ["HighRiskAISystem", "ClinicalAISystem", "MentalHealthDiagnosticSystem"],
    "high-risk ai systems": ["HighRiskAISystem", "ClinicalAISystem", "MentalHealthDiagnosticSystem"],
    "mental health diagnostic system": ["MentalHealthDiagnosticSystem", "ClinicalAISystem", "HighRiskAISystem"],
    "mental health diagnostic systems": ["MentalHealthDiagnosticSystem", "ClinicalAISystem", "HighRiskAISystem"],
}


def seed_ontology(client: Neo4jClient) -> None:
    for ontology_class in ONTOLOGY_CLASSES:
        client.run_query(
            """
            MERGE (c:OntologyClass {name: $name})
            SET c.category = $category,
                c.description = $description
            """,
            {
                "name": ontology_class.name,
                "category": ontology_class.category,
                "description": ontology_class.description,
            },
        )

    for child, parent in SUBCLASS_RELATIONSHIPS:
        client.run_query(
            """
            MERGE (child:OntologyClass {name: $child})
            MERGE (parent:OntologyClass {name: $parent})
            MERGE (child)-[:SUBCLASS_OF]->(parent)
            """,
            {"child": child, "parent": parent},
        )

    for source, relationship_type, target in ALIGNMENT_RELATIONSHIPS:
        if relationship_type not in {"RELATED_CONCEPT", "EQUIVALENT_TO", "OVERLAPS_WITH"}:
            continue
        client.run_query(
            f"""
            MERGE (a:OntologyClass {{name: $source}})
            MERGE (b:OntologyClass {{name: $target}})
            MERGE (a)-[:{relationship_type}]->(b)
            """,
            {"source": source, "target": target},
        )


def ontology_class_for_entity(
    node_type: str | None,
    canonical_name: str,
    aliases: list[str] | None = None,
) -> str | None:
    candidates = [canonical_name, *(aliases or [])]
    for value in candidates:
        normalized_name = _normalize_key(value)
        for key in ((node_type, normalized_name), (None, normalized_name)):
            if key in ENTITY_ONTOLOGY_MAPPINGS:
                return ENTITY_ONTOLOGY_MAPPINGS[key]

    normalized_type = _normalize_key(node_type or "")
    return ENTITY_ONTOLOGY_MAPPINGS.get((None, normalized_type))


def get_subclasses(client: Neo4jClient, class_name: str) -> list[str]:
    rows = client.run_query(
        """
        MATCH (child:OntologyClass)-[:SUBCLASS_OF*0..]->(parent:OntologyClass {name: $class_name})
        RETURN child.name AS name
        ORDER BY child.name
        """,
        {"class_name": class_name},
    )
    return [str(row["name"]) for row in rows if row.get("name")]


def get_static_subclasses(class_name: str) -> list[str]:
    children_by_parent: dict[str, list[str]] = {}
    for child, parent in SUBCLASS_RELATIONSHIPS:
        children_by_parent.setdefault(parent, []).append(child)

    found: list[str] = []
    queue = [class_name]
    while queue:
        parent = queue.pop(0)
        for child in children_by_parent.get(parent, []):
            if child not in found:
                found.append(child)
                queue.append(child)
    return found


def ontology_query_expansions(term: str) -> list[str]:
    return ONTOLOGY_QUERY_EXPANSIONS.get(term.lower().strip(), [])


def _normalize_key(value: str) -> str:
    normalized = value.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    return re.sub(r"\s+", " ", normalized)
