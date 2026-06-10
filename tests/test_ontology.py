from __future__ import annotations

from typing import Any

from extraction.schemas import NormalizedNode
from graph.ontology_seed import (
    ALIGNMENT_RELATIONSHIPS,
    ONTOLOGY_CLASSES,
    SUBCLASS_RELATIONSHIPS,
    get_static_subclasses,
    get_subclasses,
    seed_ontology,
)
from graph.writer import GraphWriter
from rag.query_planner import build_query_plan


class FakeOntologyClient:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.classes: dict[str, dict[str, str]] = {}
        self.subclasses: set[tuple[str, str]] = set()
        self.alignments: set[tuple[str, str, str]] = set()

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = parameters or {}
        self.queries.append((query, params))
        if "RETURN child.name AS name" in query:
            class_name = str(params["class_name"])
            return [{"name": name} for name in sorted([class_name, *get_static_subclasses(class_name)])]
        if "category = $category" in query:
            self.classes[str(params["name"])] = {
                "category": str(params["category"]),
                "description": str(params["description"]),
            }
        elif "SUBCLASS_OF" in query:
            self.subclasses.add((str(params["child"]), str(params["parent"])))
        elif "OVERLAPS_WITH" in query:
            self.alignments.add((str(params["source"]), "OVERLAPS_WITH", str(params["target"])))
        elif "RELATED_CONCEPT" in query:
            self.alignments.add((str(params["source"]), "RELATED_CONCEPT", str(params["target"])))
        elif "EQUIVALENT_TO" in query:
            self.alignments.add((str(params["source"]), "EQUIVALENT_TO", str(params["target"])))
        return []


def _class_names() -> set[str]:
    return {ontology_class.name for ontology_class in ONTOLOGY_CLASSES}


def test_ontology_seed_contains_core_classes() -> None:
    names = _class_names()

    assert "RegulatedActor" in names
    assert "MentalHealthDiagnosticSystem" in names
    assert "MentalHealthData" in names
    assert "DataProtectionImpactAssessment" in names
    assert "PostMarketMonitoring" in names


def test_ontology_seed_contains_regulated_actor_hierarchy() -> None:
    edges = set(SUBCLASS_RELATIONSHIPS)

    assert ("Provider", "AIActor") in edges
    assert ("Controller", "DataActor") in edges
    assert ("CoveredEntity", "HealthcareActor") in edges
    assert ("AIActor", "RegulatedActor") in edges
    assert ("DataActor", "RegulatedActor") in edges
    assert ("HealthcareActor", "RegulatedActor") in edges


def test_ontology_seed_contains_ai_system_hierarchy() -> None:
    edges = set(SUBCLASS_RELATIONSHIPS)

    assert ("HighRiskAISystem", "AISystem") in edges
    assert ("ClinicalAISystem", "HighRiskAISystem") in edges
    assert ("MentalHealthDiagnosticSystem", "ClinicalAISystem") in edges


def test_ontology_seed_contains_health_data_hierarchy() -> None:
    edges = set(SUBCLASS_RELATIONSHIPS)

    assert ("HealthData", "PersonalData") in edges
    assert ("SpecialCategoryData", "SensitiveData") in edges
    assert ("MentalHealthData", "HealthData") in edges
    assert ("MentalHealthData", "SpecialCategoryData") in edges
    assert ("ProtectedHealthInformation", "HealthData") in edges


def test_ontology_seed_contains_gdpr_hipaa_eu_ai_act_actor_alignment() -> None:
    alignments = set(ALIGNMENT_RELATIONSHIPS)

    assert ("CoveredEntity", "OVERLAPS_WITH", "Controller") in alignments
    assert ("BusinessAssociate", "OVERLAPS_WITH", "Processor") in alignments
    assert ("ProtectedHealthInformation", "OVERLAPS_WITH", "HealthData") in alignments
    assert ("HighRiskAISystem", "RELATED_CONCEPT", "ClinicalAISystem") in alignments


def test_seed_ontology_is_idempotent() -> None:
    client = FakeOntologyClient()

    seed_ontology(client)  # type: ignore[arg-type]
    first_counts = (len(client.classes), len(client.subclasses), len(client.alignments))
    seed_ontology(client)  # type: ignore[arg-type]

    assert (len(client.classes), len(client.subclasses), len(client.alignments)) == first_counts


def test_entity_provider_links_to_provider_ontology_class() -> None:
    client = FakeOntologyClient()
    writer = GraphWriter(client)  # type: ignore[arg-type]

    writer._merge_nodes([NormalizedNode(node_type="Provider", canonical_name="Provider")])

    assert any(params.get("class_name") == "Provider" for _, params in client.queries)


def test_entity_controller_links_to_controller_ontology_class() -> None:
    client = FakeOntologyClient()
    writer = GraphWriter(client)  # type: ignore[arg-type]

    writer._merge_nodes([NormalizedNode(node_type="Actor", canonical_name="Controller")])

    assert any(params.get("class_name") == "Controller" for _, params in client.queries)


def test_entity_phi_links_to_protected_health_information_class() -> None:
    client = FakeOntologyClient()
    writer = GraphWriter(client)  # type: ignore[arg-type]

    writer._merge_nodes([NormalizedNode(node_type="ProtectedHealthInformation", canonical_name="PHI")])

    assert any(params.get("class_name") == "ProtectedHealthInformation" for _, params in client.queries)


def test_query_plan_expands_regulated_actor_ontology_terms() -> None:
    plan = build_query_plan("Which regulated actors have obligations?")

    assert "provider" in plan.expansion_terms
    assert "deployer" in plan.expansion_terms
    assert "controller" in plan.expansion_terms
    assert "processor" in plan.expansion_terms
    assert "covered entity" in plan.expansion_terms
    assert "business associate" in plan.expansion_terms


def test_query_plan_expands_ai_system_ontology_terms() -> None:
    plan = build_query_plan("What requirements apply to mental health diagnostic systems?")

    assert "clinical ai system" in plan.expansion_terms
    assert "high-risk ai system" in plan.expansion_terms


def test_get_subclasses_returns_regulated_actor_children() -> None:
    client = FakeOntologyClient()

    subclasses = get_subclasses(client, "RegulatedActor")  # type: ignore[arg-type]

    assert "RegulatedActor" in subclasses
    assert "Provider" in subclasses
    assert "Deployer" in subclasses
    assert "Controller" in subclasses
    assert "Processor" in subclasses
    assert "CoveredEntity" in subclasses
    assert "BusinessAssociate" in subclasses


def test_get_subclasses_returns_health_data_children() -> None:
    client = FakeOntologyClient()

    subclasses = get_subclasses(client, "HealthData")  # type: ignore[arg-type]

    assert "HealthData" in subclasses
    assert "MentalHealthData" in subclasses
    assert "ProtectedHealthInformation" in subclasses
