from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ask import build_parser
from rag.answer_generator import AnswerGenerator
from rag.context_builder import build_context, context_row_ids, get_semantic_group, is_suppressed_context_row
from rag.prompts import RAG_PROMPT_TEMPLATE, build_rag_prompt
from rag.query_planner import build_query_plan
from rag.retriever import (
    GraphRetriever,
    _COMMON_RETURN,
    _dedupe_rows,
    _is_evidence_expansion_seed,
    _select_final_rows,
    extract_query_terms,
)


class FakeNeo4jClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, parameters or {}))
        if "MATCH (seed:Entity)" in query:
            return [
                {
                    "seed_name": "Covered Entity",
                    "seed_labels": ["Entity", "CoveredEntity"],
                    "relationship": "APPLIES_TO",
                    "related_name": "Minimum Necessary Requirement",
                    "related_labels": ["Statement", "Requirement"],
                    "statement_name": "Minimum Necessary Requirement",
                    "statement_labels": ["Statement", "Requirement"],
                    "evidence_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                    "citation": "45 CFR 164.502",
                    "section_title": "Uses and disclosures",
                    "page_number": 12,
                    "article_number": None,
                    "clause_number": None,
                    "source_document": "hipaa.pdf",
                    "source_chunk_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                }
            ]
        return [
            {
                "seed_name": "Minimum Necessary Requirement",
                "seed_labels": ["Statement", "Requirement"],
                "relationship": "REFERENCES",
                "related_name": "hipaa.pdf:sec-1:0",
                "related_labels": ["SourceChunk"],
                "statement_name": "Minimum Necessary Requirement",
                "statement_labels": ["Statement", "Requirement"],
                "evidence_text": "A covered entity must make reasonable efforts to limit PHI disclosure.",
                "citation": "45 CFR 164.502",
                "source_document": "hipaa.pdf",
            }
        ]


class FakeOllamaClient:
    def __init__(self) -> None:
        self.model = "fake"
        self.calls: list[tuple[str, str]] = []

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "Grounded answer"


class FakeAiRiskExpansionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.calls.append((query, parameters or {}))
        if "coalesce(seed.key, seed.id" in query and "original_entity_name" in query:
            return [
                {
                    "seed_name": "High-risk AI System",
                    "seed_labels": ["Entity"],
                    "relationship": "MITIGATES_RISK",
                    "related_name": "Risk Management System Requirement",
                    "related_labels": ["Statement", "Requirement"],
                    "statement_name": "Risk Management System Requirement",
                    "statement_labels": ["Statement", "Requirement"],
                    "evidence_text": "Providers of high-risk AI systems must establish a risk management system with control measures and safeguards.",
                    "source_document": "eu_ai_act.pdf",
                    "original_entity_name": "High-risk AI System",
                    "relationship_path": "RELATED_TO > MITIGATES_RISK",
                    "score": 34,
                }
            ]
        if "MATCH (seed:Entity)" in query:
            return [
                {
                    "seed_name": "High-risk AI System",
                    "seed_labels": ["Entity"],
                    "related_name": "High-risk AI System",
                    "related_labels": ["Entity"],
                    "statement_name": "Unknown",
                    "statement_labels": [],
                    "score": 35,
                }
            ]
        if "MATCH (seed:Statement)" in query:
            return [
                {
                    "statement_name": "Minimum Necessary Requirement",
                    "statement_labels": ["Statement", "Requirement"],
                    "relationship": "HAS_REQUIREMENT",
                    "evidence_text": "A covered entity must make reasonable efforts to limit PHI disclosure to the minimum necessary.",
                    "source_document": "hipaa.pdf",
                    "score": 22,
                }
            ]
        return []


class FakeEntitySeedExpansionClient:
    def __init__(self, seed_source_document: str | None = None) -> None:
        self.seed_source_document = seed_source_document
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = parameters or {}
        self.calls.append((query, params))
        if "MATCH (seed:Entity)" in query:
            return [
                {
                    "seed_id": 101,
                    "seed_key": "right_to_data_portability",
                    "seed_name": "Right To Data Portability",
                    "seed_labels": ["Entity"],
                    "related_name": "Right To Data Portability",
                    "related_labels": ["Entity"],
                    "statement_name": "Right To Data Portability",
                    "statement_labels": ["Entity"],
                    "source_document": self.seed_source_document,
                    "score": 25,
                }
            ]
        if "MATCH path = (seed)-[*1..2]-(statement)" in query:
            return [
                {
                    "seed_id": 101,
                    "seed_key": "right_to_data_portability",
                    "seed_name": "Right To Data Portability",
                    "seed_labels": ["Entity"],
                    "relationship": "REFERENCES",
                    "related_name": "Right to Data Portability Evidence",
                    "related_labels": ["Statement", "Permission"],
                    "statement_key": "gdpr-portability",
                    "statement_name": "Right to Data Portability",
                    "statement_labels": ["Statement", "Permission"],
                    "evidence_text": "The data subject should have the right to have personal data transmitted directly from one controller to another.",
                    "source_document": "gdpr.pdf",
                    "expanded_from_entity_seed": True,
                    "expanded_from_seed": "Right To Data Portability",
                    "expansion_path": "RELATED_TO > REFERENCES",
                    "score": 70,
                }
            ]
        return []


def test_extract_query_terms_keeps_domain_phrases_and_removes_stopwords() -> None:
    terms = extract_query_terms("When can a covered entity disclose PHI without authorization?")

    assert "covered entity" in terms
    assert "phi" in terms
    assert "authorization" in terms
    assert "when" not in terms
    assert "can" not in terms


def test_query_plan_categorizes_expands_and_decomposes_compliance_questions() -> None:
    plan = build_query_plan("What obligations do providers and deployers have for high-risk AI systems?")

    assert plan.category == "compliance_question"
    assert plan.intent == "obligations"
    assert plan.detected_actors == ["provider", "deployer"]
    assert "high-risk ai system" in plan.phrases
    assert "technical documentation" in plan.expansion_terms
    assert plan.needs_decomposition is True
    assert plan.sub_questions == [
        "what obligations do providers have for high-risk ai systems?",
        "what obligations do deployers have for high-risk ai systems?",
    ]


def test_query_plan_extracts_concept_groups_for_high_risk_provider_query() -> None:
    plan = build_query_plan("What obligations apply to providers of high-risk AI systems?")

    assert plan.intent == "obligations"
    assert "provider" in plan.detected_actors
    assert "high-risk ai system" in plan.detected_objects
    assert "ai system" in plan.detected_objects
    assert "provider" in plan.ambiguous_terms
    assert plan.concept_groups["actor"] == ["provider", "providers"]
    assert "high-risk ai system" in plan.concept_groups["object"]
    assert "ai system" in plan.concept_groups["object"]
    assert "obligation" in plan.concept_groups["modality"]
    assert "requirement" in plan.concept_groups["modality"]


def test_query_plan_detects_gdpr_domain() -> None:
    plan = build_query_plan("What obligations apply to controllers processing personal data under GDPR?")

    assert "GDPR" in plan.detected_domains
    assert "controller" in plan.detected_actors
    assert "personal data" in plan.detected_objects
    assert "gdpr" in plan.concept_groups["domain"]


def test_query_plan_detects_explicit_target_regulations() -> None:
    gdpr_plan = build_query_plan("What rights does a data subject have under GDPR?")
    hipaa_plan = build_query_plan("What safeguards are required under HIPAA?")
    eu_ai_plan = build_query_plan("What requirements apply under the EU AI Act?")

    assert gdpr_plan.explicit_regulations == ["GDPR"]
    assert gdpr_plan.target_source_documents == ["gdpr.pdf"]
    assert gdpr_plan.explicit_single_regulation is True
    assert hipaa_plan.target_source_documents == ["hipaa.pdf"]
    assert eu_ai_plan.target_source_documents == ["eu_ai_act.pdf"]


def test_query_plan_detects_article_9_special_category_data() -> None:
    plan = build_query_plan("When can special category health data be processed under Article 9?")

    assert "GDPR" in plan.detected_domains
    assert "article 9" in plan.detected_objects
    assert "special category data" in plan.detected_objects
    assert "health data" in plan.detected_objects
    assert "article 9" in plan.concept_groups["topic"]


def test_query_plan_detects_data_subject_rights() -> None:
    plan = build_query_plan("What rights of access, rectification, erasure, and data portability does a data subject have?")

    assert "GDPR" in plan.detected_domains
    assert "data subject" in plan.detected_actors
    assert "right of access" in plan.detected_objects
    assert "right to rectification" in plan.detected_objects
    assert "right to erasure" in plan.detected_objects
    assert "right to data portability" in plan.detected_objects
    assert "human intervention" in plan.expansion_terms
    assert "meaningful information" in plan.expansion_terms


def test_gdpr_health_data_query_routes_to_gdpr() -> None:
    plan = build_query_plan("What obligations apply when a controller processes health data?")

    assert "GDPR" in plan.detected_domains
    assert "controller" in plan.detected_actors
    assert "health data" in plan.detected_objects
    assert "gdpr" in plan.concept_groups["domain"]


def test_cross_regulation_health_ai_question_includes_gdpr_terms() -> None:
    plan = build_query_plan("What obligations apply when a mental health AI system processes patient data?")

    assert {"GDPR", "HIPAA", "EU AI Act"}.issubset(set(plan.detected_domains))
    assert "ai system" in plan.detected_objects
    assert "patient data" in plan.detected_objects
    assert "personal data" in plan.concept_groups["object"]
    assert "protected health information" in plan.concept_groups["object"]
    assert "gdpr" in plan.concept_groups["domain"]


def test_mental_health_human_oversight_routes_to_eu_ai_act_and_gdpr() -> None:
    plan = build_query_plan("What human oversight is required for AI-supported mental health diagnosis?")

    assert plan.is_mental_health_query is True
    assert "human_oversight" in plan.mental_health_intent_labels
    assert plan.explicit_regulations == []
    assert plan.inferred_regulations == ["EU AI Act", "GDPR"]
    assert plan.target_source_documents == ["eu_ai_act.pdf", "gdpr.pdf"]


def test_mental_health_screening_data_routes_to_gdpr_and_hipaa() -> None:
    plan = build_query_plan("What data protection risks arise from using mental health screening data?")

    assert "screening_data_risks" in plan.mental_health_intent_labels
    assert plan.inferred_regulations == ["GDPR", "HIPAA"]
    assert plan.target_source_documents == ["gdpr.pdf", "hipaa.pdf"]


def test_mental_health_risk_score_safeguards_routes_to_all_three() -> None:
    plan = build_query_plan("What safeguards are needed when an AI system generates mental health risk scores?")

    assert "risk_score_safeguards" in plan.mental_health_intent_labels
    assert plan.inferred_regulations == ["GDPR", "HIPAA", "EU AI Act"]
    assert plan.target_source_documents == ["gdpr.pdf", "hipaa.pdf", "eu_ai_act.pdf"]


def test_mental_health_transparency_explainability_routes_to_gdpr_and_eu_ai_act() -> None:
    plan = build_query_plan("How should a mental health diagnostic AI system handle transparency and explainability?")

    assert "transparency_explainability" in plan.mental_health_intent_labels
    assert set(plan.inferred_regulations) == {"GDPR", "EU AI Act"}
    assert set(plan.target_source_documents) == {"gdpr.pdf", "eu_ai_act.pdf"}


def test_explicit_gdpr_overrides_mental_health_cross_routing() -> None:
    plan = build_query_plan("What GDPR obligations apply when processing mental health data?")

    assert plan.is_mental_health_query is True
    assert plan.explicit_regulations == ["GDPR"]
    assert plan.inferred_regulations == []
    assert plan.target_source_documents == ["gdpr.pdf"]
    assert plan.explicit_single_regulation is True


def test_explicit_hipaa_overrides_mental_health_cross_routing() -> None:
    plan = build_query_plan("What HIPAA obligations apply to mental health patient data?")

    assert plan.explicit_regulations == ["HIPAA"]
    assert plan.inferred_regulations == []
    assert plan.target_source_documents == ["hipaa.pdf"]
    assert plan.explicit_single_regulation is True


def test_explicit_eu_ai_act_overrides_mental_health_cross_routing() -> None:
    plan = build_query_plan("What EU AI Act requirements apply to a clinical diagnostic AI system?")

    assert plan.explicit_regulations == ["EU AI Act"]
    assert plan.inferred_regulations == []
    assert plan.target_source_documents == ["eu_ai_act.pdf"]
    assert plan.explicit_single_regulation is True


def test_mental_health_query_expansion_adds_use_case_terms() -> None:
    plan = build_query_plan("What human oversight is required for AI-supported mental health diagnosis?")

    assert "mentalhealthdiagnosticsystem" in plan.mental_health_expansion_terms
    assert "humanoversightrequirement" in plan.expansion_terms
    assert "human intervention" in plan.expansion_terms


def test_mental_health_routing_does_not_treat_apms_as_regulation() -> None:
    plan = build_query_plan("How should APMS support mental health diagnostic AI safeguards?")

    assert "APMS" not in plan.inferred_regulations
    assert "apms.pdf" not in plan.target_source_documents


def test_query_plan_rewrites_follow_up_when_history_is_available() -> None:
    plan = build_query_plan(
        "What about providers?",
        ["What obligations apply under the EU AI Act for high-risk AI systems?"],
    )

    assert plan.category == "follow_up_question"
    assert plan.normalized_question == "what obligations apply to providers under the eu ai act?"
    assert plan.detected_actors == ["provider"]


def test_without_authorization_query_plan_prefers_permissions_and_exceptions() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")

    assert plan.intent in {"exceptions", "permissions"}
    assert plan.has_negated_authorization is True
    assert "Permission" in plan.preferred_statement_labels
    assert "Exception" in plan.preferred_statement_labels
    assert plan.preferred_relationships[0] == "HAS_EXCEPTION"
    assert "GRANTS_PERMISSION" in plan.preferred_relationships
    assert "REQUIRES_AUTHORIZATION" not in plan.preferred_relationships[:3]
    assert "required by law" in plan.expansion_terms
    assert "may disclose" in plan.expansion_terms


def test_without_authorization_ranking_prefers_permission_exception_rows() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")
    rows = [
        {
            "statement_name": "Authorization Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "A covered entity must obtain authorization for this disclosure.",
            "score": 30,
        },
        {
            "statement_name": "Disclosure Prohibition",
            "statement_labels": ["Statement", "Prohibition"],
            "relationship": "LIMITED_BY",
            "evidence_text": "A covered entity may not disclose protected health information.",
            "score": 28,
        },
        {
            "statement_name": "Permitted Disclosure Required By Law",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose protected health information without authorization when required by law.",
            "score": 10,
        },
        {
            "statement_name": "Authorization Exception Public Health",
            "statement_labels": ["Statement", "Exception"],
            "relationship": "HAS_EXCEPTION",
            "evidence_text": "Authorization or opportunity to agree or object is not required for public health activities.",
            "score": 10,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] in {
        "Permitted Disclosure Required By Law",
        "Authorization Exception Public Health",
    }
    assert ranked[1]["statement_name"] in {
        "Permitted Disclosure Required By Law",
        "Authorization Exception Public Health",
    }
    assert ranked[-1]["statement_name"] in {"Authorization Requirement", "Disclosure Prohibition"}


def test_without_authorization_downranks_with_authorization_rows() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")
    rows = [
        {
            "statement_name": "Permitted Disclosure with Authorization",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI pursuant to and in compliance with an authorization.",
            "score": 80,
        },
        {
            "statement_name": "Permitted Disclosure Public Health",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI without authorization for public health activities.",
            "score": 10,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Permitted Disclosure Public Health"
    assert ranked[-1]["statement_name"] == "Permitted Disclosure with Authorization"


def test_build_context_deduplicates_and_respects_max_chars() -> None:
    rows = [
        {
            "statement_name": "Disclose PHI",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "HAS_EXCEPTION",
            "related_name": "Authorization Exception",
            "evidence_text": "Covered entities may disclose PHI without authorization for treatment.",
            "citation": "45 CFR 164.506",
            "source_document": "hipaa.pdf",
        },
        {
            "statement_name": "Disclose PHI",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "HAS_EXCEPTION",
            "related_name": "Authorization Exception",
            "evidence_text": "Covered entities may disclose PHI without authorization for treatment.",
            "citation": "45 CFR 164.506",
            "source_document": "hipaa.pdf",
        },
        {
            "statement_name": "Very long item",
            "evidence_text": "x" * 500,
        },
    ]

    context = build_context(rows, max_chars=260)

    assert context.startswith("[1]\n")
    assert "Covered entities may disclose PHI" in context
    assert "Very long item" not in context
    assert len(context) <= 260


def test_context_builder_prefers_permitted_disclosure_before_authorization_requirements() -> None:
    rows = [
        {
            "statement_name": "Authorization Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "A covered entity must obtain authorization for uses and disclosures.",
            "score": 100,
        },
        {
            "statement_name": "Permitted Disclosure Required By Law",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI without authorization when required by law.",
            "score": 20,
        },
    ]

    context = build_context(rows)

    assert context.index("Permitted Disclosure Required By Law") < context.index("Authorization Requirement")


def test_context_builder_groups_treatment_payment_operations_rows() -> None:
    rows = [
        {
            "statement_name": "Treatment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for treatment without authorization.",
            "score": 30,
        },
        {
            "statement_name": "Payment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for payment without authorization.",
            "score": 29,
        },
        {
            "statement_name": "Health Care Operations Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for health care operations without authorization.",
            "score": 28,
        },
        {
            "statement_name": "OHCA Disclosure",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "An organized health care arrangement may use or disclose PHI for operations.",
            "score": 27,
        },
    ]

    context = build_context(rows)

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "Statement: Treatment Permission" not in context
    assert "Statement: Payment Permission" not in context
    assert "Statement: Health Care Operations Permission" not in context
    assert "- OHCA Disclosure" in context
    assert "Evidence References:" in context


def test_context_builder_groups_named_tpo_statement_variants() -> None:
    rows = [
        {
            "statement_name": "PHI Disclosure for Health Care Operations",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "Covered entities may disclose PHI for health care operations.",
            "source_document": "hipaa.pdf",
            "score": 50,
        },
        {
            "statement_name": "PHI Disclosure for Payment Activities",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "Covered entities may disclose PHI for payment activities.",
            "source_document": "hipaa.pdf",
            "score": 49,
        },
        {
            "statement_name": "PHI Disclosure for Treatment Activities",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "Covered entities may disclose PHI for treatment activities.",
            "source_document": "hipaa.pdf",
            "score": 48,
        },
        {
            "statement_name": "PHI Disclosure within Organized Health Care Arrangement",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "Participants in an organized health care arrangement may disclose PHI for operations.",
            "source_document": "hipaa.pdf",
            "score": 47,
        },
    ]

    context = build_context(rows)

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "Grouped statements:" in context
    assert "- PHI Disclosure for Health Care Operations" in context
    assert "- PHI Disclosure for Payment Activities" in context
    assert "- PHI Disclosure for Treatment Activities" in context
    assert "- PHI Disclosure within Organized Health Care Arrangement" in context
    assert "Statement: PHI Disclosure for Health Care Operations" not in context
    assert "Evidence References: [1], [2], [3], [4], Treatment, Payment, and Health Care Operations, hipaa.pdf" in context


def test_tpo_group_suppresses_individual_top_level_subcategories() -> None:
    rows = [
        {
            "statement_name": "Treatment, Payment, and Health Care Operations",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may use or disclose PHI for treatment, payment, or health care operations.",
            "score": 60,
        },
        {
            "statement_name": "PHI Disclosure for Health Care Operations",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may disclose PHI for health care operations.",
            "score": 59,
        },
    ]

    context = build_context(rows)

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "Statement: PHI Disclosure for Health Care Operations" not in context
    assert "- PHI Disclosure for Health Care Operations" in context


def test_context_builder_keeps_non_tpo_permission_categories_separate() -> None:
    rows = [
        {
            "statement_name": "Permitted Disclosure Required by Law",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may disclose PHI when required by law.",
            "score": 50,
        },
        {
            "statement_name": "Permitted Disclosure for Disaster Relief",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may disclose PHI for disaster relief purposes.",
            "score": 49,
        },
        {
            "statement_name": "Public Health Activities",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may disclose PHI for public health activities.",
            "score": 48,
        },
        {
            "statement_name": "Whistleblower Disclosure Exception",
            "statement_labels": ["Statement", "Exception"],
            "evidence_text": "A workforce member may disclose PHI for whistleblower activity under conditions.",
            "score": 47,
        },
        {
            "statement_name": "Law Enforcement Disclosure",
            "statement_labels": ["Statement", "Permission"],
            "evidence_text": "A covered entity may disclose PHI for law enforcement purposes.",
            "score": 46,
        },
    ]

    context = build_context(rows)

    assert "Treatment, Payment, and Health Care Operations" not in context
    for row in rows:
        assert f"Statement: {row['statement_name']}" in context


def test_semantic_group_excludes_research_related_treatment_authorization() -> None:
    row = {
        "statement_name": "Research-Related Treatment Authorization",
        "statement_labels": ["Statement", "Requirement"],
        "evidence_text": "Authorization for research-related treatment may be required.",
    }

    assert get_semantic_group(row) is None
    context = build_context([row])
    assert "Treatment, Payment, and Health Care Operations" not in context
    assert "Statement: Research-Related Treatment Authorization" in context


def test_context_builder_suppresses_employment_record_rows() -> None:
    employment_row = {
        "statement_name": "Employment Records Exclusion",
        "statement_labels": ["Statement", "Definition"],
        "evidence_text": "Employment records held by a covered entity in its role as employer are excluded.",
        "score": 100,
    }
    disclosure_row = {
        "statement_name": "Permitted Disclosure Required by Law",
        "statement_labels": ["Statement", "Permission"],
        "evidence_text": "A covered entity may disclose PHI when required by law.",
        "score": 20,
    }

    assert is_suppressed_context_row(employment_row) is True
    context = build_context([employment_row, disclosure_row])

    assert "Employment Records Exclusion" not in context
    assert "Permitted Disclosure Required by Law" in context


def test_phi_disclosure_ranking_downranks_employment_record_rows() -> None:
    plan = build_query_plan("When can a covered entity disclose PHI without authorization?")
    rows = [
        {
            "statement_name": "Employment Records Exclusion",
            "statement_labels": ["Statement", "Definition"],
            "relationship": "DEFINES",
            "evidence_text": "Employment records held by a covered entity in its role as employer are not PHI.",
            "score": 100,
        },
        {
            "statement_name": "Permitted Disclosure Required by Law",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may disclose PHI without authorization when required by law.",
            "score": 20,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Permitted Disclosure Required by Law"
    assert ranked[-1]["statement_name"] == "Employment Records Exclusion"


def test_high_risk_ai_provider_query_ranks_eu_ai_act_above_hipaa_provider() -> None:
    plan = build_query_plan("What obligations apply to providers of high-risk AI systems?")
    rows = [
        {
            "statement_name": "Health Care Provider Definition",
            "statement_labels": ["Statement", "Definition"],
            "evidence_text": "Health Care Provider means a provider of services.",
            "source_document": "hipaa.pdf",
            "score": 0,
        },
        {
            "statement_name": "Provider Compliance",
            "statement_labels": ["Statement", "Obligation"],
            "relationship": "HAS_REQUIREMENT",
            "evidence_text": "Providers of high-risk AI systems should ensure compliance with this Regulation.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Provider Compliance"
    assert ranked[0]["source_group"] == "eu_ai_act"
    assert ranked[-1]["statement_name"] == "Health Care Provider Definition"
    assert "hipaa_noise" in ranked[-1]["penalties"]
    assert "missing_ai_system" in ranked[-1]["penalties"]


def test_high_risk_ai_provider_query_works_without_source_metadata() -> None:
    plan = build_query_plan("What obligations apply to providers of high-risk AI systems?")
    rows = [
        {
            "statement_name": "Health Care Provider Definition",
            "statement_labels": ["Statement", "Definition"],
            "evidence_text": "Health Care Provider means a provider of services.",
            "source_document": "hipaa.pdf",
            "score": 0,
        },
        {
            "statement_name": "Provider Compliance",
            "statement_labels": ["Statement", "Obligation"],
            "relationship": "HAS_REQUIREMENT",
            "evidence_text": "Providers of high-risk AI systems should ensure compliance with this Regulation.",
            "source_document": None,
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Provider Compliance"
    assert ranked[0]["source_group"] == "eu_ai_act"


def test_ambiguous_provider_query_keeps_multiple_source_candidates() -> None:
    plan = build_query_plan("What obligations apply to providers?")
    rows = [
        {
            "statement_name": "Health Care Provider Safeguards",
            "statement_labels": ["Statement", "Obligation"],
            "evidence_text": "Health care providers must implement safeguards under HIPAA.",
            "source_document": "hipaa.pdf",
            "score": 0,
        },
        {
            "statement_name": "Provider Compliance",
            "statement_labels": ["Statement", "Obligation"],
            "evidence_text": "Providers of high-risk AI systems should ensure compliance with this Regulation.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert "provider" in plan.ambiguous_terms
    assert {row["source_group"] for row in ranked} == {"hipaa", "eu_ai_act"}


def test_hipaa_specific_provider_query_ranks_hipaa_above_ai_act() -> None:
    plan = build_query_plan("What must health care providers do under HIPAA?")
    rows = [
        {
            "statement_name": "Health Care Provider Safeguards",
            "statement_labels": ["Statement", "Obligation"],
            "evidence_text": "Health care providers must implement safeguards under HIPAA.",
            "source_document": "hipaa.pdf",
            "score": 0,
        },
        {
            "statement_name": "Provider Compliance",
            "statement_labels": ["Statement", "Obligation"],
            "evidence_text": "Providers of high-risk AI systems should ensure compliance with this Regulation.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Health Care Provider Safeguards"


def test_high_risk_ai_documentation_query_prioritizes_documentation_related_rows() -> None:
    plan = build_query_plan("What documentation is required for high-risk AI systems?")
    rows = [
        {
            "statement_name": "Technical Documentation",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_DOCUMENTATION",
            "evidence_text": "Providers of high-risk AI systems must prepare technical documentation and instructions for use.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
        {
            "statement_name": "Health Care Provider Definition",
            "statement_labels": ["Statement", "Definition"],
            "evidence_text": "Health Care Provider means a provider of services.",
            "source_document": "hipaa.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Technical Documentation"
    assert "technical documentation" in ranked[0]["matched_terms"]


def test_risk_management_query_expands_entity_seed_before_hipaa_evidence() -> None:
    client = FakeAiRiskExpansionClient()
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    rows = retriever.retrieve("What are the risk management requirements for high-risk AI systems?", limit=5)
    context = build_context(rows)

    assert rows[0]["statement_name"] == "Risk Management System Requirement"
    assert rows[0]["source_group"] == "eu_ai_act"
    assert "Minimum Necessary Requirement" not in [row["statement_name"] for row in rows]
    assert "Risk Management System Requirement" in context
    assert "Minimum Necessary Requirement" not in context
    assert retriever.last_expansion_debug["seeds"]
    assert retriever.last_expansion_debug["expanded"]
    assert retriever.last_expansion_debug["excluded"]


def test_context_builder_omits_missing_evidence_entity_when_evidence_exists() -> None:
    rows = [
        {
            "seed_name": "High-risk AI System",
            "seed_labels": ["Entity"],
            "statement_name": "Unknown",
            "statement_labels": [],
            "score": 80,
        },
        {
            "statement_name": "Risk Management System Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "evidence_text": "Providers of high-risk AI systems must establish a risk management system.",
            "source_document": "eu_ai_act.pdf",
            "score": 60,
        },
    ]

    context = build_context(rows)

    assert "Risk Management System Requirement" in context
    assert "Statement: Unknown" not in context
    assert "High-risk AI System" not in context


def test_risk_management_query_downranks_requirement_only_without_ai_object() -> None:
    plan = build_query_plan("What are the risk management requirements for high-risk AI systems?")
    rows = [
        {
            "statement_name": "Generic Risk Requirement",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "HAS_REQUIREMENT",
            "evidence_text": "Organizations must maintain risk controls and requirements.",
            "score": 20,
        },
        {
            "statement_name": "High-risk AI Risk Management",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "MITIGATES_RISK",
            "evidence_text": "High-risk AI systems must use a risk management system to mitigate known and foreseeable risks.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "High-risk AI Risk Management"
    assert "missing_ai_system" in ranked[-1]["penalties"]


def test_risk_management_ocr_normalization_matches_ai_and_topic_terms() -> None:
    plan = build_query_plan("What are the risk management requirements for high-risk AI systems?")
    rows = [
        {
            "statement_name": "OCR Risk Management",
            "statement_labels": ["Statement", "Requirement"],
            "evidence_text": "Providers of high-r isk AI syste ms must implement risk-manag ement and safegua rds.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
        {
            "statement_name": "HIPAA Safeguards",
            "statement_labels": ["Statement", "Requirement"],
            "evidence_text": "Covered entities must implement safeguards for protected health information.",
            "source_document": "hipaa.pdf",
            "score": 15,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "OCR Risk Management"
    assert "object:high-risk ai system" in ranked[0]["matched_concept_groups"]
    assert any(item.startswith("topic:") for item in ranked[0]["matched_concept_groups"])


def test_debug_metadata_includes_concepts_terms_penalties_and_source() -> None:
    plan = build_query_plan("What obligations apply to providers of high-risk AI systems?")
    rows = [
        {
            "statement_name": "Health Care Provider Definition",
            "statement_labels": ["Statement", "Definition"],
            "evidence_text": "Health Care Provider means a provider of services.",
            "source_document": "hipaa.pdf",
            "score": 0,
        }
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["source_group"] == "hipaa"
    assert "actor:provider" in ranked[0]["matched_concept_groups"]
    assert "generic_provider_only" in ranked[0]["penalties"]


def test_concrete_ai_obligation_outranks_broad_preamble_row() -> None:
    plan = build_query_plan(
        "What concrete obligations do providers have for high-risk AI systems, including documentation, "
        "risk management, conformity assessment, monitoring, and cybersecurity?"
    )
    rows = [
        {
            "statement_name": "High-Risk AI System Rules",
            "statement_labels": ["Statement", "Requirement"],
            "evidence_text": "Common rules for high-risk AI systems should be established.",
            "source_document": "eu_ai_act.pdf",
            "score": 40,
        },
        {
            "statement_name": "Risk Management System Process",
            "statement_labels": ["Statement", "Requirement"],
            "evidence_text": "The risk-management system should consist of a continuous, iterative process throughout the lifecycle of a high-risk AI system.",
            "source_document": "eu_ai_act.pdf",
            "score": 0,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "Risk Management System Process"
    assert "concrete_obligation" in ranked[0]["boosts"]
    assert "risk_management" in ranked[0]["matched_topic_groups"]
    assert "broad_preamble" in ranked[-1]["penalties"]


def test_concrete_ai_topic_boosts_cover_documentation_conformity_and_cybersecurity() -> None:
    plan = build_query_plan(
        "What concrete obligations apply to high-risk AI systems for documentation, conformity assessment, and cybersecurity?"
    )
    rows = [
        {
            "statement_name": "Technical Documentation",
            "evidence_text": "Technical documentation must describe testing, validation, and the risk-management system for the high-risk AI system.",
            "source_document": "eu_ai_act.pdf",
        },
        {
            "statement_name": "Conformity Assessment Prior to Market Placement",
            "evidence_text": "A high-risk AI system is subject to conformity assessment prior to market placement.",
            "source_document": "eu_ai_act.pdf",
        },
        {
            "statement_name": "Cybersecurity Measures",
            "evidence_text": "The high-risk AI system requires cybersecurity measures against data poisoning and adversarial attacks.",
            "source_document": "eu_ai_act.pdf",
        },
        {
            "statement_name": "High-Risk AI System Rules",
            "evidence_text": "Common rules for high-risk AI systems should be established.",
            "source_document": "eu_ai_act.pdf",
            "score": 50,
        },
    ]

    ranked = _dedupe_rows(rows, plan)

    assert {row["statement_name"] for row in ranked[:3]} == {
        "Technical Documentation",
        "Conformity Assessment Prior to Market Placement",
        "Cybersecurity Measures",
    }
    assert all("concrete_obligation" in row["boosts"] for row in ranked[:3])


def test_final_selection_excludes_broad_ai_rows_when_enough_concrete_rows_exist() -> None:
    plan = build_query_plan("What concrete obligations apply to providers of high-risk AI systems?")
    concrete_rows = [
        {
            "statement_name": f"Concrete Duty {index}",
            "evidence_text": f"Providers of high-risk AI systems must maintain a risk management system duty {index}.",
            "source_document": "eu_ai_act.pdf",
        }
        for index in range(6)
    ]
    broad_row = {
        "statement_name": "High-Risk AI System Rules",
        "evidence_text": "Common rules for high-risk AI systems should be established.",
        "source_document": "eu_ai_act.pdf",
    }
    ranked = _dedupe_rows([broad_row, *concrete_rows], plan)

    selected, excluded = _select_final_rows(ranked, plan, 10)

    assert broad_row["statement_name"] not in [row["statement_name"] for row in selected]
    assert "broad_row_excluded_concrete_available" in excluded[0]["penalties"]


def test_final_selection_keeps_concrete_eu_ai_row_without_repeated_ai_object_text() -> None:
    plan = build_query_plan("What concrete obligations apply to providers of high-risk AI systems?")
    rows = [
        {
            "statement_name": "Technical Documentation Update",
            "evidence_text": "Providers shall keep the technical documentation up to date.",
            "source_document": "eu_ai_act.pdf",
        },
        {
            "statement_name": "Risk Management System Process",
            "evidence_text": "A continuous risk management process applies throughout the lifecycle of a high-risk AI system.",
            "source_document": "eu_ai_act.pdf",
        },
    ]
    ranked = _dedupe_rows(rows, plan)

    selected, excluded = _select_final_rows(ranked, plan, 10)

    assert "Technical Documentation Update" in [row["statement_name"] for row in selected]
    assert "Technical Documentation Update" not in [row["statement_name"] for row in excluded]


def test_context_builder_groups_concrete_ai_obligations_by_duty_area() -> None:
    rows = [
        {"statement_name": "Risk Management System Process", "evidence_text": "A continuous risk management process applies throughout the lifecycle of a high-risk AI system.", "source_document": "eu_ai_act.pdf", "score": 60},
        {"statement_name": "Technical Documentation", "evidence_text": "Providers must keep technical documentation for the high-risk AI system.", "source_document": "eu_ai_act.pdf", "score": 59},
        {"statement_name": "Human Oversight Design", "evidence_text": "The high-risk AI system must support human oversight by natural persons.", "source_document": "eu_ai_act.pdf", "score": 58},
        {"statement_name": "Cybersecurity Measures", "evidence_text": "The high-risk AI system requires cybersecurity and robustness measures.", "source_document": "eu_ai_act.pdf", "score": 57},
        {"statement_name": "Conformity Assessment", "evidence_text": "The high-risk AI system requires conformity assessment prior to market placement.", "source_document": "eu_ai_act.pdf", "score": 56},
        {"statement_name": "Establish Post-Market Monitoring System", "evidence_text": "Providers must establish post-market monitoring for the high-risk AI system.", "source_document": "eu_ai_act.pdf", "score": 55},
    ]

    context = build_context(rows)

    expected_groups = [
        "Risk management system",
        "Technical documentation and record keeping",
        "Human oversight",
        "Cybersecurity, robustness, and resilience",
        "Conformity assessment",
        "Quality management and post-market monitoring",
    ]
    for group in expected_groups:
        assert f"Statement: {group}" in context
    assert [context.index(f"Statement: {group}") for group in expected_groups] == sorted(
        context.index(f"Statement: {group}") for group in expected_groups
    )


def test_hipaa_safeguards_query_keeps_hipaa_rows_above_ai_rows() -> None:
    plan = build_query_plan("What safeguards or policies are required under HIPAA?")
    rows = [
        {"statement_name": "HIPAA Safeguards and Policies", "evidence_text": "Covered entities must implement safeguards, policies, procedures, and training under HIPAA.", "source_document": "hipaa.pdf"},
        {"statement_name": "Cybersecurity Measures", "evidence_text": "High-risk AI systems require cybersecurity safeguards.", "source_document": "eu_ai_act.pdf"},
    ]

    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] == "HIPAA Safeguards and Policies"
    assert "hipaa_noise" not in ranked[0]["penalties"]


def test_gdpr_only_question_prefers_gdpr_source_document() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {"statement_name": "Right of Access", "evidence_text": "The data subject has the right of access to personal data.", "source_document": "gdpr.pdf"},
        {"statement_name": "Right to Rectification", "evidence_text": "The data subject has the right to rectification.", "source_document": "gdpr.pdf"},
        {"statement_name": "Right to Erasure", "evidence_text": "The data subject has the right to erasure.", "source_document": "gdpr.pdf"},
        {"statement_name": "EU AI Act GDPR Reference", "evidence_text": "This Regulation references GDPR data subject rights.", "source_document": "eu_ai_act.pdf"},
        {"statement_name": "HIPAA Individual Access", "evidence_text": "HIPAA gives individuals access to PHI.", "source_document": "hipaa.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, excluded = _select_final_rows(ranked, plan, 5)

    assert selected
    assert {row["source_document"] for row in selected} == {"gdpr.pdf"}
    assert selected[0]["source_document"] == "gdpr.pdf"
    assert any("wrong_source_for_explicit_regulation" in row["penalties"] for row in excluded)


def test_hipaa_only_question_prefers_hipaa_source_document() -> None:
    plan = build_query_plan("What safeguards are required under HIPAA?")
    rows = [
        {"statement_name": "HIPAA Safeguards", "evidence_text": "Covered entities must implement safeguards.", "source_document": "hipaa.pdf"},
        {"statement_name": "HIPAA Policies", "evidence_text": "Covered entities must maintain policies and procedures.", "source_document": "hipaa.pdf"},
        {"statement_name": "HIPAA Training", "evidence_text": "Covered entities must train workforce members.", "source_document": "hipaa.pdf"},
        {"statement_name": "AI Cybersecurity", "evidence_text": "High-risk AI systems require cybersecurity safeguards.", "source_document": "eu_ai_act.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    assert {row["source_document"] for row in selected} == {"hipaa.pdf"}


def test_eu_ai_act_only_question_prefers_eu_ai_act_source_document() -> None:
    plan = build_query_plan("What requirements apply under the EU AI Act?")
    rows = [
        {"statement_name": "Risk Management", "evidence_text": "High-risk AI systems require a risk management system.", "source_document": "eu_ai_act.pdf"},
        {"statement_name": "Human Oversight", "evidence_text": "High-risk AI systems require human oversight.", "source_document": "eu_ai_act.pdf"},
        {"statement_name": "Technical Documentation", "evidence_text": "Providers must keep technical documentation.", "source_document": "eu_ai_act.pdf"},
        {"statement_name": "GDPR DPIA", "evidence_text": "Controllers may need a data protection impact assessment.", "source_document": "gdpr.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    assert {row["source_document"] for row in selected} == {"eu_ai_act.pdf"}


def test_gdpr_data_subject_rights_uses_rights_expansion_terms() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {"statement_name": "Generic Personal Data", "evidence_text": "Personal data relates to a data subject.", "source_document": "gdpr.pdf", "score": 10},
        {"statement_name": "Right to Data Portability", "evidence_text": "The data subject has the right to data portability.", "source_document": "gdpr.pdf", "score": 0},
        {"statement_name": "Automated Decision-Making", "evidence_text": "The data subject may obtain human intervention and meaningful information.", "source_document": "gdpr.pdf", "score": 0},
    ]
    ranked = _dedupe_rows(rows, plan)

    assert ranked[0]["statement_name"] in {"Right to Data Portability", "Automated Decision-Making"}
    assert any(term in ranked[0]["matched_terms"] for term in ("right to data portability", "human intervention", "meaningful information"))


def test_exact_entity_matches_are_preserved_as_seeds_when_missing_evidence() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {
            "seed_id": 101,
            "seed_name": "Right To Data Portability",
            "seed_labels": ["Entity"],
            "statement_name": "Right To Data Portability",
            "statement_labels": ["Entity"],
            "score": 25,
        }
    ]
    ranked = _dedupe_rows(rows, plan)

    assert _is_evidence_expansion_seed(ranked[0], plan) is True
    assert "missing_evidence" in ranked[0]["penalties"]


def test_entity_seed_expands_to_connected_evidence_statement() -> None:
    client = FakeEntitySeedExpansionClient()
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    rows = retriever.retrieve("What rights does a data subject have under GDPR?", limit=5)

    assert rows[0]["statement_name"] == "Right to Data Portability"
    assert rows[0]["source_document"] == "gdpr.pdf"
    assert rows[0]["expanded_from_entity_seed"] is True
    assert "expanded_from_entity_seed" in rows[0]["boosts"]
    assert retriever.last_expansion_debug["seeds"]
    assert retriever.last_expansion_debug["expanded"][0]["expanded_from_entity_seed"] is True


def test_entity_seed_expansion_respects_explicit_source_document() -> None:
    client = FakeEntitySeedExpansionClient()
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    retriever.retrieve("What rights does a data subject have under GDPR?", limit=5)
    expansion_params = [params for query, params in client.calls if "MATCH path = (seed)-[*1..2]-(statement)" in query][0]

    assert expansion_params["target_source_documents"] == ["gdpr.pdf"]


def test_gdpr_rights_question_expands_to_specific_rights_terms() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")

    for term in [
        "right of access",
        "right to rectification",
        "right to erasure",
        "restriction of processing",
        "right to data portability",
        "right to object",
        "profiling",
        "human intervention",
        "meaningful information",
        "legal effects",
    ]:
        assert term in plan.expansion_terms


def test_gdpr_rights_question_selects_access_erasure_portability_object_when_available() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {"statement_name": "Generic Data Subject Information", "evidence_text": "The controller should inform the data subject.", "source_document": "gdpr.pdf", "score": 200},
        {"statement_name": "Right of Access", "evidence_text": "A data subject has the right of access.", "source_document": "gdpr.pdf", "score": 1, "expanded_from_entity_seed": True, "expanded_from_seed": "Data Subject Access Right"},
        {"statement_name": "Right to Erasure", "evidence_text": "The data subject has the right to erasure.", "source_document": "gdpr.pdf", "score": 1, "expanded_from_entity_seed": True, "expanded_from_seed": "Right To Erasure"},
        {"statement_name": "Right to Data Portability", "evidence_text": "The data subject may transmit those data to another controller.", "source_document": "gdpr.pdf", "score": 1, "expanded_from_entity_seed": True, "expanded_from_seed": "Right To Data Portability"},
        {"statement_name": "Right to Object", "evidence_text": "The data subject has the right to object to processing.", "source_document": "gdpr.pdf", "score": 1, "expanded_from_entity_seed": True, "expanded_from_seed": "Right To Object"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    selected_names = [row["statement_name"] for row in selected[:4]]
    assert selected_names == ["Right of Access", "Right to Erasure", "Right to Data Portability", "Right to Object"]


def test_wrong_source_entity_seed_does_not_reintroduce_eu_ai_act_for_gdpr_question() -> None:
    client = FakeEntitySeedExpansionClient(seed_source_document="eu_ai_act.pdf")
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    rows = retriever.retrieve("What rights does a data subject have under GDPR?", limit=5)

    assert rows == []
    assert not any("MATCH path = (seed)-[*1..2]-(statement)" in query for query, _ in client.calls)


def test_context_includes_expanded_evidence_rows_from_entity_seeds() -> None:
    row = {
        "statement_name": "Right to Data Portability",
        "evidence_text": "The data subject should have personal data transmitted directly from one controller to another.",
        "source_document": "gdpr.pdf",
        "expanded_from_entity_seed": True,
        "_retriever_ranked": True,
    }

    context = build_context([row])

    assert "Right to Data Portability" in context
    assert "transmitted directly" in context


def test_diversity_prevents_only_generic_data_subject_rows() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {"statement_name": f"Generic Data Subject Row {index}", "evidence_text": "The controller should provide information to the data subject.", "source_document": "gdpr.pdf", "score": 100 - index}
        for index in range(5)
    ]
    rows.extend(
        [
            {"statement_name": "Right of Access", "evidence_text": "The data subject has the right of access.", "source_document": "gdpr.pdf", "score": 1},
            {"statement_name": "Right to Rectification", "evidence_text": "The data subject may rectify inaccurate personal data.", "source_document": "gdpr.pdf", "score": 1},
            {"statement_name": "Right to Erasure", "evidence_text": "The data subject has the right to erasure.", "source_document": "gdpr.pdf", "score": 1},
        ]
    )
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    assert {"Right of Access", "Right to Rectification", "Right to Erasure"}.issubset(
        {row["statement_name"] for row in selected}
    )


def test_gdpr_article_9_question_prefers_gdpr_over_eu_ai_act_references() -> None:
    plan = build_query_plan("When can special category health data be processed under GDPR Article 9?")
    rows = [
        {"statement_name": "Article 9 Explicit Consent", "evidence_text": "Article 9 permits processing special category health data with explicit consent.", "source_document": "gdpr.pdf"},
        {"statement_name": "Article 9 Health Care", "evidence_text": "Article 9 permits processing health data for health care conditions.", "source_document": "gdpr.pdf"},
        {"statement_name": "Article 9 Public Interest", "evidence_text": "Article 9 includes public interest processing conditions.", "source_document": "gdpr.pdf"},
        {"statement_name": "EU AI GDPR Reference", "evidence_text": "The EU AI Act references GDPR Article 9 and health data.", "source_document": "eu_ai_act.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    assert {row["source_document"] for row in selected} == {"gdpr.pdf"}


def test_cross_regulation_question_retrieves_all_requested_sources() -> None:
    plan = build_query_plan("How do GDPR, HIPAA, and the EU AI Act apply to a mental health diagnostic AI system?")
    rows = [
        {"statement_name": "GDPR Health Data", "evidence_text": "GDPR applies to health data.", "source_document": "gdpr.pdf"},
        {"statement_name": "HIPAA PHI", "evidence_text": "HIPAA applies to protected health information.", "source_document": "hipaa.pdf"},
        {"statement_name": "EU AI High Risk", "evidence_text": "The EU AI Act applies to high-risk AI systems.", "source_document": "eu_ai_act.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 10)

    assert {"gdpr.pdf", "hipaa.pdf", "eu_ai_act.pdf"}.issubset({row["source_document"] for row in selected})


def test_cross_regulation_retrieval_selects_at_least_one_row_per_target_source() -> None:
    plan = build_query_plan("What safeguards are needed when an AI system generates mental health risk scores?")
    rows = [
        {"statement_name": "EU AI Risk Management", "evidence_text": "High-risk AI systems require risk management safeguards.", "source_document": "eu_ai_act.pdf", "score": 100},
        {"statement_name": "EU AI Human Oversight", "evidence_text": "High-risk AI systems require human oversight.", "source_document": "eu_ai_act.pdf", "score": 95},
        {"statement_name": "GDPR Security", "evidence_text": "GDPR requires security for special category health data.", "source_document": "gdpr.pdf", "score": 20},
        {"statement_name": "HIPAA Safeguards", "evidence_text": "HIPAA requires safeguards for protected health information.", "source_document": "hipaa.pdf", "score": 10},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 3)

    assert {row["source_document"] for row in selected} == {"gdpr.pdf", "hipaa.pdf", "eu_ai_act.pdf"}


def test_mental_health_gdpr_question_prefers_gdpr_source() -> None:
    plan = build_query_plan("What GDPR obligations apply when processing mental health data?")
    rows = [
        {"statement_name": "GDPR Special Category Data", "evidence_text": "Mental health data is health data and special category data.", "source_document": "gdpr.pdf"},
        {"statement_name": "GDPR DPIA", "evidence_text": "Controllers may need a data protection impact assessment.", "source_document": "gdpr.pdf"},
        {"statement_name": "GDPR Controller Duties", "evidence_text": "Controllers must comply with GDPR processing principles.", "source_document": "gdpr.pdf"},
        {"statement_name": "HIPAA Mental Health PHI", "evidence_text": "HIPAA protects mental health patient information.", "source_document": "hipaa.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 5)

    assert {row["source_document"] for row in selected} == {"gdpr.pdf"}


def test_mental_health_cross_regulation_question_includes_gdpr_hipaa_eu_ai_act() -> None:
    plan = build_query_plan("What obligations apply to a mental health diagnostic AI system processing patient data?")
    rows = [
        {"statement_name": "GDPR Controller Duties", "evidence_text": "Controllers process health data under GDPR.", "source_document": "gdpr.pdf"},
        {"statement_name": "HIPAA Covered Entity Duties", "evidence_text": "Covered entities protect patient PHI under HIPAA.", "source_document": "hipaa.pdf"},
        {"statement_name": "EU AI Human Oversight", "evidence_text": "High-risk AI systems require human oversight.", "source_document": "eu_ai_act.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 10)

    assert plan.cross_regulation is True
    assert {"gdpr.pdf", "hipaa.pdf", "eu_ai_act.pdf"}.issubset({row["source_document"] for row in selected})


def test_context_rows_match_debug_selected_rows() -> None:
    rows = [
        {"statement_name": "Right of Access", "evidence_text": "Access evidence.", "source_document": "gdpr.pdf", "score": 90},
        {"statement_name": "Right to Erasure", "evidence_text": "Erasure evidence.", "source_document": "gdpr.pdf", "score": 80},
    ]

    assert context_row_ids(rows) == ["gdpr.pdf|Right of Access|", "gdpr.pdf|Right to Erasure|"]


def test_secondary_references_do_not_override_primary_source() -> None:
    plan = build_query_plan("What rights does a data subject have under GDPR?")
    rows = [
        {"statement_name": "EU AI Act Exercise of Data Subject Rights", "evidence_text": "This Regulation refers to GDPR data subject rights.", "source_document": "eu_ai_act.pdf", "score": 100},
        {"statement_name": "Right of Access", "evidence_text": "The data subject has the right of access.", "source_document": "gdpr.pdf", "score": 0},
        {"statement_name": "Right to Rectification", "evidence_text": "The data subject has the right to rectification.", "source_document": "gdpr.pdf", "score": 0},
        {"statement_name": "Right to Object", "evidence_text": "The data subject has the right to object.", "source_document": "gdpr.pdf", "score": 0},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 4)

    assert selected[0]["source_document"] == "gdpr.pdf"
    assert "secondary_reference_for_explicit_regulation" in next(row for row in ranked if row["source_document"] == "eu_ai_act.pdf")["penalties"]


def test_cross_regulation_query_keeps_hipaa_and_eu_ai_act_rows() -> None:
    plan = build_query_plan("Compare HIPAA and the EU AI Act on safeguards for sensitive data.")
    rows = [
        {"statement_name": "HIPAA Safeguards", "evidence_text": "Covered entities must implement safeguards for protected health information.", "source_document": "hipaa.pdf"},
        {"statement_name": "AI Cybersecurity Measures", "evidence_text": "High-risk AI systems require cybersecurity safeguards and robustness.", "source_document": "eu_ai_act.pdf"},
    ]
    ranked = _dedupe_rows(rows, plan)
    selected, _ = _select_final_rows(ranked, plan, 10)

    assert {row["source_group"] for row in selected} == {"hipaa", "eu_ai_act"}
    hipaa_row = next(row for row in selected if row["source_group"] == "hipaa")
    assert "unrelated_hipaa_for_ai_query" not in hipaa_row["penalties"]


def test_context_builder_does_not_group_unrelated_rows_into_tpo() -> None:
    rows = [
        {
            "statement_name": "Treatment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for treatment without authorization.",
            "score": 40,
        },
        {
            "statement_name": "Payment Permission",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "A covered entity may use or disclose PHI for payment without authorization.",
            "score": 39,
        },
        {
            "statement_name": "Emergency Disclosure - Directory",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "GRANTS_PERMISSION",
            "evidence_text": "Emergency directory disclosure may be made when the individual has an opportunity to agree or object.",
            "score": 38,
        },
        {
            "statement_name": "Business Associate Disclosure",
            "statement_labels": ["Statement", "Permission"],
            "relationship": "DISCLOSES_TO",
            "evidence_text": "A covered entity may disclose PHI to a business associate under a business associate agreement.",
            "score": 37,
        },
        {
            "statement_name": "Authorization for Research",
            "statement_labels": ["Statement", "Requirement"],
            "relationship": "REQUIRES_AUTHORIZATION",
            "evidence_text": "Authorization for research-related treatment may be required unless an exception applies.",
            "score": 36,
        },
    ]

    context = build_context(rows)
    tpo_block = context.split("[2]")[0]

    assert context.count("Statement: Treatment, Payment, and Health Care Operations") == 1
    assert "A covered entity may use or disclose PHI for treatment without authorization." in tpo_block
    assert "A covered entity may use or disclose PHI for payment without authorization." in tpo_block
    assert "Emergency Disclosure - Directory" in context
    assert "Business Associate Disclosure" in context
    assert "Authorization for Research" in context
    assert "Emergency directory disclosure" not in tpo_block
    assert "business associate" not in tpo_block.lower()
    assert "research-related treatment" not in tpo_block.lower()


def test_retriever_formats_and_deduplicates_mocked_neo4j_rows() -> None:
    client = FakeNeo4jClient()
    retriever = GraphRetriever(neo4j_client=client)  # type: ignore[arg-type]

    rows = retriever.retrieve("What obligations apply to covered entities?", limit=10)

    assert len(client.calls) >= 4
    assert len(rows) == 1
    assert rows[0]["query_route"] == "exact_phrase_entity"
    assert rows[0]["score"] > 0
    assert rows[0]["seed_name"] == "Covered Entity"
    assert rows[0]["relationship"] == "APPLIES_TO"
    assert rows[0]["statement_name"] == "Minimum Necessary Requirement"
    assert rows[0]["evidence_text"] == "A covered entity must make reasonable efforts to limit PHI disclosure."
    assert rows[0]["citation"] == "45 CFR 164.502"


def test_answer_prompt_contains_grounding_and_no_invention_instructions() -> None:
    prompt = build_rag_prompt("Question?", "Context")

    assert "using only the retrieved knowledge graph context" in prompt
    assert "Do not invent legal requirements" in prompt
    assert "graph does not contain enough evidence" in prompt
    assert "Evidence used" in prompt
    assert "without authorization" in prompt
    assert "{question}" in RAG_PROMPT_TEMPLATE


def test_answer_prompt_contains_without_authorization_safety_and_reference_rules() -> None:
    prompt = build_rag_prompt("When can a covered entity disclose PHI without authorization?", "Context")

    assert "without relying on a standard individual authorization" in prompt
    assert "The retrieved graph evidence may not be exhaustive." in prompt
    assert "Do not transfer conditions from one category to another" in prompt
    assert "Do not mix emergency directory conditions into Treatment, Payment, and Health Care Operations" in prompt
    assert "Evidence Reference: [4], Treatment, Payment, and Health Care Operations, hipaa.pdf" in prompt
    assert "Evidence References: [4], [5], [6]" in prompt


def test_retriever_citation_expression_uses_citation_arrays_only() -> None:
    assert re.search(r"statement\.citation(?!s)", _COMMON_RETURN) is None
    assert re.search(r"related\.citation(?!s)", _COMMON_RETURN) is None
    assert "head(coalesce(statement.citations, []))" in _COMMON_RETURN
    assert "head(coalesce(related.citations, []))" in _COMMON_RETURN
    assert "head(coalesce(rel.citations, []))" in _COMMON_RETURN


def test_answer_generator_uses_ollama_text_wrapper() -> None:
    client = FakeOllamaClient()
    generator = AnswerGenerator(ollama_client=client)  # type: ignore[arg-type]

    answer = generator.generate("What is required?", "[1]\nEvidence: Test")

    assert answer == "Grounded answer"
    assert client.model == "qwen3:8b"
    assert len(client.calls) == 1
    assert "What is required?" in client.calls[0][1]


def test_answer_generator_model_override_wins() -> None:
    client = FakeOllamaClient()
    AnswerGenerator(ollama_client=client, model="llama3.1")  # type: ignore[arg-type]

    assert client.model == "llama3.1"


def test_answer_generator_short_circuits_empty_context() -> None:
    client = FakeOllamaClient()
    generator = AnswerGenerator(ollama_client=client)  # type: ignore[arg-type]

    answer = generator.generate("What is required?", "")

    assert "does not contain enough evidence" in answer
    assert client.calls == []


def test_ask_parser_accepts_arguments_without_llm() -> None:
    args = build_parser().parse_args(
        [
            "When can a covered entity disclose PHI without authorization?",
            "--limit",
            "5",
            "--show-context",
            "--debug-retrieval",
            "--model",
            "llama3.1",
        ]
    )

    assert args.question.startswith("When can")
    assert args.limit == 5
    assert args.show_context is True
    assert args.debug_retrieval is True
    assert args.model == "llama3.1"
