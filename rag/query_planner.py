from __future__ import annotations

from dataclasses import dataclass, field
import re
import string

from graph.ontology_seed import ontology_query_expansions


REGULATION_TO_SOURCE_DOCUMENT = {
    "GDPR": "gdpr.pdf",
    "HIPAA": "hipaa.pdf",
    "EU AI Act": "eu_ai_act.pdf",
    "EU_AI_ACT": "eu_ai_act.pdf",
}

QUERY_CATEGORIES = {
    "general_question",
    "compliance_question",
    "graph_lookup_question",
    "follow_up_question",
    "evidence_question",
    "unsupported_or_unclear",
}

HIPAA_PHRASES = [
    "covered entity",
    "business associate",
    "protected health information",
    "phi",
    "use and disclosure",
    "authorization",
    "without authorization",
    "minimum necessary",
    "individual rights",
    "privacy notice",
    "accounting of disclosures",
    "health oversight",
    "public health activities",
    "law enforcement",
    "judicial proceedings",
    "research purposes",
    "serious threat",
    "workers compensation",
    "safeguards",
    "training",
    "sanctions",
    "policies and procedures",
]

EU_AI_ACT_PHRASES = [
    "provider",
    "deployer",
    "importer",
    "distributor",
    "authorised representative",
    "authorized representative",
    "ai system",
    "high-risk ai system",
    "high risk ai system",
    "general-purpose ai model",
    "gpai",
    "technical documentation",
    "human oversight",
    "risk management",
    "post-market monitoring",
    "serious incident",
    "conformity assessment",
    "fundamental rights impact assessment",
    "transparency",
    "logging",
    "record keeping",
    "quality management",
    "quality management system",
    "cybersecurity",
    "robustness",
    "accuracy",
    "prohibited practices",
    "biometric identification",
    "emotion recognition",
    "ai office",
    "board",
    "market surveillance authority",
]

GDPR_PHRASES = [
    "gdpr",
    "controller",
    "data controller",
    "processor",
    "data processor",
    "data subject",
    "personal data",
    "patient data",
    "mental health data",
    "special category data",
    "special category health data",
    "special categories of personal data",
    "health data",
    "data concerning health",
    "article 9",
    "lawful basis",
    "legal basis",
    "consent",
    "explicit consent",
    "dpo",
    "data protection officer",
    "dpia",
    "data protection impact assessment",
    "supervisory authority",
    "data protection authority",
    "dpa",
    "personal data breach",
    "data breach",
    "breach notification",
    "right of access",
    "rights of access",
    "right to rectification",
    "rectification",
    "right to erasure",
    "erasure",
    "right to be forgotten",
    "right to restriction",
    "right to data portability",
    "data portability",
    "right to object",
    "automated decision-making",
    "automated decision making",
    "profiling",
    "records of processing activities",
    "privacy by design",
    "privacy by default",
]

ONTOLOGY_PHRASES = [
    "regulated actor",
    "regulated actors",
    "ai actor",
    "ai actors",
    "data actor",
    "data actors",
    "healthcare actor",
    "healthcare actors",
    "sensitive data",
    "clinical ai system",
    "clinical ai systems",
    "mental health diagnostic system",
    "mental health diagnostic systems",
    "mental health diagnostic ai system",
    "diagnostic ai",
    "clinical diagnostic",
    "clinical ai",
    "ai-supported diagnosis",
    "ai supported diagnosis",
    "risk score",
    "patient risk score",
    "mental health risk score",
    "screening data",
    "mental health screening",
    "clinical decision support",
    "symptom severity",
]

GENERAL_COMPLIANCE_PHRASES = [
    "obligation",
    "requirement",
    "permission",
    "prohibition",
    "exception",
    "condition",
    "deadline",
    "documentation",
    "monitoring",
    "reporting",
    "notification",
    "risk",
    "control",
    "safeguard",
    "evidence",
    "citation",
]

DOMAIN_PHRASES = HIPAA_PHRASES + EU_AI_ACT_PHRASES + GDPR_PHRASES + ONTOLOGY_PHRASES + GENERAL_COMPLIANCE_PHRASES

MENTAL_HEALTH_TERMS = [
    "mental health",
    "mental-health",
    "psychiatric",
    "depression",
    "anxiety",
    "ptsd",
    "adhd",
    "bipolar",
    "psychotic",
    "eating disorder",
    "self-harm",
    "suicide risk",
    "screening data",
    "mental health screening",
    "diagnostic ai",
    "clinical diagnostic",
    "clinical ai",
    "ai-supported diagnosis",
    "ai supported diagnosis",
    "risk score",
    "patient risk score",
    "mental health risk score",
    "symptom severity",
    "clinical decision support",
]

MENTAL_HEALTH_INTENT_RULES = [
    (
        "hipaa_phi",
        ("protected health information", "phi", "covered entity", "business associate", "authorization", "hipaa safeguards"),
        ["HIPAA"],
    ),
    (
        "gdpr_data_protection",
        (
            "controller",
            "processor",
            "special category",
            "health data",
            "lawful basis",
            "explicit consent",
            "data protection impact assessment",
            "dpia",
            "data subject rights",
        ),
        ["GDPR"],
    ),
    (
        "technical_provider_obligations",
        (
            "technical documentation",
            "post-market monitoring",
            "post market monitoring",
            "provider",
            "conformity assessment",
            "quality management",
            "risk management system",
        ),
        ["EU AI Act"],
    ),
    (
        "risk_score_safeguards",
        (
            "risk score",
            "patient risk score",
            "mental health risk score",
            "safeguard",
            "safeguards",
            "patient safety",
            "bias",
            "risk management",
            "security",
        ),
        ["GDPR", "HIPAA", "EU AI Act"],
    ),
    (
        "human_oversight",
        (
            "human oversight",
            "ai-supported diagnosis",
            "ai supported diagnosis",
            "clinical diagnostic ai",
            "clinical diagnostic",
            "clinical ai",
            "diagnostic ai",
            "automated decision",
            "human intervention",
        ),
        ["EU AI Act", "GDPR"],
    ),
    (
        "screening_data_risks",
        (
            "screening data",
            "mental health data",
            "data protection risks",
            "data protection risk",
            "patient data",
            "health data",
            "special category",
            "privacy risk",
            "disclosure",
        ),
        ["GDPR", "HIPAA"],
    ),
    (
        "transparency_explainability",
        (
            "transparency",
            "explainability",
            "explanation",
            "meaningful information",
            "instructions for use",
            "technical documentation",
        ),
        ["GDPR", "EU AI Act"],
    ),
]

MENTAL_HEALTH_USE_CASE_EXPANSION = [
    "MentalHealthDiagnosticSystem",
    "Mental Health Diagnostic AI System",
    "MentalHealthData",
    "mental health data",
    "HealthData",
    "health data",
    "SpecialCategoryData",
    "special category data",
    "ProtectedHealthInformation",
    "protected health information",
    "HighRiskAISystem",
    "high-risk AI system",
    "ClinicalAISystem",
    "clinical AI system",
    "DiagnosticPrediction",
    "diagnostic prediction",
    "PatientRiskScore",
    "patient risk score",
    "ClinicalDecisionSupportOutput",
    "clinical decision support",
    "HumanOversightRequirement",
    "human oversight",
    "DataProtectionImpactAssessment",
    "data protection impact assessment",
    "TechnicalDocumentation",
    "technical documentation",
    "PostMarketMonitoring",
    "post-market monitoring",
    "TransparencyRequirement",
    "transparency",
    "ExplainabilityRequirement",
    "explainability",
    "BiasRisk",
    "bias risk",
    "PatientSafetyRisk",
    "patient safety risk",
]

MENTAL_HEALTH_INTENT_EXPANSIONS = {
    "human_oversight": [
        "HumanOversightRequirement",
        "human oversight",
        "human intervention",
        "automated decision-making",
        "meaningful information",
        "profiling",
        "high-risk AI system",
        "deployer",
        "provider",
    ],
    "screening_data_risks": [
        "mental health data",
        "screening data",
        "health data",
        "special category data",
        "protected health information",
        "disclosure",
        "safeguards",
        "data protection impact assessment",
    ],
    "risk_score_safeguards": [
        "patient risk score",
        "mental health risk score",
        "risk management",
        "human oversight",
        "security",
        "safeguards",
        "bias risk",
        "patient safety risk",
        "protected health information",
        "special category data",
    ],
    "transparency_explainability": [
        "transparency",
        "explainability",
        "meaningful information",
        "instructions for use",
        "technical documentation",
        "automated decision-making",
    ],
    "technical_provider_obligations": [
        "technical documentation",
        "post-market monitoring",
        "provider",
        "conformity assessment",
        "quality management system",
        "risk management system",
    ],
    "hipaa_phi": [
        "protected health information",
        "phi",
        "covered entity",
        "business associate",
        "authorization",
        "safeguards",
    ],
    "gdpr_data_protection": [
        "controller",
        "processor",
        "special category data",
        "health data",
        "lawful basis",
        "explicit consent",
        "data protection impact assessment",
        "data subject rights",
    ],
}

NEGATED_AUTHORIZATION_PHRASES = [
    "without authorization",
    "authorization is not required",
    "no authorization required",
    "authorization or opportunity to agree or object is not required",
    "not require authorization",
]

PERMISSION_EXCEPTION_EXPANSIONS = [
    "authorization or opportunity to agree or object is not required",
    "permitted disclosure",
    "permitted use",
    "may disclose",
    "may use or disclose",
    "required by law",
    "public health activities",
    "health oversight activities",
    "judicial and administrative proceedings",
    "law enforcement purposes",
    "research purposes",
    "serious threat to health or safety",
    "specialized government functions",
    "workers compensation",
    "business associate",
    "minimum necessary",
]

PHRASE_ALIASES = {
    "covered entities": "covered entity",
    "business associates": "business associate",
    "providers": "provider",
    "deployers": "deployer",
    "importers": "importer",
    "distributors": "distributor",
    "authorised representatives": "authorised representative",
    "authorized representatives": "authorized representative",
    "high risk ai systems": "high-risk ai system",
    "high-risk ai systems": "high-risk ai system",
    "ai systems": "ai system",
    "general-purpose ai models": "general-purpose ai model",
    "obligations": "obligation",
    "requirements": "requirement",
    "permissions": "permission",
    "prohibitions": "prohibition",
    "exceptions": "exception",
    "conditions": "condition",
    "controls": "control",
    "safeguards": "safeguard",
    "citations": "citation",
    "controllers": "controller",
    "data controllers": "controller",
    "processors": "processor",
    "data processors": "processor",
    "data subjects": "data subject",
    "special category health data": "special category data",
    "special categories of personal data": "special category data",
    "data concerning health": "health data",
    "legal basis": "lawful basis",
    "dpos": "dpo",
    "data protection officers": "data protection officer",
    "data protection authorities": "supervisory authority",
    "dpa": "supervisory authority",
    "dpias": "dpia",
    "personal data breaches": "personal data breach",
    "data breaches": "personal data breach",
    "right to be forgotten": "right to erasure",
    "rights of access": "right of access",
    "rectification": "right to rectification",
    "erasure": "right to erasure",
    "data portability": "right to data portability",
    "automated decision making": "automated decision-making",
    "regulated actor": "regulated actors",
    "ai actor": "ai actors",
    "data actor": "data actors",
    "healthcare actor": "healthcare actors",
    "clinical ai systems": "clinical ai system",
    "mental health diagnostic systems": "mental health diagnostic system",
}

ACTOR_PHRASES = {
    "covered entity",
    "business associate",
    "provider",
    "deployer",
    "importer",
    "distributor",
    "authorised representative",
    "authorized representative",
    "ai office",
    "board",
    "market surveillance authority",
    "controller",
    "processor",
    "data subject",
    "supervisory authority",
    "data protection officer",
    "dpo",
}

OBJECT_PHRASES = {
    "protected health information",
    "phi",
    "ai system",
    "high-risk ai system",
    "high risk ai system",
    "general-purpose ai model",
    "gpai",
    "technical documentation",
    "fundamental rights impact assessment",
    "privacy notice",
    "accounting of disclosures",
    "personal data",
    "patient data",
    "mental health data",
    "special category data",
    "special category health data",
    "health data",
    "article 9",
    "lawful basis",
    "consent",
    "explicit consent",
    "data protection impact assessment",
    "dpia",
    "personal data breach",
    "right of access",
    "rights of access",
    "right to rectification",
    "rectification",
    "right to erasure",
    "erasure",
    "right to restriction",
    "right to data portability",
    "data portability",
    "right to object",
    "automated decision-making",
    "profiling",
}

MODALITY_KEYWORDS = {
    "must",
    "required",
    "requirement",
    "requirements",
    "obligation",
    "obligations",
    "duties",
    "responsibilities",
    "may",
    "can",
    "allowed",
    "permitted",
    "cannot",
    "must not",
    "prohibited",
    "not allowed",
    "forbidden",
    "exceptions",
    "unless",
    "except",
    "without authorization",
    "evidence",
    "source",
    "citation",
    "where does it say",
}

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "any",
    "apply",
    "are",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "is",
    "may",
    "must",
    "not",
    "of",
    "on",
    "or",
    "over",
    "shall",
    "should",
    "the",
    "their",
    "there",
    "to",
    "under",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "without",
}

EXPANSION_TERMS = {
    "gdpr": [
        "data subject",
        "article 9",
        "special category data",
        "health data",
        "data protection impact assessment",
    ],
    "hipaa": [
        "protected health information",
        "covered entity",
        "business associate",
        "use and disclosure",
        "authorization",
        "minimum necessary",
        "safeguards",
    ],
    "eu ai act": [
        "high-risk ai system",
        "provider",
        "deployer",
        "risk management",
        "technical documentation",
        "human oversight",
        "post-market monitoring",
    ],
    "authorization": [
        "use and disclosure",
        "permission",
        "exception",
        "without authorization",
        "required authorization",
    ],
    "without authorization": [
        "authorization",
        "exception",
        "permission",
        "use and disclosure",
        *PERMISSION_EXCEPTION_EXPANSIONS,
    ],
    "phi": ["protected health information", "disclosure", "use", "covered entity", "business associate", "minimum necessary"],
    "protected health information": [
        "phi",
        "disclosure",
        "use",
        "covered entity",
        "business associate",
        "minimum necessary",
    ],
    "high-risk ai system": [
        "provider",
        "deployer",
        "risk management",
        "technical documentation",
        "human oversight",
        "conformity assessment",
        "post-market monitoring",
        "quality management system",
        "cybersecurity",
        "record keeping",
    ],
    "high risk ai system": [
        "provider",
        "deployer",
        "risk management",
        "technical documentation",
        "human oversight",
        "conformity assessment",
        "post-market monitoring",
    ],
    "provider": [
        "technical documentation",
        "risk management",
        "conformity assessment",
        "instructions for use",
        "post-market monitoring",
        "quality management system",
        "cybersecurity",
    ],
    "deployer": [
        "human oversight",
        "fundamental rights impact assessment",
        "monitoring",
        "instructions for use",
    ],
    "documentation": ["technical documentation", "record keeping", "logs", "evidence", "instructions for use"],
    "technical documentation": ["documentation", "record keeping", "instructions for use", "evidence"],
    "risk": ["mitigation", "safeguard", "control", "risk management", "fundamental rights"],
    "risk management": [
        "risk management system",
        "risk mitigation",
        "residual risk",
        "known and foreseeable risks",
        "testing",
        "validation",
        "accuracy",
        "robustness",
        "cybersecurity",
        "human oversight",
        "technical documentation",
        "post-market monitoring",
        "conformity assessment",
        "control measures",
        "safeguards",
        "mitigation",
        "safeguard",
        "control",
        "risk",
        "fundamental rights",
    ],
    "conformity assessment": ["prior to market placement", "provider responsibility", "quality management system"],
    "monitoring": ["post-market monitoring", "serious incident", "corrective action"],
    "post-market monitoring": ["monitoring", "serious incident", "corrective action"],
    "cybersecurity": ["cyber resilience", "security controls", "data poisoning", "adversarial attacks", "robustness"],
    "human oversight": ["human oversight measures", "natural persons can oversee", "human operator", "operational constraints"],
    "record keeping": ["record-keeping", "logs", "traceability"],
    "quality management": ["quality management system", "post-market monitoring"],
    "personal data": ["data subject", "controller", "processor", "lawful basis", "consent", "special category data"],
    "patient data": [
        "personal data",
        "health data",
        "special category data",
        "article 9",
        "controller",
        "processor",
        "protected health information",
        "covered entity",
    ],
    "health data": ["special category data", "article 9", "lawful basis", "explicit consent", "controller", "processor"],
    "mental health data": ["health data", "special category data", "article 9", "personal data", "protected health information"],
    "special category data": ["health data", "article 9", "explicit consent", "lawful basis", "processing condition"],
    "article 9": ["special category data", "health data", "explicit consent", "processing condition"],
    "controller": ["processor", "data subject", "lawful basis", "data protection impact assessment", "breach notification"],
    "processor": ["controller", "data subject", "personal data"],
    "data subject": [
        "right of access",
        "access",
        "obtain confirmation",
        "right to rectification",
        "inaccurate personal data",
        "right to erasure",
        "right to be forgotten",
        "right to restriction",
        "restriction of processing",
        "restrict processing",
        "right to data portability",
        "data portability",
        "transmit those data",
        "right to object",
        "automated decision-making",
        "profiling",
        "human intervention",
        "meaningful information",
        "legal effects",
    ],
    "lawful basis": ["legal basis", "consent", "explicit consent"],
    "consent": ["lawful basis", "explicit consent"],
    "explicit consent": ["consent", "lawful basis", "special category data", "article 9"],
    "dpia": ["data protection impact assessment", "high risk", "health data", "special category data"],
    "data protection impact assessment": ["dpia", "high risk", "health data", "special category data"],
    "personal data breach": ["data breach", "breach notification", "supervisory authority"],
    "data breach": ["personal data breach", "breach notification", "supervisory authority"],
    "breach notification": ["personal data breach", "data breach", "supervisory authority"],
    "right of access": ["access", "obtain confirmation", "data subject"],
    "right to rectification": ["rectification", "inaccurate personal data", "data subject"],
    "right to erasure": ["right to be forgotten", "erasure", "data subject"],
    "right to restriction": ["restriction of processing", "restrict processing", "data subject"],
    "right to data portability": ["data portability", "transmit those data", "data subject"],
    "right to object": ["object to processing", "profiling", "data subject"],
    "automated decision-making": ["profiling", "human intervention", "meaningful information", "legal effects", "right to object", "data subject"],
    "mental health diagnostic system": [
        "MentalHealthDiagnosticSystem",
        "MentalHealthData",
        "HealthData",
        "SpecialCategoryData",
        "ProtectedHealthInformation",
        "HighRiskAISystem",
        "HumanOversightRequirement",
        "DataProtectionImpactAssessment",
        "TechnicalDocumentation",
        "PostMarketMonitoring",
        "Provider",
        "Deployer",
        "Controller",
        "Processor",
        "CoveredEntity",
        "BusinessAssociate",
    ],
    "mental health data": [
        "MentalHealthData",
        "HealthData",
        "SpecialCategoryData",
        "ProtectedHealthInformation",
        "article 9",
        "explicit consent",
        "controller",
        "processor",
        "covered entity",
        "business associate",
    ],
}

INTENT_RULES = [
    (
        "evidence",
        ("evidence", "source", "citation", "where does it say", "where does the regulation say"),
        ["Citation", "SourceChunk", "Section", "Regulation"],
        ["CITES", "CONTAINS", "HAS_SECTION", "REFERENCES"],
    ),
    (
        "exceptions",
        ("exceptions", "exception", "unless", "except", "without authorization"),
        ["Permission", "Exception", "Statement"],
        ["HAS_EXCEPTION", "HAS_CONDITION", "GRANTS_PERMISSION", "DISCLOSES_TO", "USES_FOR", "APPLIES_TO", "RELATED_TO"],
    ),
    (
        "prohibitions",
        ("cannot", "must not", "prohibited", "not allowed", "forbidden"),
        ["Statement", "Prohibition"],
        ["PROHIBITS", "LIMITED_BY", "APPLIES_TO", "RELATED_TO"],
    ),
    (
        "permissions",
        ("may", "can", "allowed", "permitted", "when can"),
        ["Statement", "Permission"],
        ["GRANTS_PERMISSION", "HAS_EXCEPTION", "APPLIES_TO", "RELATED_TO", "DISCLOSES_TO", "USES_FOR"],
    ),
    (
        "obligations",
        ("must", "required", "requirements", "requirement", "obligations", "obligation", "duties", "responsibilities"),
        ["Statement", "Obligation", "Requirement"],
        [
            "HAS_REQUIREMENT",
            "IMPOSES_ON",
            "REQUIRES_DOCUMENTATION",
            "REQUIRES_ASSESSMENT",
            "REQUIRES_MONITORING",
            "REQUIRES_HUMAN_OVERSIGHT",
        ],
    ),
]


@dataclass
class QueryPlan:
    original_question: str
    normalized_question: str
    category: str
    intent: str
    terms: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    detected_actors: list[str] = field(default_factory=list)
    detected_objects: list[str] = field(default_factory=list)
    detected_modalities: list[str] = field(default_factory=list)
    detected_domains: list[str] = field(default_factory=list)
    preferred_statement_labels: list[str] = field(default_factory=list)
    preferred_relationships: list[str] = field(default_factory=list)
    expansion_terms: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    concept_groups: dict[str, list[str]] = field(default_factory=dict)
    required_concepts: list[str] = field(default_factory=list)
    ambiguous_terms: list[str] = field(default_factory=list)
    needs_rewrite: bool = False
    needs_decomposition: bool = False
    has_negated_authorization: bool = False
    explicit_regulations: list[str] = field(default_factory=list)
    target_source_documents: list[str] = field(default_factory=list)
    explicit_single_regulation: bool = False
    cross_regulation: bool = False
    is_mental_health_query: bool = False
    mental_health_intent_labels: list[str] = field(default_factory=list)
    inferred_regulations: list[str] = field(default_factory=list)
    mental_health_expansion_terms: list[str] = field(default_factory=list)


def build_query_plan(question: str, conversation_history: list[str] | None = None) -> QueryPlan:
    normalized = normalize_question(question)
    history = conversation_history or []
    needs_rewrite = _is_follow_up(normalized)
    rewritten = _rewrite_follow_up(normalized, history) if needs_rewrite else normalized
    effective_question = normalize_question(rewritten or normalized)

    explicit_regulations = _detect_explicit_regulations(effective_question)
    explicit_terms = [_regulation_search_term(regulation) for regulation in explicit_regulations]
    phrases = _dedupe([*detect_phrases(effective_question), *explicit_terms])
    terms = extract_query_terms(effective_question, phrases)
    modalities = _detect_modalities(effective_question)
    intent, labels, relationships = _detect_intent(effective_question)
    has_negated_authorization = _has_negated_authorization(effective_question)
    if has_negated_authorization:
        intent = "exceptions"
        labels = ["Permission", "Exception", "Statement"]
        relationships = [
            "HAS_EXCEPTION",
            "HAS_CONDITION",
            "GRANTS_PERMISSION",
            "DISCLOSES_TO",
            "USES_FOR",
            "APPLIES_TO",
            "RELATED_TO",
        ]
    category = categorize_query(effective_question, phrases, modalities, needs_rewrite, bool(history))
    sub_questions = decompose_question(effective_question)
    is_mental_health_query = _is_mental_health_query(effective_question)
    mental_health_intent_labels = _detect_mental_health_intents(effective_question) if is_mental_health_query else []
    inferred_regulations = (
        _infer_mental_health_regulations(mental_health_intent_labels, effective_question)
        if is_mental_health_query and not explicit_regulations
        else []
    )
    mental_health_expansion_terms = (
        _mental_health_expansion_terms(mental_health_intent_labels) if is_mental_health_query else []
    )
    expansion_terms = _dedupe([*expand_terms([*phrases, *terms, *explicit_terms]), *mental_health_expansion_terms])
    domains = _dedupe_regulations([*_detect_domains([*phrases, *terms, *expansion_terms]), *explicit_regulations])
    concept_groups = _build_concept_groups(effective_question, phrases, terms, modalities, intent)
    target_source_documents = [
        REGULATION_TO_SOURCE_DOCUMENT[regulation]
        for regulation in explicit_regulations
        if regulation in REGULATION_TO_SOURCE_DOCUMENT
    ]
    cross_regulation = len(explicit_regulations) > 1 or _asks_cross_regulation(effective_question)
    if not explicit_regulations and cross_regulation:
        target_source_documents = [
            REGULATION_TO_SOURCE_DOCUMENT["GDPR"],
            REGULATION_TO_SOURCE_DOCUMENT["HIPAA"],
            REGULATION_TO_SOURCE_DOCUMENT["EU AI Act"],
        ]
    if inferred_regulations:
        domains = _dedupe_regulations([*domains, *inferred_regulations])
        target_source_documents = [
            REGULATION_TO_SOURCE_DOCUMENT[regulation]
            for regulation in inferred_regulations
            if regulation in REGULATION_TO_SOURCE_DOCUMENT
        ]
        cross_regulation = len(inferred_regulations) > 1
    required_concepts = [name for name, values in concept_groups.items() if values]
    ambiguous_terms = _detect_ambiguous_terms(effective_question, terms)

    if category == "follow_up_question" and needs_rewrite and not history:
        category = "unsupported_or_unclear"

    return QueryPlan(
        original_question=question,
        normalized_question=effective_question,
        category=category,
        intent=intent,
        terms=terms,
        phrases=phrases,
        detected_actors=[phrase for phrase in phrases if phrase in ACTOR_PHRASES],
        detected_objects=[_canonical_phrase(phrase) for phrase in phrases if phrase in OBJECT_PHRASES],
        detected_modalities=modalities,
        detected_domains=domains,
        preferred_statement_labels=labels,
        preferred_relationships=relationships,
        expansion_terms=expansion_terms,
        sub_questions=sub_questions,
        concept_groups=concept_groups,
        required_concepts=required_concepts,
        ambiguous_terms=ambiguous_terms,
        needs_rewrite=needs_rewrite and bool(rewritten),
        needs_decomposition=bool(sub_questions),
        has_negated_authorization=has_negated_authorization,
        explicit_regulations=explicit_regulations,
        target_source_documents=_dedupe(target_source_documents),
        explicit_single_regulation=len(explicit_regulations) == 1 and not cross_regulation,
        cross_regulation=cross_regulation,
        is_mental_health_query=is_mental_health_query,
        mental_health_intent_labels=mental_health_intent_labels,
        inferred_regulations=inferred_regulations,
        mental_health_expansion_terms=mental_health_expansion_terms,
    )


def categorize_query(
    normalized_question: str,
    phrases: list[str] | None = None,
    modalities: list[str] | None = None,
    is_follow_up: bool = False,
    has_history: bool = False,
) -> str:
    text = normalized_question.strip()
    if not text:
        return "unsupported_or_unclear"
    if is_follow_up:
        return "follow_up_question" if has_history else "unsupported_or_unclear"
    if any(marker in text for marker in ("evidence", "source", "citation", "where does it say", "where does the regulation say")):
        return "evidence_question"
    if phrases or modalities:
        return "compliance_question"
    if any(word in text for word in ("node", "graph", "relationship", "entity", "statement", "citation", "section")):
        return "graph_lookup_question"
    if any(word in text for word in ("weather", "sports", "recipe", "movie", "song")):
        return "general_question"
    return "general_question"


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.lower().replace("–", "-").replace("—", "-")).strip()


def detect_phrases(question: str) -> list[str]:
    normalized = normalize_question(question)
    phrases = []
    for phrase in DOMAIN_PHRASES:
        if phrase in normalized:
            phrases.append(_canonical_phrase(phrase))
    for phrase in NEGATED_AUTHORIZATION_PHRASES:
        if phrase in normalized:
            phrases.append(_canonical_phrase(phrase))
    for alias, canonical in PHRASE_ALIASES.items():
        if alias in normalized:
            phrases.append(canonical)
    return _dedupe(phrases)


def extract_query_terms(question: str, phrases: list[str] | None = None) -> list[str]:
    normalized = normalize_question(question)
    terms = list(phrases or detect_phrases(normalized))
    translator = str.maketrans({char: " " for char in string.punctuation if char != "-"})
    tokenized = normalized.translate(translator)
    for token in re.split(r"\s+", tokenized):
        token = token.strip("-")
        if len(token) >= 3 and token not in STOPWORDS:
            terms.append(_canonical_phrase(token))
    return _dedupe(terms)


def expand_terms(terms: list[str]) -> list[str]:
    expanded: list[str] = []
    for term in terms:
        expanded.extend(EXPANSION_TERMS.get(term, []))
        expanded.extend(_ontology_expansion_terms(term))
    if any(term in NEGATED_AUTHORIZATION_PHRASES for term in terms):
        expanded.extend(PERMISSION_EXCEPTION_EXPANSIONS)
    return [term for term in _dedupe(expanded) if term not in set(terms)]


def _detect_explicit_regulations(question: str) -> list[str]:
    text = normalize_question(question)
    regulations: list[str] = []
    if re.search(r"\b(?:under\s+(?:the\s+)?)?gdpr\b", text) or "general data protection regulation" in text:
        regulations.append("GDPR")
    if re.search(r"\b(?:under\s+(?:the\s+)?)?hipaa\b", text):
        regulations.append("HIPAA")
    if (
        re.search(r"\b(?:under\s+(?:the\s+)?)?eu ai act\b", text)
        or "european union artificial intelligence act" in text
        or "regulation eu 2024 1689" in text
        or "regulation (eu) 2024/1689" in text
    ):
        regulations.append("EU AI Act")
    return _dedupe_regulations(regulations)


def _regulation_search_term(regulation: str) -> str:
    if regulation == "EU AI Act":
        return "eu ai act"
    return regulation.lower()


def _dedupe_regulations(values: list[str]) -> list[str]:
    canonical = {
        "gdpr": "GDPR",
        "general data protection regulation": "GDPR",
        "hipaa": "HIPAA",
        "eu ai act": "EU AI Act",
        "eu_ai_act": "EU AI Act",
        "regulation (eu) 2024/1689": "EU AI Act",
        "regulation eu 2024 1689": "EU AI Act",
        "general compliance": "general compliance",
    }
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        label = canonical.get(str(value).lower().strip(), str(value).strip())
        if label and label not in seen:
            seen.add(label)
            unique.append(label)
    return unique


def _asks_cross_regulation(question: str) -> bool:
    text = normalize_question(question)
    if any(marker in text for marker in ("compare", "across", "cross-regulation", "cross regulation")):
        return True
    explicit = _detect_explicit_regulations(text)
    if len(explicit) > 1:
        return True
    mental_health_ai_patient = (
        any(term in text for term in ("mental health diagnostic", "clinical diagnostic", "mental health ai", "diagnostic ai"))
        and any(term in text for term in ("patient data", "health data", "mental health data", "personal data"))
        and not explicit
    )
    return mental_health_ai_patient


def _is_mental_health_query(question: str) -> bool:
    text = normalize_question(question)
    return any(term in text for term in MENTAL_HEALTH_TERMS)


def _detect_mental_health_intents(question: str) -> list[str]:
    text = normalize_question(question)
    labels: list[str] = []
    for label, markers, _regulations in MENTAL_HEALTH_INTENT_RULES:
        if any(marker in text for marker in markers):
            labels.append(label)
    if _is_mental_health_query(text) and "mental_health_use_case" not in labels:
        labels.append("mental_health_use_case")
    return _dedupe_raw(labels)


def _infer_mental_health_regulations(intent_labels: list[str], question: str) -> list[str]:
    text = normalize_question(question)
    if not intent_labels:
        return []

    regulations: list[str] = []
    for label, _markers, rule_regulations in MENTAL_HEALTH_INTENT_RULES:
        if label in intent_labels:
            regulations.extend(rule_regulations)

    if "mental_health_use_case" in intent_labels and not regulations:
        if any(term in text for term in ("ai system", "diagnostic ai", "clinical ai", "patient data", "health data")):
            regulations.extend(["GDPR", "HIPAA", "EU AI Act"])

    return _dedupe_regulations(regulations)


def _mental_health_expansion_terms(intent_labels: list[str]) -> list[str]:
    if not intent_labels:
        return []
    expanded: list[str] = []
    base_terms = [
        "MentalHealthDiagnosticSystem",
        "Mental Health Diagnostic AI System",
        "MentalHealthData",
        "mental health data",
        "HealthData",
        "health data",
        "ClinicalAISystem",
        "clinical AI system",
        "DiagnosticPrediction",
        "diagnostic prediction",
        "ClinicalDecisionSupportOutput",
        "clinical decision support",
    ]
    expanded.extend(base_terms)
    for label in intent_labels:
        expanded.extend(MENTAL_HEALTH_INTENT_EXPANSIONS.get(label, []))
    if len(intent_labels) == 1 and intent_labels[0] == "mental_health_use_case":
        expanded.extend(MENTAL_HEALTH_USE_CASE_EXPANSION[:12])
    return _dedupe(expanded)


def decompose_question(question: str) -> list[str]:
    text = normalize_question(question)
    sub_questions: list[str] = []

    if "providers and deployers" in text and "high-risk ai system" in text:
        sub_questions.append("What obligations do providers have for high-risk AI systems?")
        sub_questions.append("What obligations do deployers have for high-risk AI systems?")
    elif "provider and deployer" in text and "high-risk ai system" in text:
        sub_questions.append("What obligations do providers have for high-risk AI systems?")
        sub_questions.append("What obligations do deployers have for high-risk AI systems?")

    if "documentation and monitoring requirements" in text or "monitoring and documentation requirements" in text:
        sub_questions.append("What documentation requirements apply?")
        sub_questions.append("What monitoring requirements apply?")

    return _dedupe(sub_questions)


def _detect_intent(question: str) -> tuple[str, list[str], list[str]]:
    text = normalize_question(question)
    if _has_negated_authorization(text):
        return (
            "exceptions",
            ["Permission", "Exception", "Statement"],
            ["HAS_EXCEPTION", "HAS_CONDITION", "GRANTS_PERMISSION", "DISCLOSES_TO", "USES_FOR", "APPLIES_TO", "RELATED_TO"],
        )
    for intent, markers, labels, relationships in INTENT_RULES:
        if any(marker in text for marker in markers):
            return intent, labels, relationships
    return "lookup", ["Statement"], ["APPLIES_TO", "RELATED_TO", "REFERENCES"]


def _detect_modalities(question: str) -> list[str]:
    text = normalize_question(question)
    return [keyword for keyword in MODALITY_KEYWORDS if keyword in text]


def _has_negated_authorization(question: str) -> bool:
    text = normalize_question(question)
    return any(phrase in text for phrase in NEGATED_AUTHORIZATION_PHRASES)


def _detect_domains(terms: list[str]) -> list[str]:
    term_set = set(terms)
    domains: list[str] = []
    if term_set.intersection({_canonical_phrase(term) for term in HIPAA_PHRASES}):
        domains.append("HIPAA")
    if term_set.intersection({_canonical_phrase(term) for term in EU_AI_ACT_PHRASES}):
        domains.append("EU AI Act")
    if term_set.intersection({_canonical_phrase(term) for term in GDPR_PHRASES}):
        domains.append("GDPR")
    if term_set.intersection({_canonical_phrase(term) for term in GENERAL_COMPLIANCE_PHRASES}):
        domains.append("general compliance")
    return domains


def _build_concept_groups(
    question: str,
    phrases: list[str],
    terms: list[str],
    modalities: list[str],
    intent: str,
) -> dict[str, list[str]]:
    text = normalize_question(question)
    term_set = set(phrases) | set(terms)
    groups: dict[str, list[str]] = {}

    actors = []
    if "provider" in term_set or "provider" in text:
        actors.extend(["provider", "providers"])
    if "deployer" in term_set or "deployer" in text:
        actors.extend(["deployer", "deployers"])
    if "health care provider" in text:
        actors.extend(["health care provider", "health care providers"])
    if "covered entity" in term_set:
        actors.append("covered entity")
    if "controller" in term_set or "controller" in text:
        actors.extend(["controller", "data controller"])
    if "processor" in term_set or "processor" in text:
        actors.extend(["processor", "data processor"])
    if "data subject" in term_set or "data subject" in text:
        actors.append("data subject")
    if "supervisory authority" in term_set or "supervisory authority" in text:
        actors.extend(["supervisory authority", "data protection authority"])
    if "data protection officer" in term_set or "dpo" in term_set or "dpo" in text:
        actors.extend(["data protection officer", "dpo"])
    if actors:
        groups["actor"] = _dedupe_raw(actors)

    objects = []
    if "high-risk ai system" in term_set or "high risk ai system" in text or "high-risk" in text:
        objects.extend(["high-risk ai system", "high risk ai system", "high-risk artificial intelligence system"])
    if "ai system" in term_set or "ai system" in text:
        objects.extend(["ai system", "artificial intelligence system"])
    if "phi" in term_set or "protected health information" in term_set:
        objects.extend(["phi", "protected health information"])
    if "personal data" in term_set or "personal data" in text:
        objects.append("personal data")
    if "patient data" in term_set or "patient data" in text:
        objects.extend(["patient data", "personal data", "protected health information"])
    if "mental health data" in term_set or "mental health data" in text:
        objects.extend(["mental health data", "health data", "special category data", "protected health information"])
    if "health data" in term_set or "health data" in text or "data concerning health" in text:
        objects.extend(["health data", "data concerning health", "special category data"])
    if "special category data" in term_set or "special category" in text:
        objects.extend(["special category data", "special categories of personal data"])
    if "article 9" in term_set or "article 9" in text:
        objects.extend(["article 9", "special category data", "health data"])
    if "lawful basis" in term_set or "lawful basis" in text:
        objects.extend(["lawful basis", "legal basis"])
    if "explicit consent" in term_set or "explicit consent" in text:
        objects.extend(["explicit consent", "consent"])
    elif "consent" in term_set or "consent" in text:
        objects.append("consent")
    if "data protection impact assessment" in term_set or "dpia" in term_set or "dpia" in text:
        objects.extend(["data protection impact assessment", "dpia"])
    if "personal data breach" in term_set or "data breach" in term_set or "data breach" in text:
        objects.extend(["personal data breach", "data breach", "breach notification"])
    if "right of access" in term_set or "right of access" in text:
        objects.append("right of access")
    if "right to rectification" in term_set or "rectification" in term_set or "rectification" in text:
        objects.append("right to rectification")
    if "right to erasure" in term_set or "right to erasure" in text or "right to be forgotten" in text:
        objects.extend(["right to erasure", "right to be forgotten"])
    if "right to restriction" in term_set or "right to restriction" in text:
        objects.append("right to restriction")
    if "right to data portability" in term_set or "data portability" in term_set or "data portability" in text:
        objects.append("right to data portability")
    if "right to object" in term_set or "right to object" in text:
        objects.append("right to object")
    if "automated decision-making" in term_set or "automated decision-making" in text or "profiling" in term_set:
        objects.extend(["automated decision-making", "profiling"])
    if objects:
        groups["object"] = _dedupe_raw(objects)

    topics = []
    topic_groups = {
        "documentation": ["documentation", "technical documentation", "record keeping", "record-keeping", "logs", "traceability", "instructions for use"],
        "risk management": ["risk management", "risk management system", "risk mitigation", "lifecycle", "residual risk", "known and foreseeable risks"],
        "conformity assessment": ["conformity assessment", "prior to market placement", "provider responsibility for conformity assessment"],
        "monitoring": ["monitoring", "post-market monitoring", "serious incident", "corrective action"],
        "cybersecurity": ["cybersecurity", "cyber resilience", "security controls", "data poisoning", "adversarial attacks", "robustness"],
        "human oversight": ["human oversight", "human oversight measures", "natural persons can oversee", "human operator", "operational constraints"],
        "quality management": ["quality management", "quality management system"],
        "record keeping": ["record keeping", "record-keeping", "logs", "traceability"],
        "lawful basis": ["lawful basis", "legal basis", "consent", "explicit consent", "article 6"],
        "article 9": ["article 9", "special category data", "health data", "explicit consent", "processing condition"],
        "data subject rights": [
            "right of access",
            "right to rectification",
            "right to erasure",
            "right to restriction",
            "right to data portability",
            "right to object",
            "automated decision-making",
            "profiling",
        ],
        "breach notification": ["personal data breach", "data breach", "breach notification", "notify supervisory authority"],
        "privacy by design": ["privacy by design", "privacy by default", "data protection by design and by default"],
        "dpia": ["dpia", "data protection impact assessment", "high risk processing"],
        "records of processing": ["records of processing activities", "record of processing activities"],
    }
    for topic_name, variants in topic_groups.items():
        if topic_name in text or any(variant in text for variant in variants):
            topics.extend(variants)
    if topics:
        groups["topic"] = _dedupe_raw(topics)

    modality = []
    if intent == "obligations" or any(item in text for item in ("obligation", "required", "must", "should")):
        modality.extend(["obligation", "obligations", "requirement", "requirements", "compliance", "must", "should"])
    if "documentation" in term_set or "documentation" in text:
        modality.extend(["documentation", "technical documentation", "instructions for use"])
    if modalities:
        modality.extend(modalities)
    if modality:
        groups["modality"] = _dedupe_raw(modality)

    domain = []
    if any(item in text for item in ("eu ai act", "high-risk ai", "ai system", "artificial intelligence")):
        domain.extend(["eu ai act", "eu_ai_act", "regulation (eu) 2024/1689", "this regulation"])
    if any(item in text for item in ("hipaa", "phi", "covered entity", "health care provider")):
        domain.extend(["hipaa", "protected health information", "covered entity"])
    if any(
        item in text
        for item in (
            "gdpr",
            "controller",
            "processor",
            "data subject",
            "personal data",
            "patient data",
            "health data",
            "mental health data",
            "special category",
            "article 9",
            "lawful basis",
            "consent",
            "dpo",
            "dpia",
            "supervisory authority",
            "data breach",
            "right of access",
            "rectification",
            "erasure",
            "data portability",
            "automated decision",
            "profiling",
        )
    ):
        domain.extend(["gdpr", "personal data", "health data", "controller", "processor"])
    if any(item in text for item in ("patient data", "health data", "mental health data")):
        domain.extend(["hipaa", "protected health information"])
    if domain:
        groups["domain"] = _dedupe_raw(domain)

    return groups


def _detect_ambiguous_terms(question: str, terms: list[str]) -> list[str]:
    text = normalize_question(question)
    ambiguous = []
    if "provider" in terms or "provider" in text:
        ambiguous.append("provider")
    return ambiguous


def _is_follow_up(question: str) -> bool:
    text = normalize_question(question)
    return (
        len(text.split()) <= 5
        and (
            text.startswith(("what about", "and ", "also ", "what if"))
            or text in {"exceptions", "and exceptions", "providers", "deployers"}
        )
    )


def _rewrite_follow_up(question: str, conversation_history: list[str]) -> str:
    if not conversation_history:
        return ""
    history_text = normalize_question(" ".join(conversation_history[-3:]))
    text = normalize_question(question)

    if "provider" in text and ("eu ai act" in history_text or "high-risk ai system" in history_text or "ai system" in history_text):
        return "What obligations apply to providers under the EU AI Act?"
    if "deployer" in text and ("eu ai act" in history_text or "high-risk ai system" in history_text or "ai system" in history_text):
        return "What obligations apply to deployers under the EU AI Act?"
    if "exception" in text:
        if "phi" in history_text or "protected health information" in history_text or "disclosure" in history_text:
            return "What exceptions apply to PHI disclosure?"
        if "authorization" in history_text:
            return "What exceptions apply to authorization requirements?"
    return f"{text} {history_text}".strip()


def _canonical_phrase(phrase: str) -> str:
    phrase = phrase.lower().strip()
    if phrase in PHRASE_ALIASES:
        return PHRASE_ALIASES[phrase]
    if phrase == "high risk ai system":
        return "high-risk ai system"
    return phrase


def _ontology_expansion_terms(term: str) -> list[str]:
    expanded: list[str] = []
    for class_name in ontology_query_expansions(term):
        expanded.append(_class_name_to_phrase(class_name))
    return expanded


def _class_name_to_phrase(class_name: str) -> str:
    known = {
        "AISystem": "ai system",
        "HighRiskAISystem": "high risk ai system",
        "ClinicalAISystem": "clinical ai system",
    }
    if class_name in known:
        return _canonical_phrase(known[class_name])
    spaced = re.sub(r"(?<!^)(?=[A-Z])", " ", class_name)
    return _canonical_phrase(spaced.lower())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = _canonical_phrase(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _dedupe_raw(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = value.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique
