from __future__ import annotations

from dataclasses import dataclass, field
import re
import string


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
    "prohibited practices",
    "biometric identification",
    "emotion recognition",
    "ai office",
    "board",
    "market surveillance authority",
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

DOMAIN_PHRASES = HIPAA_PHRASES + EU_AI_ACT_PHRASES + GENERAL_COMPLIANCE_PHRASES

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
    "risk management": ["mitigation", "safeguard", "control", "risk", "fundamental rights"],
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
    needs_rewrite: bool = False
    needs_decomposition: bool = False
    has_negated_authorization: bool = False


def build_query_plan(question: str, conversation_history: list[str] | None = None) -> QueryPlan:
    normalized = normalize_question(question)
    history = conversation_history or []
    needs_rewrite = _is_follow_up(normalized)
    rewritten = _rewrite_follow_up(normalized, history) if needs_rewrite else normalized
    effective_question = normalize_question(rewritten or normalized)

    phrases = detect_phrases(effective_question)
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
    expansion_terms = expand_terms([*phrases, *terms])
    domains = _detect_domains([*phrases, *terms, *expansion_terms])

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
        needs_rewrite=needs_rewrite and bool(rewritten),
        needs_decomposition=bool(sub_questions),
        has_negated_authorization=has_negated_authorization,
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
    if any(term in NEGATED_AUTHORIZATION_PHRASES for term in terms):
        expanded.extend(PERMISSION_EXCEPTION_EXPANSIONS)
    return [term for term in _dedupe(expanded) if term not in set(terms)]


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
    if term_set.intersection({_canonical_phrase(term) for term in GENERAL_COMPLIANCE_PHRASES}):
        domains.append("general compliance")
    return domains


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        normalized = _canonical_phrase(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique
