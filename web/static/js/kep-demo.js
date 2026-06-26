// ============================================================================
// kep-demo.js — sample /ask payloads (REAL backend response shape) used as a
// fallback when the Flask /ask endpoint is unreachable (e.g. opened as a static
// file), and as the example-question seed list. Exposed as window.KEP_DEMO.
//
// Shape mirrors web/services/rag_service.answer_question():
//   { answer, evidence[], graph{nodes,edges,meta}, context, debug{query_plan,
//     top_rows, expansion, selection}, metrics{...}, response_source, error }
// ============================================================================

(function () {
  // ---- helpers to keep the sample graphs faithful to _build_graph_payload --
  function reg(id, label, src) { return { id, label, type: "regulation", source_document: src }; }
  function stmt(id, label, src, cite, route, score) { return { id, label, type: "statement", source_document: src, citation: cite, route, score }; }
  function ent(id, label, src) { return { id, label, type: "entity", source_document: src }; }
  function ev(id, label, src, cite) { return { id, label, type: "evidence", source_document: src, citation: cite }; }

  // ===================== Q1 · HIPAA =========================================
  const HIPAA = {
    response_source: "live", error: null, context: "",
    answer:
      "A covered entity may disclose protected health information (PHI) **without the individual's authorization** in a defined set of situations under the HIPAA Privacy Rule.\n\n" +
      "- **Treatment, payment, and health care operations (TPO)** are permitted without authorization [E1].\n" +
      "- Disclosures **required by another law** are permitted to the extent that law compels them [E2].\n" +
      "- **Public health activities**, such as disease reporting to a public health authority, are permitted [E3].\n\n" +
      "Every permitted use or disclosure remains subject to the **minimum necessary** standard, except disclosures made for treatment [E4].",
    evidence: [
      { id: 1, statement: "Uses and disclosures for treatment, payment, or health care operations", source_document: "HIPAA Privacy Rule", evidence_text: "A covered entity may use or disclose protected health information for treatment, payment, or health care operations without authorization, subject to the consent provisions where applicable.", citation: "45 CFR § 164.506", score: 0.91, source_group: "hipaa" },
      { id: 2, statement: "Disclosures required by law", source_document: "HIPAA Privacy Rule", evidence_text: "A covered entity may use or disclose protected health information to the extent that such use or disclosure is required by law and is limited to the relevant requirements of such law.", citation: "45 CFR § 164.512(a)", score: 0.86, source_group: "hipaa" },
      { id: 3, statement: "Disclosures for public health activities", source_document: "HIPAA Privacy Rule", evidence_text: "A covered entity may disclose protected health information for public health activities to a public health authority that is authorized by law to collect or receive such information for the purpose of preventing or controlling disease, injury, or disability.", citation: "45 CFR § 164.512(b)", score: 0.82, source_group: "hipaa" },
      { id: 4, statement: "Minimum necessary standard", source_document: "HIPAA Privacy Rule", evidence_text: "A covered entity must make reasonable efforts to limit protected health information to the minimum necessary to accomplish the intended purpose. This does not apply to disclosures for treatment.", citation: "45 CFR § 164.502(b)", score: 0.78, source_group: "hipaa" },
    ],
    graph: {
      nodes: [
        { id: "question", label: "Disclose PHI without authorization?", type: "question" },
        ent("entity:tpo", "Treatment / payment / ops", "HIPAA Privacy Rule"),
        ent("entity:required-law", "Required by law", "HIPAA Privacy Rule"),
        ent("entity:public-health", "Public health", "HIPAA Privacy Rule"),
        ent("entity:min-necessary", "Minimum necessary", "HIPAA Privacy Rule"),
        stmt("stmt:164-506", "§ 164.506", "HIPAA Privacy Rule", "45 CFR § 164.506", "concept→statement", 0.91),
        stmt("stmt:164-512a", "§ 164.512(a)", "HIPAA Privacy Rule", "45 CFR § 164.512(a)", "phrase→statement", 0.86),
        stmt("stmt:164-512b", "§ 164.512(b)", "HIPAA Privacy Rule", "45 CFR § 164.512(b)", "concept→statement", 0.82),
        stmt("stmt:164-502b", "§ 164.502(b)", "HIPAA Privacy Rule", "45 CFR § 164.502(b)", "concept→statement", 0.78),
        ev("evidence:1", "§ 164.506", "HIPAA Privacy Rule", "45 CFR § 164.506"),
        ev("evidence:2", "§ 164.512(a)", "HIPAA Privacy Rule", "45 CFR § 164.512(a)"),
        ev("evidence:3", "§ 164.512(b)", "HIPAA Privacy Rule", "45 CFR § 164.512(b)"),
        ev("evidence:4", "§ 164.502(b)", "HIPAA Privacy Rule", "45 CFR § 164.502(b)"),
        reg("reg:HIPAA Privacy Rule", "HIPAA Privacy Rule", "HIPAA Privacy Rule"),
      ],
      edges: [
        e("question", "entity:tpo", "RELATED_CONCEPT"), e("question", "entity:required-law", "RELATED_CONCEPT"),
        e("question", "entity:public-health", "RELATED_CONCEPT"), e("question", "entity:min-necessary", "RELATED_CONCEPT"),
        e("entity:tpo", "stmt:164-506", "ranked_for_question", "1"), e("entity:required-law", "stmt:164-512a", "ranked_for_question", "2"),
        e("entity:public-health", "stmt:164-512b", "ranked_for_question", "3"), e("entity:min-necessary", "stmt:164-502b", "ranked_for_question", "4"),
        e("stmt:164-506", "evidence:1", "retrieved_as_evidence", "1"), e("stmt:164-512a", "evidence:2", "retrieved_as_evidence", "2"),
        e("stmt:164-512b", "evidence:3", "retrieved_as_evidence", "3"), e("stmt:164-502b", "evidence:4", "retrieved_as_evidence", "4"),
        e("stmt:164-506", "reg:HIPAA Privacy Rule", "supports_answer", "1"), e("stmt:164-512a", "reg:HIPAA Privacy Rule", "supports_answer", "2"),
        e("stmt:164-512b", "reg:HIPAA Privacy Rule", "supports_answer", "3"), e("stmt:164-502b", "reg:HIPAA Privacy Rule", "supports_answer", "4"),
      ],
      meta: { source: "live_retrieval_rows" },
    },
    debug: {
      query_plan: {
        category: "Permission / Exception",
        intent: "Enumerate lawful disclosures of PHI that do not require prior authorization.",
        detected_phrases: ["without authorization", "covered entity", "disclose PHI"],
        detected_actors: ["Covered Entity", "Business Associate"],
        detected_objects: ["Protected Health Information"],
        detected_domains: ["Privacy", "Permitted Uses & Disclosures"],
        concept_groups: ["treatment", "payment", "operations", "public health"],
        expansion_terms: ["treatment", "payment", "health care operations", "required by law", "public health activity", "minimum necessary"],
        sub_questions: [
          "What disclosures are permitted for treatment, payment, or operations?",
          "When is disclosure required or permitted by law?",
          "What public-interest exceptions apply without authorization?",
        ],
        explicit_regulations: ["HIPAA"], inferred_regulations: [], target_source_documents: ["HIPAA Privacy Rule"], cross_regulation: false,
      },
      top_rows: rows([
        [0.91, "concept→statement", "Treatment, payment & operations", "hipaa", ["treatment", "payment", "operations"], ["exact-citation +0.12", "domain-match +0.06"], []],
        [0.86, "phrase→statement", "Required by law", "hipaa", ["required by law"], ["intent-match +0.09"], []],
        [0.82, "concept→statement", "Public health activities", "hipaa", ["public health", "authority"], ["domain-match +0.06"], ["broad-term -0.03"]],
        [0.78, "concept→statement", "Minimum necessary", "hipaa", ["minimum necessary"], [], ["secondary-topic -0.05"]],
        [0.64, "phrase→statement", "Judicial & administrative proceedings", "hipaa", ["proceeding"], [], ["low-relevance -0.11"]],
      ], "HIPAA Privacy Rule"),
      expansion: {}, selection: {},
    },
    metrics: metrics({ rows: 38, ev: 12, routing: 180, retrieval: 1050, gq: 910, rank: 260, gen: 1320, cov: { hipaa: 12, eu_ai_act: 0, gdpr: 0, unknown: 0 } }),
  };

  // ===================== Q2 · EU AI Act =====================================
  const AIACT = {
    response_source: "live", error: null, context: "",
    answer:
      "Providers of high-risk AI systems must satisfy a **cumulative set of obligations** under Chapter III of the EU AI Act before and after placing the system on the market.\n\n" +
      "1. Establish and maintain a continuous **risk-management system** across the lifecycle [E1].\n" +
      "2. Apply **data-governance** practices to training, validation, and testing data sets [E2].\n" +
      "3. Draw up **technical documentation** and keep automatically generated **logs** [E3].\n" +
      "4. Ensure **transparency** and enable effective **human oversight** [E4].\n" +
      "5. Operate an overarching **quality-management system** covering the above [E5].",
    evidence: [
      { id: 1, statement: "Risk management system", source_document: "Regulation (EU) 2024/1689", evidence_text: "A risk management system shall be established, implemented, documented and maintained in relation to high-risk AI systems as a continuous iterative process planned and run throughout the entire lifecycle of the system.", citation: "AI Act Art. 9", score: 0.93, source_group: "eu_ai_act" },
      { id: 2, statement: "Data and data governance", source_document: "Regulation (EU) 2024/1689", evidence_text: "High-risk AI systems which make use of techniques involving the training of models with data shall be developed on the basis of training, validation and testing data sets that meet the quality criteria, subject to appropriate data governance and management practices.", citation: "AI Act Art. 10", score: 0.90, source_group: "eu_ai_act" },
      { id: 3, statement: "Technical documentation and record-keeping", source_document: "Regulation (EU) 2024/1689", evidence_text: "The technical documentation of a high-risk AI system shall be drawn up before that system is placed on the market. High-risk AI systems shall technically allow for the automatic recording of events (logs) over the lifetime of the system.", citation: "AI Act Art. 11 & 12", score: 0.88, source_group: "eu_ai_act" },
      { id: 4, statement: "Transparency and human oversight", source_document: "Regulation (EU) 2024/1689", evidence_text: "High-risk AI systems shall be designed so their operation is sufficiently transparent to enable deployers to interpret a system's output, and shall be designed such that they can be effectively overseen by natural persons.", citation: "AI Act Art. 13 & 14", score: 0.85, source_group: "eu_ai_act" },
      { id: 5, statement: "Quality management system", source_document: "Regulation (EU) 2024/1689", evidence_text: "Providers of high-risk AI systems shall put a quality management system in place that ensures compliance with this Regulation, documented in a systematic and orderly manner in the form of written policies, procedures and instructions.", citation: "AI Act Art. 16 & 17", score: 0.81, source_group: "eu_ai_act" },
    ],
    graph: {
      nodes: [
        { id: "question", label: "Provider obligations · high-risk AI", type: "question" },
        ent("entity:risk-mgmt", "Risk management", "Regulation (EU) 2024/1689"),
        ent("entity:data-gov", "Data governance", "Regulation (EU) 2024/1689"),
        ent("entity:docs", "Documentation & logs", "Regulation (EU) 2024/1689"),
        ent("entity:transparency", "Transparency & oversight", "Regulation (EU) 2024/1689"),
        ent("entity:qms", "Quality management", "Regulation (EU) 2024/1689"),
        stmt("stmt:art9", "Art. 9", "Regulation (EU) 2024/1689", "AI Act Art. 9", "concept→statement", 0.93),
        stmt("stmt:art10", "Art. 10", "Regulation (EU) 2024/1689", "AI Act Art. 10", "concept→statement", 0.90),
        stmt("stmt:art11", "Art. 11–12", "Regulation (EU) 2024/1689", "AI Act Art. 11 & 12", "phrase→statement", 0.88),
        stmt("stmt:art13", "Art. 13–14", "Regulation (EU) 2024/1689", "AI Act Art. 13 & 14", "concept→statement", 0.85),
        stmt("stmt:art16", "Art. 16–17", "Regulation (EU) 2024/1689", "AI Act Art. 16 & 17", "concept→statement", 0.81),
        ev("evidence:1", "Art. 9", "Regulation (EU) 2024/1689", "AI Act Art. 9"),
        ev("evidence:2", "Art. 10", "Regulation (EU) 2024/1689", "AI Act Art. 10"),
        ev("evidence:3", "Art. 11–12", "Regulation (EU) 2024/1689", "AI Act Art. 11 & 12"),
        ev("evidence:4", "Art. 13–14", "Regulation (EU) 2024/1689", "AI Act Art. 13 & 14"),
        ev("evidence:5", "Art. 16–17", "Regulation (EU) 2024/1689", "AI Act Art. 16 & 17"),
        reg("reg:euai", "EU AI Act · Ch. III", "Regulation (EU) 2024/1689"),
      ],
      edges: [
        e("question", "entity:risk-mgmt", "RELATED_CONCEPT"), e("question", "entity:data-gov", "RELATED_CONCEPT"),
        e("question", "entity:docs", "RELATED_CONCEPT"), e("question", "entity:transparency", "RELATED_CONCEPT"), e("question", "entity:qms", "RELATED_CONCEPT"),
        e("entity:risk-mgmt", "stmt:art9", "ranked_for_question", "1"), e("entity:data-gov", "stmt:art10", "ranked_for_question", "2"),
        e("entity:docs", "stmt:art11", "ranked_for_question", "3"), e("entity:transparency", "stmt:art13", "ranked_for_question", "4"), e("entity:qms", "stmt:art16", "ranked_for_question", "5"),
        e("stmt:art9", "evidence:1", "retrieved_as_evidence", "1"), e("stmt:art10", "evidence:2", "retrieved_as_evidence", "2"),
        e("stmt:art11", "evidence:3", "retrieved_as_evidence", "3"), e("stmt:art13", "evidence:4", "retrieved_as_evidence", "4"), e("stmt:art16", "evidence:5", "retrieved_as_evidence", "5"),
        e("stmt:art9", "reg:euai", "supports_answer", "1"), e("stmt:art10", "reg:euai", "supports_answer", "2"),
        e("stmt:art11", "reg:euai", "supports_answer", "3"), e("stmt:art13", "reg:euai", "supports_answer", "4"), e("stmt:art16", "reg:euai", "supports_answer", "5"),
      ],
      meta: { source: "live_retrieval_rows" },
    },
    debug: {
      query_plan: {
        category: "Obligation / Requirement",
        intent: "Enumerate the compliance obligations imposed on providers of high-risk AI systems.",
        detected_phrases: ["high-risk AI system", "providers", "obligations"],
        detected_actors: ["Provider", "Deployer", "Notified Body"],
        detected_objects: ["High-Risk AI System"],
        detected_domains: ["Risk Management", "Data Governance", "Transparency", "Human Oversight"],
        concept_groups: ["risk management", "data governance", "documentation", "oversight", "quality"],
        expansion_terms: ["risk management system", "data governance", "technical documentation", "record-keeping", "transparency", "human oversight", "accuracy", "quality management"],
        sub_questions: [
          "What risk-management and data-governance duties apply before market placement?",
          "What documentation, logging, and transparency obligations apply?",
          "What ongoing oversight and quality-management duties apply?",
        ],
        explicit_regulations: ["EU AI Act"], inferred_regulations: [], target_source_documents: ["Regulation (EU) 2024/1689"], cross_regulation: false,
      },
      top_rows: rows([
        [0.93, "concept→statement", "Risk-management system", "eu_ai_act", ["risk management", "lifecycle"], ["exact-citation +0.12", "intent-match +0.09"], []],
        [0.90, "concept→statement", "Data governance", "eu_ai_act", ["data governance", "training data"], ["domain-match +0.06"], []],
        [0.88, "phrase→statement", "Technical documentation & logs", "eu_ai_act", ["technical documentation", "logs"], ["exact-citation +0.12"], []],
        [0.85, "concept→statement", "Transparency & human oversight", "eu_ai_act", ["transparency", "human oversight"], ["domain-match +0.06"], []],
        [0.81, "concept→statement", "Quality-management system", "eu_ai_act", ["quality management"], [], ["secondary-topic -0.04"]],
        [0.59, "phrase→statement", "Conformity assessment", "eu_ai_act", ["conformity"], [], ["low-relevance -0.13"]],
      ], "Regulation (EU) 2024/1689"),
      expansion: {}, selection: {},
    },
    metrics: metrics({ rows: 46, ev: 16, routing: 160, retrieval: 1180, gq: 1040, rank: 300, gen: 1480, cov: { hipaa: 0, eu_ai_act: 16, gdpr: 0, unknown: 0 } }),
  };

  // ===================== Q3 · GDPR ==========================================
  const GDPR = {
    response_source: "live", error: null, context: "",
    answer:
      "**No — explicit consent is not always required.** Processing of health data is prohibited by default under Article 9(1), but Article 9(2) lists several conditions that lift the prohibition, of which explicit consent is only one.\n\n" +
      "- Health data is a **special category**; its processing is prohibited by default [E1].\n" +
      "- **Explicit consent** under Art. 9(2)(a) is one way to lift the prohibition — but not the only one [E2].\n" +
      "- Processing for **medical diagnosis or healthcare** (Art. 9(2)(h)) is permitted without consent [E3].\n" +
      "- **Public-interest grounds in public health** (Art. 9(2)(i)) also permit processing [E4].\n\n" +
      "A separate **Article 6 lawful basis** must still be identified alongside the Article 9 condition.",
    evidence: [
      { id: 1, statement: "Processing of special categories of personal data", source_document: "Regulation (EU) 2016/679", evidence_text: "Processing of personal data revealing data concerning health shall be prohibited.", citation: "GDPR Art. 9(1)", score: 0.89, source_group: "gdpr" },
      { id: 2, statement: "Explicit consent", source_document: "Regulation (EU) 2016/679", evidence_text: "Paragraph 1 shall not apply if the data subject has given explicit consent to the processing of those personal data for one or more specified purposes, except where Union or Member State law provide that the prohibition may not be lifted by the data subject.", citation: "GDPR Art. 9(2)(a)", score: 0.87, source_group: "gdpr" },
      { id: 3, statement: "Processing for healthcare provision", source_document: "Regulation (EU) 2016/679", evidence_text: "Processing is necessary for the purposes of preventive or occupational medicine, medical diagnosis, the provision of health or social care or treatment, on the basis of Union or Member State law or pursuant to contract with a health professional.", citation: "GDPR Art. 9(2)(h)", score: 0.84, source_group: "gdpr" },
      { id: 4, statement: "Processing for public health", source_document: "Regulation (EU) 2016/679", evidence_text: "Processing is necessary for reasons of public interest in the area of public health, such as protecting against serious cross-border threats to health, on the basis of Union or Member State law.", citation: "GDPR Art. 9(2)(i)", score: 0.80, source_group: "gdpr" },
    ],
    graph: {
      nodes: [
        { id: "question", label: "Consent for health data?", type: "question" },
        ent("entity:special-cat", "Special-category ban", "Regulation (EU) 2016/679"),
        ent("entity:consent", "Explicit consent", "Regulation (EU) 2016/679"),
        ent("entity:healthcare", "Healthcare provision", "Regulation (EU) 2016/679"),
        ent("entity:public-health", "Public health", "Regulation (EU) 2016/679"),
        stmt("stmt:9-1", "Art. 9(1)", "Regulation (EU) 2016/679", "GDPR Art. 9(1)", "concept→statement", 0.89),
        stmt("stmt:9-2a", "Art. 9(2)(a)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(a)", "phrase→statement", 0.87),
        stmt("stmt:9-2h", "Art. 9(2)(h)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(h)", "concept→statement", 0.84),
        stmt("stmt:9-2i", "Art. 9(2)(i)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(i)", "concept→statement", 0.80),
        ev("evidence:1", "Art. 9(1)", "Regulation (EU) 2016/679", "GDPR Art. 9(1)"),
        ev("evidence:2", "Art. 9(2)(a)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(a)"),
        ev("evidence:3", "Art. 9(2)(h)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(h)"),
        ev("evidence:4", "Art. 9(2)(i)", "Regulation (EU) 2016/679", "GDPR Art. 9(2)(i)"),
        reg("reg:gdpr", "GDPR · Art. 6 & 9", "Regulation (EU) 2016/679"),
      ],
      edges: [
        e("question", "entity:special-cat", "RELATED_CONCEPT"), e("question", "entity:consent", "RELATED_CONCEPT"),
        e("question", "entity:healthcare", "RELATED_CONCEPT"), e("question", "entity:public-health", "RELATED_CONCEPT"),
        e("entity:special-cat", "stmt:9-1", "ranked_for_question", "1"), e("entity:consent", "stmt:9-2a", "ranked_for_question", "2"),
        e("entity:healthcare", "stmt:9-2h", "ranked_for_question", "3"), e("entity:public-health", "stmt:9-2i", "ranked_for_question", "4"),
        e("stmt:9-1", "evidence:1", "retrieved_as_evidence", "1"), e("stmt:9-2a", "evidence:2", "retrieved_as_evidence", "2"),
        e("stmt:9-2h", "evidence:3", "retrieved_as_evidence", "3"), e("stmt:9-2i", "evidence:4", "retrieved_as_evidence", "4"),
        e("stmt:9-1", "reg:gdpr", "supports_answer", "1"), e("stmt:9-2a", "reg:gdpr", "supports_answer", "2"),
        e("stmt:9-2h", "reg:gdpr", "supports_answer", "3"), e("stmt:9-2i", "reg:gdpr", "supports_answer", "4"),
      ],
      meta: { source: "live_retrieval_rows" },
    },
    debug: {
      query_plan: {
        category: "Conditional / Nuance",
        intent: "Determine whether explicit consent is the only lawful basis for processing health data.",
        detected_phrases: ["explicit consent", "health data", "always need"],
        detected_actors: ["Controller", "Data Subject"],
        detected_objects: ["Health Data (special category)"],
        detected_domains: ["Lawful Basis", "Special Categories", "Consent"],
        concept_groups: ["consent", "special category", "healthcare", "public health"],
        expansion_terms: ["explicit consent", "special categories", "lawful basis", "vital interests", "public health", "healthcare provision", "substantial public interest"],
        sub_questions: [
          "Is processing of health data prohibited by default?",
          "Is explicit consent the only way to lift that prohibition?",
          "What alternative Article 9(2) conditions permit processing?",
        ],
        explicit_regulations: ["GDPR"], inferred_regulations: [], target_source_documents: ["Regulation (EU) 2016/679"], cross_regulation: false,
      },
      top_rows: rows([
        [0.89, "concept→statement", "Special-category prohibition", "gdpr", ["health data", "prohibited"], ["exact-citation +0.12"], []],
        [0.87, "phrase→statement", "Explicit consent", "gdpr", ["explicit consent"], ["intent-match +0.09"], []],
        [0.84, "concept→statement", "Healthcare provision", "gdpr", ["medical diagnosis", "health care"], ["domain-match +0.06"], []],
        [0.80, "concept→statement", "Public-health interest", "gdpr", ["public health", "public interest"], [], ["broad-term -0.03"]],
        [0.61, "phrase→statement", "Vital interests", "gdpr", ["vital interests"], [], ["low-relevance -0.10"]],
      ], "Regulation (EU) 2016/679"),
      expansion: {}, selection: {},
    },
    metrics: metrics({ rows: 34, ev: 11, routing: 170, retrieval: 1010, gq: 880, rank: 250, gen: 1260, cov: { hipaa: 0, eu_ai_act: 0, gdpr: 11, unknown: 0 } }),
  };

  // ---- small builders ------------------------------------------------------
  function e(source, target, label, evidence_id) {
    const o = { source, target, label, relationship_type: label.includes("→") || label === "RELATED_CONCEPT" ? label : "UI_RETRIEVAL" };
    if (evidence_id) o.evidence_id = evidence_id;
    return o;
  }
  function rows(arr, src) {
    return arr.map(([score, route, statement, group, matched, boosts, penalties]) => ({
      score, route, statement, relationship: "", source_group: group, source_document: src,
      matched_concept_groups: [], matched_terms: matched, matched_topic_groups: [], boosts, penalties,
    }));
  }
  function metrics({ rows, ev, routing, retrieval, gq, rank, gen, cov }) {
    const total = routing + retrieval + rank + gen;
    return {
      top_k: 30, rows_retrieved: rows, evidence_items: ev, elapsed_ms: total,
      routing_ms: routing, retrieval_ms: retrieval, graph_query_ms: gq, ranking_ms: rank,
      generation_ms: gen, total_ms: total, model: "qwen3:8b", source_coverage: cov,
      breakdown: { hipaa: cov.hipaa, eu_ai_act: cov.eu_ai_act, gdpr: cov.gdpr, source_chunks: rows, entity_nodes: 0 },
    };
  }

  window.KEP_DEMO = {
    examples: [
      { regulation: "HIPAA", question: "When can a covered entity disclose PHI without the individual's authorization?", payload: HIPAA },
      { regulation: "EU AI Act", question: "What are the obligations for providers of high-risk AI systems?", payload: AIACT },
      { regulation: "GDPR", question: "Do you always need explicit consent to process health data under GDPR?", payload: GDPR },
    ],
    // crude keyword match used only in offline fallback mode
    match(question) {
      const q = (question || "").toLowerCase();
      if (q.includes("ai") || q.includes("high-risk") || q.includes("provider")) return AIACT;
      if (q.includes("gdpr") || q.includes("consent") || q.includes("health data")) return GDPR;
      return HIPAA;
    },
  };
})();
