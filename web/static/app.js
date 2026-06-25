const DEMO_QUESTIONS = [
  "What obligations apply to a mental health diagnostic AI system?",
  "Can patient health data be used without explicit authorisation?",
  "What must providers of high-risk AI systems document?",
  "Which GDPR rights apply to a patient using this app?",
  "What risks should be mitigated before deployment?",
];

const PIPELINE_STEPS = [
  "Question intake",
  "Regulation routing",
  "Graph traversal",
  "Evidence ranking",
  "Answer synthesis",
  "Response ready",
];
const TECHNICAL_TABS = [
  { id: "matched-entities", label: "Matched entities" },
  { id: "retrieval-scores", label: "Retrieval scores" },
  { id: "evidence-ids", label: "Evidence IDs" },
  { id: "graph-traversal", label: "Graph traversal details" },
  { id: "pipeline-timings", label: "Pipeline timings" },
];

const EVENT_DEMO_CONFIG = {
  EVENT_DEMO_MODE: true,
  DEMO_CACHE_ENABLED: false,
  DEMO_FALLBACK_ENABLED: true,
  LIVE_CUSTOM_QUESTIONS_ENABLED: true,
  DEMO_PRESET_TIMEOUT_MS: 8000,
  CUSTOM_QUESTION_TIMEOUT_MS: 60000,
};
const SESSION_STORAGE_KEY = "kepEventDemoSession";
const UI_STORAGE_KEY = "kepEventDemoUiPrefs";
const SELECTED_STORAGE_KEY = "kepEventDemoSelectedQuestion";
const DEFAULT_UI_PREFS = {
  sidebarCollapsed: false,
  leftWidth: 340,
  middleWidth: 1040,
  rightWidth: 470,
  technicalDetailsOpen: false,
  technicalDetailsTab: "matched-entities",
  collapsedSections: {
    graph: false,
    technical: true,
  },
  graphMode: "demo",
  graphZoom: 1,
  graphFullscreen: false,
};

const state = {
  questions: readSession(),
  selectedId: null,
  isLoading: false,
  ui: readUiPrefs(),
};

const els = {
  form: document.querySelector("#question-form"),
  input: document.querySelector("#question-input"),
  askButton: document.querySelector("#ask-button"),
  composerSourceNote: document.querySelector("#composer-source-note"),
  charCount: document.querySelector("#char-count"),
  newSessionButton: document.querySelector("#new-session-button"),
  clearSessionButton: document.querySelector("#clear-session-button"),
  resetDemoButton: document.querySelector("#reset-demo-button"),
  demoGrid: document.querySelector("#demo-grid"),
  leftSidebar: document.querySelector("#left-sidebar"),
  questionCards: document.querySelector("#question-cards"),
  sessionState: document.querySelector("#session-state"),
  stepper: document.querySelector("#stepper"),
  graphStage: document.querySelector("#graph-stage"),
  technicalDetailsDrawer: document.querySelector("#technical-details-drawer"),
  answerContent: document.querySelector("#answer-content"),
  evidenceContent: document.querySelector("#evidence-content"),
  copyAnswerButton: document.querySelector("#copy-answer-button"),
};

if (state.questions.length) {
  const storedSelectedId = localStorage.getItem(SELECTED_STORAGE_KEY);
  state.selectedId = state.questions.some((item) => item.id === storedSelectedId) ? storedSelectedId : state.questions.at(-1).id;
}

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = els.input.value.trim();
  if (!question || state.isLoading) return;
  els.input.value = "";
  await submitQuestion(question);
});

els.input.addEventListener("input", () => {
  const badge = document.querySelector("#composer-source-badge");
  if (badge) {
    const source = composerSourceState();
    badge.textContent = source.label;
    badge.className = `active-response-state source-${slug(source.key)}`;
  }
  if (els.composerSourceNote) els.composerSourceNote.textContent = composerSourceNote();
  updateCharacterCount();
});

els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) event.preventDefault();
});

els.newSessionButton.addEventListener("click", () => {
  resetDemo();
});

els.clearSessionButton?.addEventListener("click", (event) => {
  event.preventDefault();
  event.stopPropagation();
  resetDemo();
});

els.resetDemoButton.addEventListener("click", () => {
  resetDemo();
});

function resetDemo() {
  state.questions = [];
  state.selectedId = null;
  state.isLoading = false;
  state.ui.graphMode = "demo";
  state.ui.graphZoom = 1;
  persistSession();
  persistSelectedQuestion();
  persistUiPrefs();
  render();
}

els.copyAnswerButton.addEventListener("click", () => {
  const selected = getSelectedQuestion();
  if (selected?.answer) navigator.clipboard?.writeText(selected.answer);
});

document.addEventListener("click", (event) => {
  const questionCard = event.target.closest("[data-question-id]");
  if (questionCard) {
    state.selectedId = questionCard.dataset.questionId;
    const selected = getSelectedQuestion();
    if (selected && els.input) {
      els.input.value = selected.question;
      updateCharacterCount();
    }
    persistSelectedQuestion();
    render();
  }

  const sourcesToggle = event.target.closest("[data-sources-toggle]");
  if (sourcesToggle) {
    toggleSourcesForSelected();
    return;
  }

  const technicalTab = event.target.closest("[data-technical-tab]");
  if (technicalTab) {
    state.ui.technicalDetailsTab = technicalTab.dataset.technicalTab;
    persistUiPrefs();
    renderTechnicalDetails();
    return;
  }

  const sectionToggle = event.target.closest("[data-section-toggle]");
  if (sectionToggle) {
    toggleSection(sectionToggle.dataset.sectionToggle);
    return;
  }

  const graphAction = event.target.closest("[data-graph-action]");
  if (graphAction) {
    handleGraphAction(graphAction.dataset.graphAction);
    return;
  }
});

setupResizeHandles();
window.addEventListener?.("resize", () => {
  applyLayoutPrefs();
  fitGraph();
});
applyLayoutPrefs();

async function submitQuestion(question) {
  const item = {
    id: crypto.randomUUID(),
    question,
    createdAt: new Date().toISOString(),
    expanded: true,
    status: "loading",
    pipelineStep: 0,
    domain: detectDomain(question),
    regulations: inferRegulations(question),
    answer: "",
    summary: "",
    obligations: [],
    verdict: buildVerdict(question, [], detectDomain(question)),
    evidence: [],
    graph: illustrativeDemoGraph(question, [], false),
    error: "",
    usedCache: false,
    responseSource: "live",
    sourcesCollapsed: false,
  };

  state.questions = state.questions.map((entry) => ({ ...entry, expanded: false }));
  state.questions.push(item);
  state.selectedId = item.id;
  persistSelectedQuestion();
  state.isLoading = true;
  render();

  const timer = startPipelineAnimation(item.id);

  try {
    const result = await askQuestionWithDemoPolicy(question);
    updateQuestion(item.id, {
      ...result,
      status: "complete",
      pipelineStep: PIPELINE_STEPS.length - 1,
      expanded: true,
      error: "",
    });
  } catch (error) {
    const responseSource = error.isTimeout ? "timeout" : "error";
    updateQuestion(item.id, {
      status: "error",
      pipelineStep: PIPELINE_STEPS.length - 1,
      error: responseSource === "timeout" ? "Live retrieval timed out." : "Live retrieval failed.",
      responseSource,
      graph: errorGraph(question),
      evidence: [],
      answer: "",
    });
  } finally {
    clearInterval(timer);
    state.isLoading = false;
    persistSession();
    render();
  }
}

function startPipelineAnimation(id) {
  return setInterval(() => {
    const item = state.questions.find((entry) => entry.id === id);
    if (!item || item.status !== "loading") return;
    updateQuestion(id, { pipelineStep: Math.min(item.pipelineStep + 1, PIPELINE_STEPS.length - 2) }, false);
    renderStepper();
    renderQuestionCards();
  }, 540);
}

async function askQuestionWithDemoPolicy(question) {
  const policy = buildRequestPolicy(question);
  logRequestPolicy(policy);

  if (policy.useCache) {
    await wait(180);
    return normalizeDemoResponse(question, true, "cached_fallback");
  }

  if (!policy.livePipelineCalled) {
    throw new Error("Live custom questions are disabled.");
  }

  try {
    const data = await fetchAnswer(question, policy.timeoutMs);
    return normalizeBackendResponse(question, data, false, "live");
  } catch (error) {
    if (policy.cacheHit && policy.fallbackEnabled && policy.isDemoPreset) {
      return normalizeDemoResponse(question, true, "cached_fallback");
    }
    throw error;
  }
}

async function fetchAnswer(question, timeoutMs) {
  const safeTimeoutMs = normalizeTimeoutMs(timeoutMs, EVENT_DEMO_CONFIG.CUSTOM_QUESTION_TIMEOUT_MS);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), safeTimeoutMs);
  let response;
  try {
    response = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: controller.signal,
    body: JSON.stringify({
      question,
      limit: 24,
      show_context: false,
      debug: true,
    }),
  });
  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error("Live retrieval timed out.");
      timeoutError.isTimeout = true;
      throw timeoutError;
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
  const data = await response.json();
  if (!response.ok || data.error) throw new Error(data.error || "The backend returned an error.");
  return data;
}

function normalizeBackendResponse(question, data, usedCache, responseSource = "live") {
  const evidence = (data.evidence || []).slice(0, 6).map((item, index) => {
    const sourceDocument = item.source_document || item.document || item.source || "Unknown source document";
    const statementTitle = item.statement || item.statement_name || item.title || "Matched regulatory statement";
    const regulation = regulationFromText(`${item.source_group || ""} ${sourceDocument} ${statementTitle}`);
    const reference = citationFromEvidenceItem(item, sourceDocument);
    return {
      id: String(item.id || index + 1),
      regulation,
      sourceDocument,
      reference,
      statementTitle,
      concept: item.matched_concept || item.concept || statementTitle || "Matched compliance concept",
      explanation: item.why_selected || explainEvidenceUse(regulation, statementTitle, question),
      passage: item.evidence_text || "",
      score: normalizeScore(item.score, 0.92 - index * 0.04),
    };
  });
  const noEvidence = evidence.length === 0 || data.response_source === "no_evidence" || data.metrics?.no_evidence === true;
  if (noEvidence) {
    return normalizeNoEvidenceResponse(question, data, usedCache);
  }

  return {
    answer: data.answer || `The live backend returned no answer text for: ${question}`,
    summary: summarizeAnswer(data.answer, question),
    obligations: obligationsForQuestion(question),
    evidence,
    graph: normalizeBackendGraphPayload(data.graph, question, evidence),
    debug: data.debug || {},
    domain: detectDomain(question, data.debug?.query_plan),
    regulations: inferRegulations(question, evidence),
    verdict: buildVerdict(question, evidence, detectDomain(question, data.debug?.query_plan)),
    metrics: {
      evidencePassages: data.metrics?.evidence_items || evidence.length,
      searchTimeMs: data.metrics?.elapsed_ms || null,
      routingMs: data.metrics?.routing_ms || null,
      retrievalMs: data.metrics?.retrieval_ms || null,
      graphQueryMs: data.metrics?.graph_query_ms || null,
      rankingMs: data.metrics?.ranking_ms || null,
      generationMs: data.metrics?.generation_ms || null,
      totalMs: data.metrics?.total_ms || data.metrics?.elapsed_ms || null,
    },
    usedCache,
    responseSource: data.response_source || responseSource,
  };
}

function normalizeNoEvidenceResponse(question, data = {}, usedCache = false) {
  const message = data.answer || "No sufficient regulatory evidence was found in the knowledge graph for this question.";
  const domain = detectDomain(question, data.debug?.query_plan);
  return {
    answer: message,
    summary: message,
    obligations: [],
    evidence: [],
    graph: normalizeBackendGraphPayload(data.graph, question, []),
    debug: data.debug || {},
    domain,
    regulations: [],
    verdict: {
      assessment: "No evidence found",
      concern: "The knowledge graph did not return regulatory sources for this question.",
      laws: [],
    },
    metrics: {
      evidencePassages: 0,
      searchTimeMs: data.metrics?.elapsed_ms || null,
      routingMs: data.metrics?.routing_ms || null,
      retrievalMs: data.metrics?.retrieval_ms || null,
      graphQueryMs: data.metrics?.graph_query_ms || null,
      rankingMs: data.metrics?.ranking_ms || null,
      generationMs: data.metrics?.generation_ms || 0,
      totalMs: data.metrics?.total_ms || data.metrics?.elapsed_ms || null,
      noEvidence: true,
    },
    usedCache,
    responseSource: "no_evidence",
  };
}

function normalizeBackendGraphPayload(graph, question, evidence = []) {
  const rawNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const rawEdges = Array.isArray(graph?.edges) ? graph.edges : [];
  if (graph?.meta?.source === "no_evidence" || (!evidence.length && rawNodes.length <= 1 && rawEdges.length === 0)) {
    return noEvidenceGraph(question, graph?.meta?.message);
  }
  if (rawNodes.length) {
    return {
      nodes: normalizeGraphNodes(rawNodes),
      edges: normalizeGraphEdges(rawEdges),
      advancedNodes: [],
      advancedEdges: [],
      meta: {
        ...(graph.meta || {}),
        source: graph.meta?.source || "live_retrieval_rows",
      },
    };
  }
  return evidenceGraphFromRetrievedEvidence(question, evidence);
}

function noEvidenceGraph(question, message = "No sufficient regulatory evidence was found in the knowledge graph for this question.") {
  return {
    nodes: normalizeGraphNodes([{ id: "question", label: question || "Question", type: "question" }]),
    edges: [],
    advancedNodes: [],
    advancedEdges: [],
    meta: { source: "no_evidence", message },
  };
}

function evidenceGraphFromRetrievedEvidence(question, evidence = []) {
  const nodes = [{ id: "question", label: question || "Question", type: "question" }];
  const edges = [];
  evidence.slice(0, 8).forEach((item, index) => {
    const evidenceId = String(item.id || index + 1);
    const regulationId = `reg:${slug(item.sourceDocument || item.regulation || "source")}`;
    const statementId = `stmt:${slug(item.statementTitle || item.concept || evidenceId)}`;
    const evidenceNodeId = `evidence:${evidenceId}`;
    nodes.push(
      { id: regulationId, label: item.sourceDocument || item.regulation || "Source document", type: "regulation", source_document: item.sourceDocument },
      { id: statementId, label: item.statementTitle || item.concept || "Retrieved statement", type: "statement", source_document: item.sourceDocument, citation: item.reference },
      { id: evidenceNodeId, label: item.reference || `Evidence ${evidenceId}`, type: "evidence", source_document: item.sourceDocument, citation: item.reference, evidence_id: evidenceId },
    );
    edges.push(
      { source: "question", target: statementId, label: "ranked_for_question", relationship_type: "UI_RETRIEVAL", evidence_id: evidenceId },
      { source: statementId, target: evidenceNodeId, label: "retrieved_as_evidence", relationship_type: "UI_RETRIEVAL", evidence_id: evidenceId },
      { source: evidenceNodeId, target: regulationId, label: "source_document", relationship_type: "UI_RETRIEVAL", evidence_id: evidenceId },
    );
  });
  return {
    nodes: normalizeGraphNodes(nodes),
    edges: normalizeGraphEdges(edges),
    advancedNodes: [],
    advancedEdges: [],
    meta: { source: "retrieved_evidence_ui_graph" },
  };
}

function normalizeDemoResponse(question, usedCache, responseSource = usedCache ? "cached_fallback" : "live") {
  const demo = findDemoResponse(question) || DEMO_CACHE.default;
  return {
    answer: demo.answer,
    summary: demo.summary,
    obligations: demo.obligations,
    evidence: demo.evidence.map((item, index) => ({
      ...item,
      id: String(index + 1),
      sourceDocument: item.sourceDocument || demoSourceDocument(item.regulation),
      statementTitle: item.statementTitle || item.concept || "Regulatory evidence",
      score: item.score || Number((0.96 - index * 0.04).toFixed(2)),
    })),
    graph: illustrativeDemoGraph(question, demo.evidence, false, demo.path),
    debug: demo.debug || {},
    domain: demo.domain,
    regulations: demo.regulations,
    verdict: demo.verdict || buildVerdict(question, demo.evidence, demo.domain),
    metrics: responseSource === "cached_fallback" ? {} : { evidencePassages: demo.evidence.length, searchTimeMs: 360, routingMs: 24, retrievalMs: 140, graphQueryMs: 80, rankingMs: 48, generationMs: 108, totalMs: 390 },
    usedCache,
    responseSource,
  };
}

function illustrativeDemoGraph(question, evidence, failed, preferredPath) {
  if (failed) {
    return errorGraph(question);
  }

  const path = normalizeGraphPath(preferredPath || inferPath(question, evidence));
  if (isRiskGraphPath(question, path)) return buildRiskGraphJourney();
  const nodes = path.slice(0, 12).map((label, index) => ({
    id: `n${index + 1}`,
    label: shortGraphLabel(label, index),
    type: nodeTypeFor(label, index),
  }));
  const edges = nodes.slice(0, -1).map((node, index) => ({
    source: node.id,
    target: nodes[index + 1].id,
    label: edgeLabelFor(index),
  }));
  const advancedNodes = buildAdvancedNodes(evidence);
  const advancedEdges = advancedNodes.map((node, index) => ({
    source: nodes[Math.min(index + 2, nodes.length - 1)]?.id || nodes.at(-1).id,
    target: node.id,
    label: "supported by",
  }));

  return { nodes, edges, advancedNodes, advancedEdges, meta: { source: "illustrative_demo_graph" } };
}

function errorGraph(question) {
  return {
    nodes: [
      { id: "question", label: question || "Question", type: "question" },
      { id: "error", label: "Live retrieval unavailable", type: "risk" },
    ],
    edges: [{ source: "question", target: "error", label: "error", relationship_type: "UI_STATE" }],
    advancedNodes: [],
    advancedEdges: [],
    meta: { source: "error_state" },
  };
}

function isRiskGraphPath(question, path) {
  const text = `${question} ${path.join(" ")}`.toLowerCase();
  return text.includes("deployment") || (text.includes("risk management") && text.includes("privacy controls"));
}

function buildRiskGraphJourney() {
  const nodeDefs = [
    ["mental-health-ai", "Mental Health AI", "system"],
    ["health-data", "Health Data", "concept"],
    ["high-risk-ai", "High-Risk AI System", "concept"],
    ["gdpr", "GDPR", "regulation"],
    ["eu-ai-act", "EU AI Act", "regulation"],
    ["article-9", "Article 9", "evidence"],
    ["privacy-controls", "Privacy Controls", "control"],
    ["risk-management", "Risk Management", "control"],
    ["human-oversight", "Human Oversight", "control"],
    ["technical-documentation", "Technical Documentation", "control"],
    ["gdpr-article-35", "GDPR Article 35", "evidence"],
    ["ai-act-articles-9-15", "AI Act Articles 9-15", "evidence"],
  ];
  const nodes = nodeDefs.map(([id, label, type]) => ({
    id,
    label,
    type,
  }));
  const edges = [
    ["mental-health-ai", "health-data", "processes"],
    ["health-data", "gdpr", "regulated by"],
    ["gdpr", "article-9", "cites"],
    ["gdpr", "privacy-controls", "requires"],
    ["privacy-controls", "gdpr-article-35", "cites"],
    ["mental-health-ai", "high-risk-ai", "classified as"],
    ["high-risk-ai", "eu-ai-act", "regulated by"],
    ["eu-ai-act", "risk-management", "requires"],
    ["eu-ai-act", "human-oversight", "requires"],
    ["eu-ai-act", "technical-documentation", "requires"],
    ["risk-management", "ai-act-articles-9-15", "cites"],
  ].map(([source, target, label]) => ({ source, target, label, relationship_type: "ILLUSTRATIVE" }));
  const advancedNodes = [
    { id: "a1", label: "Audit Logs", type: "Control" },
    { id: "a2", label: "Drift Testing", type: "Risk" },
    { id: "a3", label: "DPIA Record", type: "Evidence" },
    { id: "a4", label: "Access Controls", type: "Control" },
  ];
  const advancedEdges = [
    { source: "privacy-controls", target: "a4", label: "requires", relationship_type: "ILLUSTRATIVE" },
    { source: "monitoring", target: "a2", label: "requires", relationship_type: "ILLUSTRATIVE" },
    { source: "gdpr-article-35", target: "a3", label: "cites", relationship_type: "ILLUSTRATIVE" },
    { source: "technical-documentation", target: "a1", label: "supports", relationship_type: "ILLUSTRATIVE" },
  ];
  return { nodes, edges, advancedNodes, advancedEdges, meta: { source: "illustrative_demo_graph" } };
}

function inferPath(question, evidence) {
  const regulations = inferRegulations(question, evidence);
  const concepts = evidence.map((item) => item.concept).filter(Boolean).slice(0, 3);
  return [
    "User question",
    detectDomain(question),
    regulations.join(" + ") || "Healthcare regulation",
    ...(concepts.length ? concepts : ["Relevant obligations"]),
    "Evidence passages",
    "Compliance assessment",
  ].filter(Boolean);
}

function buildAdvancedNodes(evidence) {
  return evidence.slice(0, 8).map((item, index) => ({
    id: `a${index + 1}`,
    label: shortGraphLabel(item.reference, index),
    type: "Evidence",
  }));
}

function render() {
  applyLayoutPrefs();
  renderQuestionCards();
  renderStepper();
  renderGraph();
  renderTechnicalDetails();
  applySectionStates();
  renderAnswer();
  els.askButton.disabled = state.isLoading;
  updateCharacterCount();
  els.sessionState.textContent = state.isLoading ? "Running pipeline" : state.questions.length ? "Session active" : "Ready";
  updateSectionToggles();
  els.composerSourceBadge = document.querySelector("#composer-source-badge");
  if (els.composerSourceBadge) {
    const source = composerSourceState();
    els.composerSourceBadge.textContent = source.label;
    els.composerSourceBadge.className = `active-response-state source-${slug(source.key)}`;
  }
  if (els.composerSourceNote) els.composerSourceNote.textContent = composerSourceNote();
}

function updateCharacterCount() {
  if (!els.charCount) return;
  els.charCount.textContent = `${els.input?.value?.length || 0} / 500`;
}

function composerSourceState() {
  if (state.isLoading) return { key: "live", label: "Live backend running" };
  const selected = getSelectedQuestion();
  if (selected?.status === "error") return { key: selected.responseSource || "error", label: responseSourceLabel(selected) };
  if (selected?.status === "complete" && !els.input?.value?.trim()) return { key: selected.responseSource || "live", label: responseSourceLabel(selected) };
  const question = els.input?.value?.trim() || getSelectedQuestion()?.question || "";
  const policy = buildRequestPolicy(question || "custom");
  return policy.useCache ? { key: "cached_fallback", label: "Pre-loaded demo response — not live retrieval" } : { key: "live", label: "Live backend request" };
}

function composerSourceText() {
  return composerSourceState().label;
}

function composerSourceNote() {
  const selected = getSelectedQuestion();
  if (!selected) return "Ask a question to start";
  if (selected.status === "complete") return selected.responseSource === "cached_fallback" ? "Fallback shown after live failure" : "Backend /ask completed";
  if (selected.status === "error") return selected.responseSource === "timeout" ? "Timed out after configured limit" : "Backend run failed";
  return "Current run only";
}

function renderQuestionCards() {
  if (!state.questions.length) {
    els.questionCards.innerHTML = `
      <div class="empty-card">
        <strong>No questions yet</strong>
        <p>Choose a demo question or ask your own to populate the graph journey and evidence panel.</p>
      </div>
    `;
    return;
  }

  els.questionCards.innerHTML = state.questions.map((item, index) => {
    const isSelected = item.id === state.selectedId;
    const isExpanded = item.expanded || isSelected;
    const statusLabel = statusLabelForQuestion(item);
    return `
      <article class="question-card ${isSelected ? "selected" : ""} ${item.status}" data-question-id="${item.id}" ${titleIfTruncated(item.question, 72)}>
        <header>
          <span>Q${index + 1}</span>
          <strong ${titleIfTruncated(item.question, 72)}>${escapeHtml(truncate(item.question, 72))}</strong>
          <em>${statusLabel}</em>
        </header>
        ${isExpanded ? `
          <div class="question-card-body">
            <p>${escapeHtml(item.domain || "Healthcare AI compliance")}</p>
            <div class="mini-badges">${renderRegulationBadges(item.regulations)}</div>
            <small>${responseSourceLabel(item)}${item.status === "complete" ? ` · ${formatQuestionMeta(item)}` : ""}</small>
          </div>
        ` : ""}
      </article>
    `;
  }).join("");
}

function renderStepper() {
  const selected = getSelectedQuestion();
  if (!selected) {
    els.stepper.innerHTML = `<div class="journey-empty">Choose a demo preset or ask a question to start the reasoning journey.</div>`;
    return;
  }

  els.stepper.innerHTML = PIPELINE_STEPS.map((label, index) => {
    const status = selected.status === "error" && index === selected.pipelineStep
      ? "error"
      : index < selected.pipelineStep || selected.status === "complete"
        ? "complete"
        : index === selected.pipelineStep
          ? "active"
          : "pending";
    const marker = status === "complete" ? "✓" : status === "error" ? "!" : index + 1;
    return `
      <div class="step ${status}">
        <span>${marker}</span>
        <strong>${escapeHtml(label)}</strong>
        <small>${escapeHtml(stepDescription(label, selected))}</small>
      </div>
    `;
  }).join("");
}

function renderGraph() {
  const selected = getSelectedQuestion();
  if (!selected) {
    els.graphStage.innerHTML = `
      <div class="graph-welcome-card">
        <strong>Ready to inspect healthcare AI compliance</strong>
        <p>Choose a demo preset or ask a question to generate a graph path, compliance assessment, and regulatory sources.</p>
      </div>
    `;
    return;
  }

  const graph = selected.graph || evidenceGraphFromRetrievedEvidence(selected.question, selected.evidence || []);
  if (isNoEvidenceResponse(selected) || graph?.meta?.source === "no_evidence") {
    els.graphStage.innerHTML = `
      <div class="graph-meta">
        <span>0 highlighted nodes</span>
        <span>0 highlighted links</span>
        <span>Evidence missing</span>
      </div>
      <div class="graph-welcome-card evidence-missing">
        <strong>No evidence graph available</strong>
        <p>${escapeHtml(graph?.meta?.message || "No sufficient regulatory evidence was found in the knowledge graph for this question.")}</p>
      </div>
    `;
    return;
  }
  const illustrative = isIllustrativeGraph(graph);
  const technicalGraph = state.ui.graphMode === "technical" && !illustrative;
  const rawNodes = normalizeGraphNodes(technicalGraph ? [...(graph.nodes || []), ...(graph.advancedNodes || [])] : graph.nodes || []);
  const rawEdges = normalizeGraphEdges(technicalGraph ? [...(graph.edges || []), ...(graph.advancedEdges || [])] : graph.edges || []);
  const expandedGraph = splitOverloadedRegulationNodes(rawNodes, rawEdges);
  const allNodes = dedupeGraphNodes(expandedGraph.nodes).map((node) => ({ ...node, preferredLayer: graphLayer(node) }));
  const allEdges = dedupeGraphEdges(expandedGraph.edges);
  const journey = visibleJourneyGraph(allNodes, allEdges, technicalGraph);
  const nodes = journey.nodes;
  const visibleNodeIds = new Set(nodes.map((node) => node.id));
  const edges = journey.edges.filter((edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target));
  const canvas = graphCanvasSize();
  const layout = layoutGraph(nodes, selected, canvas);
  const byId = Object.fromEntries(layout.map((node) => [node.id, node]));

  const edgeSvg = edges.map((edge) => {
    const source = byId[edge.source];
    const target = byId[edge.target];
    if (!source || !target) return "";
    const path = graphEdgePath(source, target);
    const edgeLabel = visibleEdgeLabel(edge, source, target, technicalGraph);
    const labelPoint = edgeLabelPoint(source, target, edgeLabel);
    const labelWidth = Math.min(Math.max(edgeLabel.length * 7 + 18, 58), 132);
    const label = edgeLabel ? `
      <g class="edge-label">
        <title>${escapeHtml(edgeTooltip(edge))}</title>
        <rect x="${labelPoint.x - labelWidth / 2}" y="${labelPoint.y - 11}" width="${labelWidth}" height="19" rx="7"></rect>
        <text x="${labelPoint.x}" y="${labelPoint.y - 1}">${escapeHtml(edgeLabel)}</text>
      </g>` : "";
    return `
      <g class="graph-edge ${edgeLabel ? "primary-edge" : "secondary-edge"}" data-edge-source="${escapeHtml(edge.source)}" data-edge-target="${escapeHtml(edge.target)}">
      <path marker-end="url(#arrowhead)" d="${path}"><title>${escapeHtml(edgeTooltip(edge))}</title></path>
      ${label}
      </g>
    `;
  }).join("");

  const nodeSvg = layout.map((node, index) => {
    const isPrimary = isPrimaryGraphNode(selected, node, index, technicalGraph);
    const width = graphNodeWidth(node);
    const height = graphNodeHeight(node);
    const step = isPrimary ? `<text class="node-step" x="${-width / 2 + 16}" y="-18">${primaryStepNumber(layout, selected, index)}</text>` : "";
    return `
    <g class="graph-node graph-${slug(node.type)} ${isPrimary ? "primary" : "secondary"}" data-node-id="${escapeHtml(node.id)}" transform="translate(${node.x}, ${node.y})">
      <title>${escapeHtml(nodeTooltip(node, graph))}</title>
      <rect x="${-width / 2}" y="${-height / 2}" width="${width}" height="${height}" rx="12"></rect>
      <text text-anchor="middle">${svgLines(node.label, width)}</text>
      ${step}
      <text class="node-type" y="${height / 2 + 17}" text-anchor="middle">${escapeHtml(node.type)}</text>
    </g>
  `;
  }).join("");
  const viewport = graphViewport(layout, canvas);

  els.graphStage.innerHTML = `
    <div class="graph-meta">
      <span>${allNodes.length} retrieved nodes${nodes.length < allNodes.length ? ` · ${nodes.length} shown` : ""}</span>
      <span>${allEdges.length} retrieved links${edges.length < allEdges.length ? ` · ${edges.length} shown` : ""}</span>
      <span>${graphModeLabel(graph, technicalGraph)}</span>
      <div class="graph-controls" aria-label="Graph controls">
        <button type="button" data-graph-action="fit">Fit graph</button>
        <button type="button" data-graph-action="zoom-in">Zoom in</button>
        <button type="button" data-graph-action="zoom-out">Zoom out</button>
        <button type="button" data-graph-action="reset-layout">Reset layout</button>
        <button type="button" data-graph-action="fullscreen">${state.ui.graphFullscreen ? "Exit fullscreen" : "Fullscreen"}</button>
      </div>
    </div>
    <div class="graph-legend" aria-label="Graph legend">
      <span><i class="legend-blue"></i>Blue = system / concept</span>
      <span><i class="legend-purple"></i>Purple = regulation / source</span>
      <span><i class="legend-orange"></i>Orange = risk</span>
      <span><i class="legend-green"></i>Green = control / evidence</span>
    </div>
    <svg viewBox="${viewport.viewBox}" style="min-width: ${viewport.width}px; min-height: ${viewport.height}px;" role="img" aria-label="Highlighted compliance knowledge graph">
      <defs>
        <marker id="arrowhead" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z"></path>
        </marker>
      </defs>
      <g id="graph-transform" transform="translate(${graphTranslateX(nodes.length)} ${graphTranslateY(nodes.length)}) scale(${state.ui.graphZoom})">
        <g class="graph-links">${edgeSvg}</g>
        <g>${nodeSvg}</g>
      </g>
    </svg>
  `;
  initializeGraphHover();
}

function initializeGraphHover() {
  if (!els.graphStage?.querySelectorAll) return;
  const nodes = [...els.graphStage.querySelectorAll(".graph-node[data-node-id]")];
  const edges = [...els.graphStage.querySelectorAll(".graph-edge[data-edge-source][data-edge-target]")];
  const clear = () => {
    els.graphStage.classList.remove("is-hovering-graph");
    nodes.forEach((node) => node.classList.remove("is-dimmed", "is-highlighted"));
    edges.forEach((edge) => edge.classList.remove("is-dimmed", "is-highlighted"));
  };

  nodes.forEach((node) => {
    node.addEventListener("mouseenter", () => {
      const id = node.dataset.nodeId;
      const connected = new Set([id]);
      edges.forEach((edge) => {
        if (edge.dataset.edgeSource === id || edge.dataset.edgeTarget === id) {
          connected.add(edge.dataset.edgeSource);
          connected.add(edge.dataset.edgeTarget);
        }
      });
      els.graphStage.classList.add("is-hovering-graph");
      nodes.forEach((candidate) => {
        const active = connected.has(candidate.dataset.nodeId);
        candidate.classList.toggle("is-highlighted", active);
        candidate.classList.toggle("is-dimmed", !active);
      });
      edges.forEach((edge) => {
        const active = edge.dataset.edgeSource === id || edge.dataset.edgeTarget === id;
        edge.classList.toggle("is-highlighted", active);
        edge.classList.toggle("is-dimmed", !active);
      });
    });
    node.addEventListener("mouseleave", clear);
  });

  edges.forEach((edge) => {
    edge.addEventListener("mouseenter", () => {
      const connected = new Set([edge.dataset.edgeSource, edge.dataset.edgeTarget]);
      els.graphStage.classList.add("is-hovering-graph");
      edges.forEach((candidate) => {
        const active = candidate === edge;
        candidate.classList.toggle("is-highlighted", active);
        candidate.classList.toggle("is-dimmed", !active);
      });
      nodes.forEach((candidate) => {
        const active = connected.has(candidate.dataset.nodeId);
        candidate.classList.toggle("is-highlighted", active);
        candidate.classList.toggle("is-dimmed", !active);
      });
    });
    edge.addEventListener("mouseleave", clear);
  });
}

function renderTechnicalDetails() {
  const selected = getSelectedQuestion();
  els.technicalDetailsDrawer.hidden = false;
  if (!selected) {
    els.technicalDetailsDrawer.innerHTML = `
      ${renderInspectorOverview(null)}
      ${renderTechnicalEmpty("Ask or select a question to inspect routing, matched entities, retrieval scores, evidence IDs, graph traversal, and timings.")}
    `;
    return;
  }
  const activeTab = state.ui.technicalDetailsTab || "matched-entities";
  els.technicalDetailsDrawer.innerHTML = `
    <header class="technical-content-header">
      <div>
        <strong>Response state</strong>
        <span>${escapeHtml(responseSourceDetailedLabel(selected))}</span>
        ${isCachedFallback(selected) ? `<small>This answer was not produced by live retrieval.</small>` : ""}
        ${isNoEvidenceResponse(selected) ? `<small>No evidence-backed answer was generated because the graph returned 0 regulatory sources.</small>` : ""}
      </div>
      <div class="technical-actions">
        ${isCachedFallback(selected) || isNoEvidenceResponse(selected) ? "" : `<button type="button" data-graph-action="technical-mode">${state.ui.graphMode === "technical" ? "Use demo graph" : "Expand graph"}</button>`}
        <button type="button" data-graph-action="fit">Fit graph</button>
      </div>
    </header>
    ${renderInspectorOverview(selected)}
    <div class="technical-tabs">
      ${TECHNICAL_TABS.map((tab) => `<button type="button" data-technical-tab="${tab.id}" class="${activeTab === tab.id ? "active" : ""}" title="${escapeHtml(tab.label)}">${tab.label}</button>`).join("")}
    </div>
    <div class="technical-tab-content">${renderTechnicalTabContent(selected, activeTab)}</div>
  `;
}

function renderInspectorOverview(selected) {
  if (!selected) {
    return `
      <section class="inspector-overview">
        <div><dt>Status</dt><dd>Ready for a live or preset question</dd></div>
        <div><dt>Route</dt><dd>Awaiting question</dd></div>
        <div><dt>Sources</dt><dd>0 selected</dd></div>
      </section>
    `;
  }
  const laws = isNoEvidenceResponse(selected)
    ? []
    : (selected.regulations || inferRegulations(selected.question, selected.evidence || []));
  return `
    <section class="inspector-overview">
      <div><dt>Status</dt><dd>${escapeHtml(responseSourceDetailedLabel(selected))}${isCachedFallback(selected) ? `<small>This answer was not produced by live retrieval.</small>` : ""}${isNoEvidenceResponse(selected) ? `<small>No evidence-backed answer was generated.</small>` : ""}</dd></div>
      <div><dt>Route</dt><dd>${escapeHtml(selected.domain || "Healthcare AI compliance")}</dd></div>
      <div><dt>Laws</dt><dd>${laws.length ? escapeHtml(laws.join(" · ")) : "No retrieved laws"}</dd></div>
      <div><dt>Sources</dt><dd>${selected.evidence?.length || 0} regulatory source${selected.evidence?.length === 1 ? "" : "s"}</dd></div>
    </section>
  `;
}

function renderTechnicalTabContent(selected, tabId) {
  if (isNoEvidenceResponse(selected)) {
    if (tabId === "pipeline-timings") {
      const timings = getPipelineTimings(selected);
      if (!timings.length) return `${renderTechnicalSummary("No evidence found", "The backend completed retrieval but found 0 regulatory sources. Answer generation was skipped.")}${renderTechnicalEmpty("No pipeline timings available for this response.")}`;
      return `${renderTechnicalSummary("No evidence found", "The backend completed retrieval but found 0 regulatory sources. Answer generation was skipped.")}<div class="technical-table timings">${timings.map((item) => `
        <div><strong>${escapeHtml(item.label)}</strong><em>${escapeHtml(item.value)}</em></div>
      `).join("")}</div>`;
    }
    return `${renderTechnicalSummary("No evidence found", "The knowledge graph returned 0 regulatory sources for this question.")}${renderTechnicalEmpty("No live evidence rows are available for this tab.")}`;
  }
  if (isCachedFallback(selected)) {
    if (tabId === "matched-entities") {
      return `${renderTechnicalSummary("Pre-loaded fallback", "This answer was not produced by live retrieval. Matched entities are inferred from the demo preset only.")}${renderChipList(getMatchedEntities(selected), "No preset concepts available for this response.")}`;
    }
    if (tabId === "evidence-ids") {
      return `${renderTechnicalSummary("Pre-loaded fallback sources", "These source labels come from the cached demo response, not live retrieval rows.")}${renderChipList(getEvidenceIds(selected), "No fallback source labels available for this response.")}`;
    }
    return `${renderTechnicalSummary("No live debug data", "The backend did not complete, so this cached fallback has no live retrieval scores, graph debug rows, or pipeline timings.")}${renderTechnicalEmpty("No live technical data available for this fallback response.")}`;
  }
  if (tabId === "matched-entities") {
    return `${renderTechnicalSummary("Which regulation route was selected", `${(selected.regulations || []).join(" · ") || "Healthcare AI route"} for ${selected.domain || "this question"}.`)}${renderChipList(getMatchedEntities(selected), "No matched entities available for this response.")}`;
  }
  if (tabId === "retrieval-scores") {
    const scores = getRetrievalScores(selected);
    if (!scores.length) return `${renderTechnicalSummary("Why this evidence was retrieved", "The backend did not expose ranked scores for this response.")}${renderTechnicalEmpty("No retrieval scores available for this response.")}`;
    return `${renderTechnicalSummary("Why this evidence was retrieved", "Top passages were selected because they matched the routed regulation and compliance concepts.")}<div class="technical-table">${scores.map((item) => `
      <div><strong>${escapeHtml(item.reference || "Evidence passage")}</strong><span>${escapeHtml(item.regulation)}</span><em title="${escapeHtml(rawScoreTooltip(item.score))}">${escapeHtml(formatScore(item.score))}</em></div>
    `).join("")}</div>`;
  }
  if (tabId === "evidence-ids") {
    return `${renderTechnicalSummary("Which sources were included", `${selected.evidence?.length || 0} regulatory source${selected.evidence?.length === 1 ? "" : "s"} are attached to this answer.`)}${renderChipList(getEvidenceIds(selected), "No evidence IDs available for this response.")}`;
  }
  if (tabId === "graph-traversal") {
    const details = getGraphTraversalDetails(selected);
    if (!details.length) return `${renderTechnicalSummary("Knowledge graph path", "No traversal relationships were exposed for this response.")}${renderTechnicalEmpty("No graph traversal details available for this response.")}`;
    return `${renderTechnicalSummary("Knowledge graph path", "The graph path links the question to regulations, concepts, controls, and evidence.")}<div class="technical-table traversal">${details.map((item) => `
      <div><strong>${escapeHtml(item.from)}</strong><span>${escapeHtml(item.relationship)}</span><strong>${escapeHtml(item.to)}</strong></div>
    `).join("")}</div>`;
  }
  if (tabId === "pipeline-timings") {
    const timings = getPipelineTimings(selected);
    if (!timings.length) return `${renderTechnicalSummary("Pipeline timings", "Timing metadata was not returned for this response.")}${renderTechnicalEmpty("No pipeline timings available for this response.")}`;
    return `${renderTechnicalSummary("Pipeline timings", "Stage timings help explain backend latency without exposing raw debug output.")}<div class="technical-table timings">${timings.map((item) => `
      <div><strong>${escapeHtml(item.label)}</strong><em>${escapeHtml(item.value)}</em></div>
    `).join("")}</div>`;
  }
  return renderTechnicalEmpty("No technical detail available for this tab.");
}

function renderTechnicalSummary(title, body) {
  return `<div class="technical-summary"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(body)}</span></div>`;
}

function renderChipList(items, emptyMessage) {
  if (!items.length) return renderTechnicalEmpty(emptyMessage);
  return `<div class="entity-chips">${items.map((item) => {
    const more = String(item).match(/^(\+\d+ more):\s*(.+)$/);
    if (more) return `<span class="more-chip" title="${escapeHtml(more[2])}">${escapeHtml(more[1])}</span>`;
    return `<span title="${escapeHtml(item)}">${escapeHtml(item)}</span>`;
  }).join("")}</div>`;
}

function renderTechnicalEmpty(message) {
  return `<div class="technical-empty">${escapeHtml(message)}</div>`;
}

function getMatchedEntities(item) {
  return technicalEntityChips(item);
}

function getRetrievalScores(item) {
  if (isCachedFallback(item)) return [];
  return (item.evidence || []).map((entry) => ({
    reference: safeEvidenceReference(entry.reference, entry.sourceDocument) || entry.statementTitle || entry.id || "Evidence passage",
    regulation: entry.regulation || "Regulation",
    score: entry.score,
  }));
}

function getEvidenceIds(item) {
  return (item.evidence || []).map((entry) => `${entry.regulation || "Source"} · ${entry.reference || entry.id || "Evidence"}`);
}

function getGraphTraversalDetails(item) {
  if (isCachedFallback(item)) return [];
  const graph = item.graph || evidenceGraphFromRetrievedEvidence(item.question, item.evidence || []);
  const byId = Object.fromEntries([...(graph.nodes || []), ...(graph.advancedNodes || [])].map((node) => [node.id, node.label || node.id]));
  return [...(graph.edges || []), ...(graph.advancedEdges || [])].map((edge) => ({
    from: byId[edge.source] || edge.source || "Source",
    relationship: shortEdgeLabel(edge.label) || "related to",
    to: byId[edge.target] || edge.target || "Target",
  }));
}

function getPipelineTimings(item) {
  if (isCachedFallback(item)) return [];
  const metrics = item.metrics || {};
  return [
    ["Routing", metrics.routingMs],
    ["Evidence search", metrics.retrievalMs || metrics.searchTimeMs],
    ["Graph query", metrics.graphQueryMs],
    ["Ranking", metrics.rankingMs],
    ["Generation", metrics.generationMs],
    ["Total", metrics.totalMs || metrics.searchTimeMs],
  ].filter(([, value]) => value !== null && value !== undefined).map(([label, value]) => ({ label, value: `${Math.round(Number(value))} ms` }));
}

function renderAnswer() {
  const selected = getSelectedQuestion();
  if (!selected) {
    els.answerContent.innerHTML = renderEmptyState("The compliance answer will appear here first, followed by regulation badges and evidence cards.");
    els.evidenceContent.innerHTML = "";
    return;
  }

  if (selected.status === "loading") {
    els.answerContent.innerHTML = `
      <div class="answer-loading">
        <strong>Generating compliance answer...</strong>
        <span></span><span></span><span></span>
      </div>
    `;
    els.evidenceContent.innerHTML = "";
    return;
  }

  if (selected.status === "error") {
    els.answerContent.innerHTML = `
      <div class="error-state">
        <strong>${escapeHtml(responseSourceLabel(selected))}</strong>
        <p>${escapeHtml(selected.error)}</p>
      </div>
    `;
    els.evidenceContent.innerHTML = "";
    return;
  }
  const selectedEvidence = selected.evidence || [];
  const sourcesCollapsed = Boolean(selected.sourcesCollapsed);
  const noEvidence = isNoEvidenceResponse(selected) || selectedEvidence.length === 0;

  if (noEvidence) {
    els.answerContent.innerHTML = `
      <article class="answer-card no-evidence-answer">
        <div class="answer-topline">
          <span class="cache-badge source-no-evidence">${escapeHtml(responseSourceCompactLabel(selected))}</span>
        </div>
        <section class="deployment-verdict evidence-missing">
          <span>Evidence-backed answer unavailable</span>
          <strong>No sufficient regulatory evidence was found in the knowledge graph for this question.</strong>
        </section>
        <div class="answer-copy">
          <p>${escapeHtml(selected.summary || selected.answer || "No sufficient regulatory evidence was found in the knowledge graph for this question.")}</p>
        </div>
      </article>
    `;
    els.evidenceContent.innerHTML = `
      <div class="section-title evidence-title">
        <div>
          <h3>Sources used</h3>
          <span>0 regulatory sources selected</span>
        </div>
        <button type="button" class="section-toggle-button" data-sources-toggle aria-expanded="${String(!sourcesCollapsed)}">${sourcesCollapsed ? "Show" : "Collapse"}</button>
      </div>
      <div class="evidence-list ${sourcesCollapsed ? "is-collapsed" : ""}" ${sourcesCollapsed ? "hidden" : ""}>
        ${renderEmptyState("No regulatory source passages were retrieved. Check the question wording or verify that Neo4j contains Regulation, Statement, and SourceChunk nodes.")}
      </div>
    `;
    return;
  }

  els.answerContent.innerHTML = `
    <article class="answer-card">
      <div class="answer-topline">
        ${renderRegulationBadges(selected.regulations)}
        <span class="cache-badge source-${slug(selected.responseSource || "live_pipeline")}">${escapeHtml(responseSourceCompactLabel(selected))}</span>
      </div>
      <section class="deployment-verdict">
        <span>Can this be deployed as-is?</span>
        <strong>${escapeHtml(deploymentVerdict(selected))}</strong>
      </section>
      ${renderVerdictBox(selected)}
      <div class="answer-copy">
        <p>${escapeHtml(selected.summary || summarizeAnswer(selected.answer, selected.question))}</p>
        <strong>Key controls</strong>
        <ul>${(selected.obligations?.length ? selected.obligations : obligationsForQuestion(selected.question)).map((item) => `<li>${inlineMarkdown(item)}</li>`).join("")}</ul>
      </div>
      ${renderFullAnswerToggle(selected)}
    </article>
  `;

  els.evidenceContent.innerHTML = `
    <div class="section-title evidence-title">
      <div>
        <h3>Sources used</h3>
        <span>${selectedEvidence.length} regulatory sources selected${selectedEvidence.length ? ` · ${escapeHtml((selected.regulations || inferRegulations(selected.question, selectedEvidence)).join(" · "))}` : ""}</span>
      </div>
      <button type="button" class="section-toggle-button" data-sources-toggle aria-expanded="${String(!sourcesCollapsed)}">${sourcesCollapsed ? "Show" : "Collapse"}</button>
    </div>
    <div class="evidence-list ${sourcesCollapsed ? "is-collapsed" : ""}" ${sourcesCollapsed ? "hidden" : ""}>
      ${selectedEvidence.map(renderEvidenceCard).join("") || renderEmptyState("No evidence passages were returned.")}
    </div>
  `;
}

function renderEvidenceCard(item) {
  const sourceDocument = item.sourceDocument || demoSourceDocument(item.regulation);
  const reference = safeEvidenceReference(item.reference, sourceDocument);
  const referenceLabel = reference || "Not available";
  const explanation = sourceShortExplanation(item);
  const snippet = item.passage || "No snippet returned.";
  return `
    <details class="evidence-card">
      <summary>
        <span class="reg-badge ${slug(item.regulation)}">${escapeHtml(item.regulation)}</span>
        <span class="source-card-copy">
          <strong ${titleIfTruncated(sourceDocument, 42)}>${escapeHtml(truncate(sourceDocument, 42))}</strong>
          <b ${titleIfTruncated(referenceLabel, 48)}>${escapeHtml(truncate(referenceLabel, 48))}</b>
          <small ${titleIfTruncated(explanation, 86)}>${escapeHtml(truncate(explanation, 86))}</small>
        </span>
      </summary>
      <dl>
        <div><dt>Source document</dt><dd>${escapeHtml(sourceDocument)}</dd></div>
        <div><dt>Regulation</dt><dd>${escapeHtml(item.regulation)}</dd></div>
        <div><dt>Section or article</dt><dd>${escapeHtml(referenceLabel)}</dd></div>
        <div><dt>Statement title</dt><dd>${escapeHtml(item.statementTitle || item.concept || "Regulatory evidence")}</dd></div>
        <div><dt>Matched concept</dt><dd>${escapeHtml(item.concept)}</dd></div>
        <div><dt>Evidence snippet</dt><dd ${titleIfTruncated(snippet, 260)}>${escapeHtml(truncate(snippet, 260))}</dd></div>
        <div><dt>Why this was used</dt><dd>${escapeHtml(item.explanation)}</dd></div>
      </dl>
    </details>
  `;
}

function getSelectedQuestion() {
  return state.questions.find((item) => item.id === state.selectedId) || state.questions.at(-1) || null;
}

function updateQuestion(id, patch, shouldRender = true) {
  state.questions = state.questions.map((item) => item.id === id ? { ...item, ...patch } : item);
  if (shouldRender) render();
}

function toggleSourcesForSelected() {
  const selected = getSelectedQuestion();
  if (!selected) return;
  updateQuestion(selected.id, { sourcesCollapsed: !Boolean(selected.sourcesCollapsed) });
  persistSession();
}

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function persistSession() {
  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(state.questions.slice(-10)));
}

function persistSelectedQuestion() {
  if (state.selectedId) localStorage.setItem(SELECTED_STORAGE_KEY, state.selectedId);
  else localStorage.removeItem(SELECTED_STORAGE_KEY);
}

function findDemoResponse(question) {
  const normalized = normalizeQuestion(question);
  return DEMO_CACHE[normalized] || null;
}

function citationFromEvidenceItem(item, sourceDocument) {
  const candidates = [
    item.citation,
    item.article,
    item.section,
    item.reference,
    item.chunk_reference,
    item.chunk_id,
  ];
  return candidates.map((value) => safeEvidenceReference(value, sourceDocument)).find(Boolean) || "";
}

function safeEvidenceReference(value, sourceDocument = "") {
  const raw = String(value || "").trim();
  if (!raw) return "";
  const source = String(sourceDocument || "").trim().toLowerCase();
  const normalized = raw.toLowerCase();
  if (source && normalized === source) return "";
  if (/^[\w.-]+\.pdf$/i.test(raw)) return "";
  if (/^unknown source document$/i.test(raw)) return "";
  return raw;
}

function detectDomain(question, plan) {
  const domains = plan?.detected_domains || [];
  if (domains.length) return domains.join(", ");
  const text = question.toLowerCase();
  if (text.includes("mental") || text.includes("diagnostic")) return "Mental health diagnostic AI";
  if (text.includes("patient") || text.includes("health data")) return "Patient health data processing";
  if (text.includes("provider") || text.includes("high-risk")) return "High-risk AI provider compliance";
  if (text.includes("rights")) return "Patient rights and transparency";
  return "Healthcare AI compliance";
}

function inferRegulations(question, evidence = []) {
  const text = `${question} ${evidence.map((item) => `${item.regulation} ${item.reference} ${item.passage}`).join(" ")}`.toLowerCase();
  const regs = [];
  if (text.includes("gdpr") || text.includes("patient") || text.includes("data subject") || text.includes("health data")) regs.push("GDPR");
  if (text.includes("hipaa") || text.includes("authorisation") || text.includes("authorization") || text.includes("phi")) regs.push("HIPAA");
  if (text.includes("ai act") || text.includes("high-risk") || text.includes("provider") || text.includes("deployment") || text.includes("diagnostic")) regs.push("EU AI Act");
  return regs.length ? regs : ["GDPR", "HIPAA", "EU AI Act"];
}

function renderRegulationBadges(regulations = []) {
  return regulations.map((reg) => `<span class="reg-badge ${slug(reg)}">${escapeHtml(reg)}</span>`).join("");
}

function renderVerdictBox(selected) {
  const verdict = selected.verdict || buildVerdict(selected.question, selected.evidence, selected.domain);
  return `
    <section class="verdict-box" aria-label="Assessment summary">
      ${verdict.useCase ? `<div><dt>Use case</dt><dd>${escapeHtml(verdict.useCase)}</dd></div>` : ""}
      <div><dt>Assessment</dt><dd>${escapeHtml(verdict.assessment)}</dd></div>
      <div><dt>Main concern</dt><dd>${escapeHtml(verdict.concern)}</dd></div>
      <div><dt>Relevant laws</dt><dd><span class="law-line">${escapeHtml(verdict.laws.join(" · "))}</span></dd></div>
    </section>
  `;
}

function renderFullAnswerToggle(selected) {
  const summary = selected.summary || summarizeAnswer(selected.answer, selected.question);
  const full = String(selected.answer || "").trim();
  if (!full || full === summary || full.length < 220) return "";
  return `
    <details class="full-answer">
      <summary>Show full answer</summary>
      <div class="rendered-markdown">${renderMarkdown(full)}</div>
    </details>
  `;
}

function renderAnswerText(text) {
  return renderMarkdown(text);
}

function renderMarkdown(text) {
  const cleaned = cleanAnswerMarkdown(text);
  const lines = cleaned.split(/\r?\n/);
  const html = [];
  let listType = null;

  const closeList = () => {
    if (!listType) return;
    html.push(`</${listType}>`);
    listType = null;
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();
    if (!line) {
      closeList();
      return;
    }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 2, 5);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      return;
    }
    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (listType !== "ul") {
        closeList();
        html.push("<ul>");
        listType = "ul";
      }
      html.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
      return;
    }
    const numbered = line.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) {
      if (listType !== "ol") {
        closeList();
        html.push("<ol>");
        listType = "ol";
      }
      html.push(`<li>${inlineMarkdown(numbered[1])}</li>`);
      return;
    }
    closeList();
    html.push(`<p>${inlineMarkdown(line)}</p>`);
  });
  closeList();
  return html.join("") || "<p>No answer generated.</p>";
}

function cleanAnswerMarkdown(text) {
  return String(text || "")
    .replace(/\r/g, "")
    .replace(/^\s*(?:below|answer|response)\s*:\s*(?=#{1,4}\s)/i, "")
    .replace(/\n\s*(?:below|answer|response)\s*:\s*(?=#{1,4}\s)/gi, "\n")
    .trim();
}

function renderEmptyState(message) {
  return `<div class="empty-card"><strong>Ready for demo</strong><p>${escapeHtml(message)}</p></div>`;
}

function stepDescription(label, item) {
  const descriptions = {
    "Question intake": truncate(item.question, 34),
    "Regulation routing": isNoEvidenceResponse(item) ? "No retrieved route" : ((item.regulations || []).join(", ") || "GDPR, HIPAA, EU AI Act"),
    "Graph traversal": relatedConcepts(item).slice(0, 2).join(", "),
    "Evidence ranking": `${item.evidence?.length || 0} sources selected`,
    "Answer synthesis": isNoEvidenceResponse(item) ? "Skipped without evidence" : item.status === "complete" ? "Grounded controls drafted" : item.status === "error" ? "Generation skipped" : "Building grounded answer",
    "Response ready": isNoEvidenceResponse(item) ? "Evidence-missing state" : item.status === "complete" ? "Visitor-ready assessment" : item.status === "error" ? "Error state shown" : "Preparing response",
  };
  return descriptions[label] || "";
}

function relatedConcepts(item) {
  if (isNoEvidenceResponse(item)) return ["no evidence found"];
  const concepts = (item.evidence || []).map((entry) => entry.concept).filter(Boolean).slice(0, 3);
  return concepts.length ? concepts : ["health data", "high-risk AI", "patient safeguards"];
}

function formatQuestionMeta(item) {
  if (item.status === "loading") return `Running: ${PIPELINE_STEPS[item.pipelineStep]}`;
  if (item.status === "error") return item.error;
  if (isNoEvidenceResponse(item)) return "0 regulatory sources selected";
  const time = item.metrics?.searchTimeMs ? `${item.metrics.searchTimeMs} ms` : "live backend";
  return `${item.evidence.length} regulatory sources selected · ${time}`;
}

function responseSourceLabel(item) {
  return responseSourceDetailedLabel(item);
}

function responseSourceCompactLabel(item) {
  const source = item.responseSource || (item.usedCache ? "cached_fallback" : "live");
  if (source === "live" || source === "live_pipeline") return "Live retrieval completed";
  if (source === "no_evidence") return "No evidence found";
  if (source === "cached_fallback" || source === "cached_demo" || source === "fallback_cache") return "Pre-loaded demo response";
  if (source === "timeout" || source === "live_timeout") return "Fallback response after timeout";
  if (source === "error" || source === "live_failed") return "Live retrieval failed";
  return "Live backend request";
}

function responseSourceDetailedLabel(item) {
  const source = item.responseSource || (item.usedCache ? "cached_fallback" : "live");
  if (source === "live" || source === "live_pipeline") return "Live retrieval completed";
  if (source === "no_evidence") return "No evidence found";
  if (source === "cached_fallback" || source === "cached_demo" || source === "fallback_cache") return "Pre-loaded demo response";
  if (source === "timeout" || source === "live_timeout") return "Fallback response after timeout";
  if (source === "error" || source === "live_failed") return "Live retrieval failed";
  return "Live backend request";
}

function highValueGraphNodes(nodes, limit) {
  const priority = { question: 0, regulation: 1, statement: 2, evidence: 3, entity: 4, ontology: 5, control: 6, risk: 6 };
  return [...nodes]
    .sort((a, b) => (priority[String(a.type || "").toLowerCase()] ?? 9) - (priority[String(b.type || "").toLowerCase()] ?? 9))
    .slice(0, limit);
}

function dedupeGraphNodes(nodes = []) {
  const byId = new Map();
  nodes.forEach((node, index) => {
    const id = String(node.id || `node-${index + 1}`);
    const existing = byId.get(id) || {};
    byId.set(id, { ...existing, ...node, id, label: node.label || existing.label || `Node ${index + 1}` });
  });
  return [...byId.values()];
}

function dedupeGraphEdges(edges = []) {
  const seen = new Set();
  return edges.filter((edge) => {
    const label = shortEdgeLabel(edge.label || edge.relationshipType || "");
    const key = `${edge.source}::${edge.target}::${label}`;
    if (!edge.source || !edge.target || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function visibleJourneyGraph(allNodes, allEdges, technicalGraph) {
  const degree = new Map();
  allEdges.forEach((edge) => {
    degree.set(edge.source, (degree.get(edge.source) || 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) || 0) + 1);
  });
  const layerCaps = technicalGraph ? { 0: 2, 1: 8, 2: 9, 3: 9, 4: 6 } : { 0: 1, 1: 5, 2: 6, 3: 6, 4: 4 };
  const visibleNodes = [];
  const hiddenGroups = [];
  const layers = new Map();

  allNodes.forEach((node) => {
    const layer = graphLayer(node);
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push({ ...node, preferredLayer: layer });
  });

  [...layers.keys()].sort((a, b) => a - b).forEach((layer) => {
    const group = layers.get(layer)
      .sort((a, b) => graphNodeRank(a, degree) - graphNodeRank(b, degree) || String(a.label).localeCompare(String(b.label)));
    const cap = layerCaps[layer] || (technicalGraph ? 7 : 5);
    const shown = group.slice(0, cap);
    const hidden = group.slice(cap);
    visibleNodes.push(...shown);
    if (hidden.length) {
      hiddenGroups.push({
        id: `group:${layer}`,
        label: `+ ${hidden.length} more ${journeyLayerName(layer)} node${hidden.length === 1 ? "" : "s"}`,
        type: "group",
        preferredLayer: layer,
        hiddenLabels: hidden.map((node) => node.label).join("\n"),
      });
    }
  });

  const nodes = [...visibleNodes, ...hiddenGroups];
  const visibleIds = new Set(nodes.map((node) => node.id));
  const edges = allEdges.filter((edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target));
  return { nodes, edges };
}

function graphNodeRank(node, degree) {
  const type = String(node.type || "").toLowerCase();
  const priority = { question: 0, entity: 1, concept: 1, system: 1, ontology: 2, statement: 3, risk: 3, control: 3, evidence: 4, regulation: 5, group: 9 };
  const degreeScore = -(degree.get(node.id) || 0);
  const evidenceScore = Number(node.evidenceId || 9999);
  return (priority[type] ?? 6) * 1000 + degreeScore * 20 + Math.min(evidenceScore, 999);
}

function journeyLayerName(layer) {
  return {
    0: "question",
    1: "concept",
    2: "requirement",
    3: "evidence",
    4: "source",
  }[layer] || "graph";
}

function isIllustrativeGraph(graph) {
  return String(graph?.meta?.source || "").includes("illustrative");
}

function graphModeLabel(graph, technicalGraph) {
  if (graph?.meta?.source === "no_evidence") return "Evidence missing";
  if (isIllustrativeGraph(graph)) return "Illustrative graph";
  if (technicalGraph) return "Technical graph";
  return "Retrieved evidence graph";
}

function buildRequestPolicy(question, config = EVENT_DEMO_CONFIG) {
  const isDemoPreset = isDemoPresetQuestion(question);
  const cacheHit = Boolean(findDemoResponse(question));
  const cacheEnabled = Boolean(config.EVENT_DEMO_MODE && config.DEMO_CACHE_ENABLED);
  const fallbackEnabled = Boolean(config.EVENT_DEMO_MODE && config.DEMO_FALLBACK_ENABLED);
  const useCache = Boolean(cacheEnabled && isDemoPreset && cacheHit);
  const livePipelineCalled = !useCache && (!isDemoPreset || Boolean(config.LIVE_CUSTOM_QUESTIONS_ENABLED));
  const timeoutMs = normalizeTimeoutMs(
    isDemoPreset ? config.DEMO_PRESET_TIMEOUT_MS : config.CUSTOM_QUESTION_TIMEOUT_MS,
    isDemoPreset ? 8000 : 60000,
  );
  return {
    question,
    isDemoPreset,
    cacheEnabled,
    fallbackEnabled,
    cacheHit,
    timeoutMs,
    useCache,
    livePipelineCalled,
  };
}

function isDemoPresetQuestion(question) {
  const normalized = normalizeQuestion(question);
  return DEMO_QUESTIONS.some((demoQuestion) => normalizeQuestion(demoQuestion) === normalized);
}

function normalizeTimeoutMs(value, fallbackMs) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) return fallbackMs;
  if (parsed < 1000) return parsed * 1000;
  return parsed;
}

function logRequestPolicy(policy) {
  console.info("[KEP demo request]", {
    question: policy.question,
    isDemoPreset: policy.isDemoPreset,
    cacheEnabled: policy.cacheEnabled,
    fallbackEnabled: policy.fallbackEnabled,
    cacheHit: policy.cacheHit,
    timeoutMs: policy.timeoutMs,
    useCache: policy.useCache,
    livePipelineCalled: policy.livePipelineCalled,
  });
}

function isCachedFallback(item) {
  const source = item?.responseSource;
  return source === "cached_fallback" || source === "cached_demo" || source === "fallback_cache";
}

function isNoEvidenceResponse(item) {
  return item?.responseSource === "no_evidence" || item?.metrics?.noEvidence === true || item?.metrics?.no_evidence === true;
}

function statusLabelForQuestion(item) {
  if (item.status === "loading") return "Running";
  if (item.status === "error") return "Review needed";
  if (isNoEvidenceResponse(item)) return "No evidence";
  const text = item.question.toLowerCase();
  if (text.includes("risk") || text.includes("deployment")) return "Risk flagged";
  if (text.includes("mental") || text.includes("diagnostic")) return "High sensitivity";
  return "Answered";
}

function deploymentVerdict(selected) {
  const text = `${selected.question} ${selected.domain}`.toLowerCase();
  if (text.includes("mental") && text.includes("diagnostic")) return "High compliance sensitivity — evidence review required.";
  if (text.includes("risk") || text.includes("deployment")) return "Not yet — controls required before deployment.";
  if (text.includes("health data") || text.includes("authorisation") || text.includes("authorization")) return "Review needed before deployment.";
  return "Review needed before deployment.";
}

function sourceShortExplanation(item) {
  const text = `${item.regulation} ${item.reference} ${item.concept}`.toLowerCase();
  if (text.includes("article 35")) return "Data protection impact assessment for high-risk processing";
  if (text.includes("security rule") || text.includes("164.308") || text.includes("164.312")) return "Administrative and technical safeguards where applicable";
  if (text.includes("9 and 15")) return "Risk management and post-market monitoring";
  if (text.includes("9-15") || text.includes("9–15")) return "Risk management, documentation, and oversight duties";
  if (text.includes("annex iv")) return "Technical file and validation evidence";
  if (text.includes("article 9")) return "Special-category health data safeguards";
  if (text.includes("article 6")) return "Lawful basis for personal data processing";
  return item.concept || "Regulatory evidence used for this assessment";
}

function demoSourceDocument(regulation) {
  if (regulation === "GDPR") return "GDPR regulatory evidence";
  if (regulation === "HIPAA") return "HIPAA regulatory evidence";
  if (regulation === "EU AI Act") return "EU AI Act regulatory evidence";
  return "Regulatory evidence";
}

function isPrimaryGraphNode(selected, node, index, advanced) {
  if (advanced && node.id?.startsWith("a")) return false;
  const text = selected.question.toLowerCase();
  if (text.includes("risk") || text.includes("deployment")) {
    return ["Mental Health AI", "Health Data", "GDPR", "Privacy Controls", "High-Risk AI System", "EU AI Act", "Risk Management"].includes(node.label);
  }
  return index < 6;
}

function primaryStepNumber(nodes, selected, index) {
  const primaryBefore = nodes.slice(0, index + 1).filter((node, idx) => isPrimaryGraphNode(selected, node, idx, false));
  return primaryBefore.length;
}

function graphTranslateX() {
  return 0;
}

function graphTranslateY() {
  return 0;
}

function graphCanvasSize() {
  const stageWidth = els.graphStage?.clientWidth || 0;
  const stageHeight = els.graphStage?.clientHeight || 0;
  const width = state.ui.graphFullscreen
    ? Math.max(window.innerWidth - 72, 1280)
    : Math.max(stageWidth, 1060);
  const height = state.ui.graphFullscreen
    ? Math.max(window.innerHeight - 170, 760)
    : Math.max(stageHeight - 98, 520);
  return {
    width,
    height,
    topPadding: state.ui.graphFullscreen ? 96 : 72,
    bottomPadding: state.ui.graphFullscreen ? 110 : 88,
    sidePadding: state.ui.graphFullscreen ? 92 : 70,
  };
}

function graphViewport(nodes, canvas = graphCanvasSize()) {
  if (!nodes.length) return { viewBox: `0 0 ${canvas.width} ${canvas.height}`, width: canvas.width, height: canvas.height };
  const padding = state.ui.graphFullscreen ? 96 : 76;
  const extents = nodes.reduce((acc, node) => {
    const width = graphNodeWidth(node);
    const height = graphNodeHeight(node);
    acc.minX = Math.min(acc.minX, node.x - width / 2);
    acc.maxX = Math.max(acc.maxX, node.x + width / 2);
    acc.minY = Math.min(acc.minY, node.y - height / 2 - 28);
    acc.maxY = Math.max(acc.maxY, node.y + height / 2 + 44);
    return acc;
  }, { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
  const width = Math.max(canvas.width, Math.ceil(extents.maxX - extents.minX + padding * 2));
  const height = Math.max(canvas.height, Math.ceil(extents.maxY - extents.minY + padding * 2));
  const minX = Math.min(0, Math.floor(extents.minX - padding));
  const minY = Math.min(0, Math.floor(extents.minY - padding));
  return {
    viewBox: `${minX} ${minY} ${width} ${height}`,
    width: Math.ceil(width * state.ui.graphZoom),
    height: Math.ceil(height * state.ui.graphZoom),
  };
}

function toggleSection(section) {
  if (!["graph", "technical"].includes(section)) return;
  if (section === "technical") {
    state.ui.technicalDetailsOpen = !state.ui.technicalDetailsOpen;
  } else {
    state.ui.collapsedSections = { ...DEFAULT_UI_PREFS.collapsedSections, ...(state.ui.collapsedSections || {}) };
    state.ui.collapsedSections[section] = !state.ui.collapsedSections[section];
    if (section === "graph" && state.ui.collapsedSections.graph) state.ui.graphFullscreen = false;
  }
  persistUiPrefs();
  render();
}

function isSectionCollapsed(section) {
  if (section === "technical") return !state.ui.technicalDetailsOpen;
  if (section !== "graph") return false;
  return Boolean(state.ui.collapsedSections?.[section]);
}

function applySectionStates() {
  if (typeof document.querySelectorAll !== "function") return;
  document.querySelectorAll("[data-section-block]").forEach((block) => {
    const section = block.dataset.sectionBlock;
    const collapsed = isSectionCollapsed(section);
    block.classList.toggle("is-collapsed", collapsed);
    block.classList.toggle("is-expanded", !collapsed);
    block.querySelectorAll(".block-body").forEach((body) => {
      body.hidden = collapsed;
    });
  });
  els.graphStage?.classList?.toggle("is-fullscreen", state.ui.graphFullscreen);
}

function updateSectionToggles() {
  if (typeof document.querySelectorAll !== "function") return;
  document.querySelectorAll("[data-section-toggle]").forEach((button) => {
    const section = button.dataset.sectionToggle;
    const collapsed = isSectionCollapsed(section);
    button.setAttribute("aria-expanded", String(!collapsed));
    button.textContent = section === "technical"
      ? collapsed ? "Show" : "Hide"
      : collapsed ? "Expand" : "Collapse";
    button.title = `${button.textContent} ${section.replace("-", " ")}`;
  });
}

function handleGraphAction(action) {
  if (action === "toggle-details") {
    toggleSection("technical");
    return;
  }
  if (action === "fit") fitGraph();
  if (action === "zoom-in") state.ui.graphZoom = Math.min(1.8, Number((state.ui.graphZoom + 0.12).toFixed(2)));
  if (action === "zoom-out") state.ui.graphZoom = Math.max(0.72, Number((state.ui.graphZoom - 0.12).toFixed(2)));
  if (action === "reset-layout") {
    state.ui.graphZoom = 1;
  }
  if (action === "fullscreen") {
    state.ui.graphFullscreen = !state.ui.graphFullscreen;
    if (state.ui.graphFullscreen) {
      state.ui.collapsedSections = { ...DEFAULT_UI_PREFS.collapsedSections, ...(state.ui.collapsedSections || {}), graph: false };
    }
  }
  if (action === "technical-mode") {
    state.ui.graphMode = state.ui.graphMode === "technical" ? "demo" : "technical";
    state.ui.graphZoom = 1;
  }
  persistUiPrefs();
  render();
  if (["fullscreen", "fit", "reset-layout", "technical-mode"].includes(action)) {
    const raf = typeof requestAnimationFrame === "function" ? requestAnimationFrame : (callback) => setTimeout(callback, 0);
    raf(() => {
      renderGraph();
      setTimeout(renderGraph, 60);
    });
  }
}

function fitGraph() {
  state.ui.graphZoom = 1;
  persistUiPrefs();
  renderGraph();
}

function setupResizeHandles() {
  if (!document.querySelectorAll) return;
  document.querySelectorAll("[data-resize-handle]").forEach((handle) => {
    handle.addEventListener("dblclick", () => {
      state.ui.leftWidth = DEFAULT_UI_PREFS.leftWidth;
      state.ui.middleWidth = DEFAULT_UI_PREFS.middleWidth;
      state.ui.rightWidth = DEFAULT_UI_PREFS.rightWidth;
      persistUiPrefs();
      applyLayoutPrefs();
      fitGraph();
    });

    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const side = handle.dataset.resizeHandle;
      const startX = event.clientX;
      const startLeft = state.ui.leftWidth;
      const startRight = state.ui.rightWidth;
      document.body.classList.add("is-resizing");

      const onMove = (moveEvent) => {
        const delta = moveEvent.clientX - startX;
        const bounds = els.demoGrid.getBoundingClientRect();
        const maxSide = Math.max(320, bounds.width - 650 - 48);
        if (side === "left") {
          state.ui.leftWidth = clampNumber(startLeft + delta, 240, Math.min(520, maxSide));
        }
        if (side === "right") {
          state.ui.rightWidth = clampNumber(startRight - delta, 320, Math.min(620, maxSide));
        }
        state.ui.middleWidth = Math.max(650, bounds.width - effectiveLeftWidth() - state.ui.rightWidth - 28);
        applyLayoutPrefs();
      };

      const onUp = () => {
        document.body.classList.remove("is-resizing");
        persistUiPrefs();
        fitGraph();
        window.removeEventListener("pointermove", onMove);
        window.removeEventListener("pointerup", onUp);
      };

      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
  });
}

function applyLayoutPrefs() {
  const leftWidth = effectiveLeftWidth();
  const rightWidth = clampNumber(Number(state.ui.rightWidth) || DEFAULT_UI_PREFS.rightWidth, 320, 620);
  els.demoGrid?.style?.setProperty("--left-width", `${leftWidth}px`);
  els.demoGrid?.style?.setProperty("--right-width", `${rightWidth}px`);
  els.graphStage?.classList?.toggle("is-fullscreen", state.ui.graphFullscreen);
}

function effectiveLeftWidth() {
  return clampNumber(Number(state.ui.leftWidth) || DEFAULT_UI_PREFS.leftWidth, 260, 520);
}

function readUiPrefs() {
  try {
    const raw = { ...DEFAULT_UI_PREFS, ...JSON.parse(localStorage.getItem(UI_STORAGE_KEY) || "{}") };
    const tabIds = TECHNICAL_TABS.map((tab) => tab.id);
    const collapsedSections = { ...DEFAULT_UI_PREFS.collapsedSections, ...(raw.collapsedSections || {}) };
    return {
      ...raw,
      sidebarCollapsed: false,
      leftWidth: clampNumber(Number(raw.leftWidth) || DEFAULT_UI_PREFS.leftWidth, 260, 520),
      rightWidth: clampNumber(Number(raw.rightWidth) || DEFAULT_UI_PREFS.rightWidth, 320, 620),
      technicalDetailsOpen: Boolean(raw.technicalDetailsOpen),
      technicalDetailsTab: tabIds.includes(raw.technicalDetailsTab) ? raw.technicalDetailsTab : "matched-entities",
      collapsedSections,
      graphMode: raw.graphMode === "technical" ? "technical" : "demo",
      graphZoom: clampNumber(Number(raw.graphZoom) || 1, 0.72, 1.8),
      graphFullscreen: Boolean(raw.graphFullscreen),
    };
  } catch {
    return { ...DEFAULT_UI_PREFS };
  }
}

function persistUiPrefs() {
  localStorage.setItem(UI_STORAGE_KEY, JSON.stringify(state.ui));
}

function clampNumber(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function layoutGraph(nodes, selected, canvas = graphCanvasSize()) {
  const illustrativeRisk = isIllustrativeGraph(selected?.graph) && (
    `${selected?.question || ""} ${selected?.domain || ""}`.toLowerCase().includes("risk")
    || `${selected?.question || ""}`.toLowerCase().includes("deployment")
  );
  if (illustrativeRisk) {
    return storyGraphLayout(nodes, canvas);
  }
  return layeredGraphLayout(nodes, canvas);
}

function storyGraphLayout(nodes, canvas) {
  const layerByLabel = {
    "Mental Health AI": 0,
    "Health Data": 1,
    "High-Risk AI System": 1,
    GDPR: 4,
    "EU AI Act": 4,
    "Article 9": 3,
    "Privacy Controls": 2,
    "Human Oversight": 2,
    "Risk Management": 2,
    "Technical Documentation": 2,
    "GDPR Article 35": 3,
    "AI Act Articles 9-15": 3,
  };
  return layeredGraphLayout(nodes.map((node) => ({ ...node, preferredLayer: layerByLabel[node.label] ?? graphLayer(node) })), canvas);
}

function layeredGraphLayout(nodes, canvas) {
  const layers = new Map();
  nodes.forEach((node) => {
    const layer = Number.isFinite(node.preferredLayer) ? node.preferredLayer : graphLayer(node);
    if (!layers.has(layer)) layers.set(layer, []);
    layers.get(layer).push(node);
  });
  const sortedLayers = [0, 1, 2, 3, 4].filter((layer) => layers.has(layer));
  const left = canvas.sidePadding;
  const right = Math.max(left + 1, canvas.width - canvas.sidePadding);
  const top = canvas.topPadding;
  const maxGroupSize = Math.max(...[...layers.values()].map((group) => group.length), 1);
  const requiredLayerHeight = (maxGroupSize + 1) * 118;
  const bottom = Math.max(top + requiredLayerHeight, canvas.height - canvas.bottomPadding);
  const columnFractions = { 0: 0.02, 1: 0.26, 2: 0.50, 3: 0.73, 4: 0.95 };
  return sortedLayers.flatMap((layer, layerIndex) => {
    const group = layers.get(layer).sort((a, b) => graphNodeRank(a, new Map()) - graphNodeRank(b, new Map()) || String(a.label).localeCompare(String(b.label)));
    const minGap = group.some((node) => ["regulation", "evidence"].includes(String(node.type).toLowerCase())) ? 124 : 112;
    const yGap = Math.max(minGap, (bottom - top) / Math.max(group.length + 1, 2));
    const groupHeight = (group.length - 1) * yGap;
    const firstY = Math.max(top + yGap, (top + bottom) / 2 - groupHeight / 2);
    const x = left + (right - left) * (columnFractions[layer] ?? (layerIndex / Math.max(sortedLayers.length - 1, 1)));
    return group.map((node, index) => ({
      ...node,
      x,
      y: clampNumber(firstY + index * yGap + (group.length > 1 && layerIndex % 2 ? yGap * 0.08 : 0), top + 54, bottom - 54),
    }));
  });
}

function graphLayer(node) {
  const type = String(node.type || "").toLowerCase();
  const label = String(node.label || "").toLowerCase();
  if (type === "question") return 0;
  if (type === "system" || type === "concept" || type === "entity" || type === "ontology") return 1;
  if (type === "risk" || type === "control" || type === "statement" || label.includes("requirement") || label.includes("obligation")) return 2;
  if (type === "evidence" || type === "citation" || isCitationLabel(node.label) || isCitationLabel(node.citation)) return 3;
  if (type === "regulation" || type === "source" || isRegulationOrSourceLabel(node.label) || isRegulationOrSourceLabel(node.sourceDocument)) return 4;
  if (type === "group") return Number.isFinite(node.preferredLayer) ? node.preferredLayer : 3;
  return 2;
}

function graphNodeWidth(node) {
  const label = String(node.label || "");
  const type = String(node.type || "").toLowerCase();
  const base = type === "question" ? 168 : type === "regulation" ? 150 : type === "evidence" ? 164 : 156;
  return clampNumber(base + Math.min(label.length, 34) * 2.8, 150, 224);
}

function graphNodeHeight(node) {
  const type = String(node.type || "").toLowerCase();
  return type === "group" ? 64 : 82;
}

function graphEdgePath(source, target) {
  const sourceWidth = graphNodeWidth(source);
  const targetWidth = graphNodeWidth(target);
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const startX = source.x + Math.sign(dx || 1) * sourceWidth / 2;
  const endX = target.x - Math.sign(dx || 1) * targetWidth / 2;
  const startY = source.y;
  const endY = target.y;
  if (Math.abs(dy) < 36) {
    return `M ${startX} ${startY} L ${endX} ${endY}`;
  }
  const curve = Math.min(Math.abs(dx) * 0.16, 56);
  const c1x = startX + Math.sign(dx || 1) * curve;
  const c2x = endX - Math.sign(dx || 1) * curve;
  return `M ${startX} ${startY} C ${c1x} ${startY}, ${c2x} ${endY}, ${endX} ${endY}`;
}

function edgeLabelPoint(source, target, label = "") {
  const midX = (source.x + target.x) / 2;
  const midY = (source.y + target.y) / 2;
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const length = Math.max(Math.hypot(dx, dy), 1);
  const touchesRegulation = source.type === "regulation" || target.type === "regulation";
  const offset = !label ? 0 : touchesRegulation ? (Math.abs(dy) < 40 ? -36 : 32) : (Math.abs(dy) < 40 ? -24 : 20);
  const pullFromTarget = touchesRegulation && target.type === "regulation" ? 0.38 : 0.5;
  return {
    x: source.x + dx * pullFromTarget + (-dy / length) * offset,
    y: source.y + dy * pullFromTarget + (dx / length) * offset,
  };
}

function visibleEdgeLabel(edge, source, target, technicalGraph) {
  const label = shortEdgeLabel(edge.label);
  if (!label) return "";
  const important = new Set(["regulated by", "requires", "cites", "mitigated by", "classified as", "retrieved evidence", "supports answer"]);
  if (important.has(label)) return label;
  if (technicalGraph && important.has(shortEdgeLabel(edge.relationshipType))) return shortEdgeLabel(edge.relationshipType);
  return "";
}

function nodeTooltip(node, graph) {
  const illustrative = isIllustrativeGraph(graph);
  const details = [
    node.label,
    `Type: ${node.type}`,
    node.sourceDocument ? `Source: ${node.sourceDocument}` : "",
    node.citation ? `Citation: ${node.citation}` : "",
    node.evidenceId ? `Evidence ID: ${node.evidenceId}` : "",
    node.score !== undefined ? `Relevance score: ${node.score}` : "",
    node.snippet ? `Snippet: ${truncate(node.snippet, 220)}` : "",
    node.hiddenLabels ? `Hidden nodes:\n${node.hiddenLabels}` : "",
    illustrative ? "Origin: illustrative fallback graph" : "Origin: retrieved evidence graph",
  ].filter(Boolean);
  return details.join("\n");
}

function edgeTooltip(edge) {
  return [
    shortEdgeLabel(edge.label) || edge.label || "relationship",
    edge.relationshipType ? `Relationship type: ${edge.relationshipType}` : "",
    edge.evidenceId ? `Evidence ID: ${edge.evidenceId}` : "",
  ].filter(Boolean).join("\n");
}

function shortEdgeLabel(label) {
  const text = String(label || "").toLowerCase();
  if (/requires\s+(gdpr|hipaa|eu ai act|ai act|art\.?\s*\d+|article\s*\d+)/i.test(String(label || ""))) return "requires";
  if (/supports\s+(bias|risk|gdpr|hipaa|eu ai act|ai act)/i.test(String(label || ""))) return "supports";
  if (/classified\s+as\s+(art\.?\s*\d+|article\s*\d+)/i.test(String(label || ""))) return "classified as";
  if (text.includes("ranked_for_question")) return "ranked for question";
  if (text.includes("retrieved_as_evidence")) return "retrieved evidence";
  if (text.includes("supports_answer")) return "supports answer";
  if (text.includes("source_document")) return "source document";
  if (text.includes("processes")) return "processes";
  if (text.includes("regulated by")) return "regulated by";
  if (text.includes("mitigated")) return "mitigated by";
  if (text.includes("generates")) return "generates";
  if (text.includes("introduces")) return "introduces risk";
  if (text.includes("controls")) return "controls";
  if (text.includes("classified")) return "classified as";
  if (text.includes("routes")) return "routes to";
  if (text.includes("matches")) return "matches";
  if (text.includes("retrieves")) return "retrieves";
  if (text.includes("grounds")) return "grounds";
  if (text.includes("applies")) return "applies to";
  if (text.includes("involves")) return "involves";
  if (text.includes("requires")) return "requires";
  if (text.includes("supported by evidence")) return "supports";
  if (text.includes("supported") || text.includes("supports")) return "supports";
  if (text.includes("cites")) return "cites";
  if (text.includes("references")) return "REFERENCES";
  if (text.includes("has_requirement")) return "HAS_REQUIREMENT";
  if (text.includes("instance_of")) return "INSTANCE_OF";
  if (text.includes("subclass_of")) return "SUBCLASS_OF";
  if (text.includes("related_concept")) return "RELATED_CONCEPT";
  if (text.includes("overlaps_with")) return "OVERLAPS_WITH";
  return "";
}

function importantDemoEdgeLabel(label) {
  const normalized = shortEdgeLabel(label);
  return [
    "ranked for question",
    "retrieved evidence",
    "supports answer",
    "source document",
    "processes",
    "regulated by",
    "mitigated by",
    "generates",
    "introduces risk",
    "controls",
    "classified as",
    "routes to",
    "matches",
    "retrieves",
    "grounds",
    "applies to",
    "involves",
    "requires",
    "supports",
    "REFERENCES",
    "HAS_REQUIREMENT",
    "INSTANCE_OF",
    "SUBCLASS_OF",
    "RELATED_CONCEPT",
    "OVERLAPS_WITH",
  ].includes(normalized) ? normalized : "";
}

function normalizeGraphNodes(nodes = []) {
  return nodes.map((node, index) => {
    const label = readableGraphLabel(node, index);
    const sourceDocument = node.source_document || node.sourceDocument || "";
    const citation = node.citation || "";
    return {
      id: node.id || `node-${index + 1}`,
      label,
      type: semanticGraphNodeType(node, label, sourceDocument, citation, index),
      sourceDocument,
      citation,
      evidenceId: node.evidence_id || node.evidenceId || "",
      score: node.score ?? node.relevance_score ?? node.relevanceScore,
      snippet: node.snippet || node.evidence_text || node.evidenceText || "",
    };
  }).filter((node) => node.id && node.label);
}

function normalizeGraphEdges(edges = []) {
  return edges
    .map((edge) => ({
      source: edge.source,
      target: edge.target,
      label: shortEdgeLabel(edge.label || edge.relationship || ""),
      relationshipType: edge.relationship_type || edge.relationshipType || "",
      evidenceId: edge.evidence_id || edge.evidenceId || "",
    }))
    .filter((edge) => edge.source && edge.target);
}

function splitOverloadedRegulationNodes(nodes, edges) {
  const outputNodes = [];
  const outputEdges = [...edges];
  const nodeIds = new Set(nodes.map((node) => node.id));
  nodes.forEach((node) => {
    const split = regulationSplitForLabel(node);
    if (!split) {
      outputNodes.push(node);
      return;
    }
    const regulationId = `reg:${slug(split.regulation)}`;
    if (!nodeIds.has(regulationId)) {
      outputNodes.push({ id: regulationId, label: split.regulation, type: "regulation" });
      nodeIds.add(regulationId);
    }
    outputNodes.push({ ...node, label: split.label, type: node.type === "regulation" ? nodeTypeFor(split.label, 0).toLowerCase() : node.type });
    outputEdges.push({
      source: regulationId,
      target: node.id,
      label: split.edgeLabel,
      relationshipType: "UI_NORMALIZATION",
    });
  });
  return { nodes: outputNodes, edges: outputEdges };
}

function regulationSplitForLabel(node) {
  const label = String(node.label || "");
  const type = String(node.type || "").toLowerCase();
  if (type === "regulation" && /^(GDPR|HIPAA|EU AI Act|AI Act|.+\.pdf)$/i.test(label.trim())) return null;
  const patterns = [
    { re: /\bGDPR\b/i, regulation: "GDPR" },
    { re: /\bEU AI Act\b|\bAI Act\b/i, regulation: "EU AI Act" },
    { re: /\bHIPAA\b/i, regulation: "HIPAA" },
  ];
  for (const pattern of patterns) {
    if (!pattern.re.test(label)) continue;
    const cleaned = label
      .replace(pattern.re, "")
      .replace(/\s{2,}/g, " ")
      .replace(/\s+[-–]\s+$/, "")
      .trim();
    if (!cleaned || cleaned === label) continue;
    return {
      regulation: pattern.regulation,
      label: cleaned,
      edgeLabel: type === "evidence" ? "cites" : "requires",
    };
  }
  return null;
}

function semanticGraphNodeType(node, label, sourceDocument, citation, index) {
  const explicit = String(node.type || node.category || "").toLowerCase();
  if (isRegulationOrSourceLabel(label) || isRegulationOrSourceLabel(sourceDocument)) return "regulation";
  if (isCitationLabel(label) || isCitationLabel(citation)) return "evidence";
  if (explicit === "source") return "regulation";
  if (["evidence", "statement", "regulation", "question", "risk", "control", "concept", "entity", "ontology", "system"].includes(explicit)) {
    if (explicit === "statement" && isCitationLabel(label)) return "evidence";
    return explicit;
  }
  return nodeTypeFor(label || "", index).toLowerCase();
}

function isCitationLabel(value) {
  const text = String(value || "").trim();
  return /^(art\.?|article|articles|annex|recital)\s+[ivx\d]/i.test(text)
    || /^45\s+CFR\b/i.test(text)
    || /^§\s*\d/i.test(text)
    || /\b\d+\s*CFR\s+\d/i.test(text);
}

function isRegulationOrSourceLabel(value) {
  const text = String(value || "").trim();
  return /^(GDPR|HIPAA|EU AI Act|AI Act)$/i.test(text)
    || /^(gdpr|hipaa|eu_ai_act|eu-ai-act|ai_act|ai-act)\.pdf$/i.test(text);
}

function formatScore(score) {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return "Match shown";
  const normalized = parsed > 1 ? parsed : parsed * 100;
  if (normalized >= 70) return "High match";
  if (normalized >= 35) return "Medium match";
  return "Low match";
}

function rawScoreTooltip(score) {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return "Internal retrieval score not available";
  return `Internal retrieval score: ${parsed}`;
}

function technicalEntityChips(selected) {
  const base = [
    "mental health AI",
    "high-risk AI system",
    "health data",
    "risk management",
    "clinical safety",
    "bias",
    "privacy",
    "human oversight",
  ];
  const concepts = relatedConcepts(selected).map((item) => String(item).toLowerCase());
  const entities = [...new Set([...base, ...concepts])];
  const visibleLimit = 14;
  const visible = entities.slice(0, visibleLimit);
  const hidden = entities.slice(visibleLimit);
  return hidden.length ? visible.concat(`+${hidden.length} more: ${hidden.join(", ")}`) : visible;
}

function nodeTypeFor(label, index) {
  const text = String(label).toLowerCase();
  if (text.includes("question")) return "Question";
  if (text.includes("assessment") || text.includes("answer")) return "Assessment";
  if (text.includes("article") || text.includes("articles") || text.includes("security rule")) return "Evidence";
  if (text.includes("gdpr") || text.includes("hipaa") || text.includes("ai act")) return "Regulation";
  if (text.includes("evidence")) return "Evidence";
  if (text.includes("privacy") || text.includes("cyber") || text.includes("human") || text.includes("monitor") || text.includes("safeguard") || text.includes("technical")) return "Control";
  if (text.includes("provider") || text.includes("patient")) return "Actor";
  if (text.includes("risk") || text.includes("clinical") || text.includes("bias")) return "Risk";
  if (index === 0) return "System";
  return "Concept";
}

function readableGraphLabel(node, index = 0) {
  const raw = String(node.label || node.name || node.source_document || node.type || `Node ${index + 1}`).trim();
  if (!raw) return `Node ${index + 1}`;
  const type = String(node.type || node.category || "").toLowerCase();
  if (/\b(GDPR|HIPAA|EU AI Act|AI Act)\b/i.test(raw) && raw.split(/\s+/).length > 1) return truncate(raw, 46);
  if (type === "regulation" && (node.source_document || raw.includes(".pdf"))) return truncate(raw, 38);
  if (type === "evidence") return truncate(raw, 42);
  if (type === "statement") return truncate(raw, 48);
  return shortGraphLabel(raw, index);
}

function edgeLabelFor(index) {
  return ["classified as", "routes to", "matches", "retrieves", "grounds", "supported by", "requires"][index] || "";
}

function explainEvidenceUse(regulation, concept, question) {
  const topic = concept || "this compliance concept";
  if (regulation === "GDPR") return `Connects ${topic} to health-data processing, transparency, lawful basis, or patient rights in this question.`;
  if (regulation === "HIPAA") return `Checks whether protected health information duties or authorisation safeguards may apply to the scenario.`;
  if (regulation === "EU AI Act") return `Links ${topic} to high-risk AI obligations, documentation, oversight, and deployment controls.`;
  return `Provides supporting context for the selected compliance concept.`;
}

function regulationFromText(value = "") {
  const text = String(value).toLowerCase();
  if (text.includes("gdpr")) return "GDPR";
  if (text.includes("hipaa") || text.includes("phi")) return "HIPAA";
  if (text.includes("ai act") || text.includes("eu_ai") || text.includes("2024/1689")) return "EU AI Act";
  return "Other";
}

function normalizeScore(score, fallback) {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return fallback;
  return parsed > 1 ? Math.min(parsed / 100, 0.99) : Math.min(parsed, 0.99);
}

function normalizeGraphPath(path) {
  const seen = new Set();
  return path
    .map((label, index) => shortGraphLabel(label, index))
    .filter((label) => {
      if (!label || seen.has(label)) return false;
      seen.add(label);
      return true;
    });
}

function shortGraphLabel(label, index = 0) {
  const text = String(label || "").trim();
  const lower = text.toLowerCase();
  if (!text) return ["Mental Health AI", "Health Data", "High-Risk AI", "Provider"][index] || "Evidence";
  if (lower.includes("user question") || lower === "question") return "Question";
  if (lower.includes("compliance assessment") || lower.includes("answer")) return "Assessment";
  if (lower.includes("mental") || lower.includes("diagnostic")) return "Mental Health AI";
  if (lower.includes("health data") || lower.includes("special category") || lower.includes("patient health")) return "Health Data";
  if (lower.includes("high-risk") || lower.includes("high risk")) return "High-Risk AI";
  if (lower.includes("provider")) return "Provider";
  if (lower.includes("gdpr") && lower.includes("9")) return "GDPR Art. 9";
  if (lower.includes("gdpr")) return "GDPR";
  if (lower.includes("hipaa") || lower.includes("phi")) return "PHI Safeguards";
  if (lower.includes("ai act")) return "EU AI Act";
  if (lower.includes("risk")) return "Risk Management";
  if (lower.includes("technical") || lower.includes("annex iv")) return "Technical Documentation";
  if (lower.includes("human")) return "Human Oversight";
  if (lower.includes("privacy")) return "Privacy Controls";
  if (lower.includes("cyber")) return "Cybersecurity";
  if (lower.includes("transparency")) return "Transparency";
  if (lower.includes("evidence")) return "Evidence Passages";
  return truncate(text.replace(/\s+/g, " "), 38);
}

function summarizeAnswer(answer, question) {
  const demo = findDemoResponse(question);
  if (demo?.summary) return demo.summary;
  const first = String(answer || "").split(/[.!?]\s/).find(Boolean) || "This question has elevated compliance sensitivity and should be reviewed against the highlighted evidence.";
  return `${truncate(first, 190)}.`;
}

function obligationsForQuestion(question) {
  const text = question.toLowerCase();
  if (text.includes("authorisation") || text.includes("authorization") || text.includes("health data")) {
    return [
      "Confirm the lawful basis and health-data condition before using patient data.",
      "Use explicit consent only when it is the right legal route; document any alternative basis.",
      "Apply minimum necessary access and PHI safeguards where HIPAA applies.",
      "Keep patient notices clear about purpose, retention, and sharing.",
    ];
  }
  if (text.includes("document")) {
    return [
      "Maintain technical documentation for intended purpose, design, and validation.",
      "Link each risk to a mitigation, owner, and monitoring signal.",
      "Record data governance, logging, oversight, and cybersecurity controls.",
      "Keep post-market monitoring evidence ready for review.",
    ];
  }
  return [
    "Run risk management before deployment.",
    "Document technical performance and intended use.",
    "Protect health data and limit access.",
    "Provide clear patient-facing transparency.",
    "Use human oversight for diagnostic decisions.",
  ];
}

function buildVerdict(question, evidence = [], domain = "") {
  const text = `${question} ${domain}`.toLowerCase();
  const laws = inferRegulations(question, evidence);
  if (text.includes("deployment") || text.includes("risk")) {
    return {
      assessment: "Compliance review needed",
      concern: "Pre-deployment healthcare AI risk controls",
      useCase: "Mental health diagnostic AI",
      laws: ["GDPR", "HIPAA", "EU AI Act"],
    };
  }
  if (text.includes("mental") && text.includes("diagnostic")) {
    return {
      assessment: "High compliance sensitivity",
      concern: "Mental health data + diagnostic AI + patient-facing use",
      useCase: "Mental health diagnostic AI",
      laws: ["GDPR", "HIPAA", "EU AI Act"],
    };
  }
  if (text.includes("health data") || text.includes("authorisation") || text.includes("authorization")) {
    return {
      assessment: "Compliance review needed",
      concern: "Patient health data needs a documented legal route",
      laws,
    };
  }
  if (text.includes("high-risk") || text.includes("provider")) {
    return {
      assessment: "Compliance review needed",
      concern: "Provider duties need audit-ready evidence",
      laws,
    };
  }
  return {
    assessment: "Compliance review needed",
    concern: domain || "Healthcare AI risk controls before launch",
    laws,
  };
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function normalizeQuestion(question) {
  return String(question || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function truncate(text, length) {
  const value = String(text || "");
  return value.length > length ? `${value.slice(0, length - 1)}...` : value;
}

function titleIfTruncated(text, length) {
  const value = String(text || "");
  return value.length > length ? `title="${escapeHtml(value)}"` : "";
}

function slug(value) {
  return String(value || "other").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "other";
}

function svgLines(label, width = 144) {
  const words = String(label || "").split(/\s+/);
  const lines = [];
  const maxChars = Math.max(13, Math.floor(width / 9.2));
  words.forEach((word) => {
    if (!lines.length) {
      lines.push(word);
      return;
    }
    const current = lines.at(-1) || "";
    if (`${current} ${word}`.trim().length > maxChars) lines.push(word);
    else lines[lines.length - 1] = `${current} ${word}`.trim();
  });
  const visibleLines = lines.slice(0, 3);
  if (lines.length > visibleLines.length) visibleLines[visibleLines.length - 1] = truncate(visibleLines.at(-1), maxChars - 1);
  return visibleLines.map((line, index) => `<tspan x="0" y="${(index - (visibleLines.length - 1) / 2) * 14}">${escapeHtml(line)}</tspan>`).join("");
}

function inlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

const DEMO_CACHE = {
  [normalizeQuestion("What obligations apply to a mental health diagnostic AI system?")]: {
    domain: "Mental health diagnostic AI",
    regulations: ["GDPR", "HIPAA", "EU AI Act"],
    summary: "Mental health diagnostic AI has high compliance sensitivity because it combines sensitive health data, diagnostic support, and patient-facing risk.",
    obligations: [
      "Treat the system as potentially high-risk AI and document risk controls.",
      "Protect mental health data with a lawful basis and GDPR Article 9 condition.",
      "Apply PHI safeguards where HIPAA is applicable.",
      "Provide clear transparency for patients and deployers.",
      "Keep human oversight and audit logging visible before deployment.",
    ],
    answer: "Mental health diagnostic AI has high compliance sensitivity because it combines sensitive health data, diagnostic support, and patient-facing risk.",
    path: ["Mental Health AI", "Health Data", "High-Risk AI", "Provider", "GDPR Art. 9", "EU AI Act", "PHI Safeguards", "Risk Management", "Technical Documentation", "Human Oversight"],
    evidence: [
      { regulation: "EU AI Act", reference: "Articles 9-15", concept: "High-risk AI system", explanation: "The question concerns diagnostic healthcare AI, so risk management, technical documentation and oversight obligations may be relevant.", passage: "High-risk AI systems require risk management, data governance, technical documentation, logging, transparency, human oversight, accuracy, robustness and cybersecurity." },
      { regulation: "GDPR", reference: "Article 9", concept: "Health data", explanation: "Mental health information is sensitive health data, so special-category processing safeguards are central to the answer.", passage: "Processing personal data requires a lawful basis, and health data requires an Article 9 condition plus suitable safeguards." },
      { regulation: "HIPAA", reference: "45 CFR 164.308 and 164.312", concept: "PHI safeguards", explanation: "Where HIPAA applies, patient health information needs access, audit, integrity and transmission safeguards.", passage: "Administrative and technical safeguards include access controls, audit controls, integrity protections and transmission security." },
    ],
  },
  [normalizeQuestion("Can patient health data be used without explicit authorisation?")]: {
    domain: "Patient health data processing",
    regulations: ["GDPR", "HIPAA"],
    summary: "Patient health data should not be used casually; the system needs a documented legal route and strong safeguards before processing.",
    obligations: [
      "Identify the GDPR lawful basis and health-data condition.",
      "Use explicit consent only when it is the appropriate route.",
      "Apply minimum necessary use and PHI safeguards where HIPAA applies.",
      "Make patient notices clear about purpose and sharing.",
    ],
    answer: "Patient health data should not be used casually; the system needs a documented legal route and strong safeguards before processing.",
    path: ["Health Data", "GDPR Art. 9", "GDPR", "PHI Safeguards", "Privacy Controls", "Patient Notice", "Compliance Answer"],
    evidence: [
      { regulation: "GDPR", reference: "Article 9", concept: "Health data", explanation: "Health data needs an additional processing condition beyond ordinary personal data rules.", passage: "Processing data concerning health is restricted unless a listed Article 9 condition applies." },
      { regulation: "GDPR", reference: "Article 6", concept: "Lawful basis", explanation: "The question asks whether authorisation is always required, so lawful basis is the key distinction.", passage: "Processing is lawful only where at least one recognised lawful basis applies." },
      { regulation: "HIPAA", reference: "45 CFR 164.506", concept: "Permitted PHI use", explanation: "Where HIPAA applies, some uses may be permitted without individual authorisation but still require safeguards.", passage: "Covered entities may use or disclose PHI for treatment, payment, and healthcare operations subject to Privacy Rule limits." },
    ],
  },
  [normalizeQuestion("What must providers of high-risk AI systems document?")]: {
    domain: "High-risk AI provider compliance",
    regulations: ["EU AI Act"],
    summary: "Providers need an audit-ready technical file that links intended use, risks, controls, monitoring, and evidence.",
    obligations: [
      "Document intended purpose, design, validation, and data governance.",
      "Maintain a risk management file with mitigations and residual risks.",
      "Keep logs, oversight measures, and cybersecurity controls reviewable.",
      "Prepare post-market monitoring evidence for ongoing compliance.",
    ],
    answer: "Providers need an audit-ready technical file that links intended use, risks, controls, monitoring, and evidence.",
    path: ["Provider", "High-Risk AI", "Technical Documentation", "Risk Management", "Data Governance", "Logging", "Cybersecurity", "Monitoring"],
    evidence: [
      { regulation: "EU AI Act", reference: "Annex IV", concept: "Technical documentation", explanation: "Defines the technical file providers need before placing high-risk AI on the market.", passage: "Technical documentation should describe the AI system, intended purpose, development methods, validation, risk controls, and monitoring arrangements." },
      { regulation: "EU AI Act", reference: "Article 9", concept: "Risk management", explanation: "Ties documentation to identification, analysis and mitigation of risks.", passage: "A risk management system should be established, implemented, documented and maintained for high-risk AI systems." },
      { regulation: "EU AI Act", reference: "Article 12", concept: "Logging", explanation: "Highlights traceability evidence for a live compliance review.", passage: "High-risk AI systems should be designed with automatic recording of events where appropriate." },
    ],
  },
  [normalizeQuestion("Which GDPR rights apply to a patient using this app?")]: {
    domain: "Patient rights and transparency",
    regulations: ["GDPR"],
    summary: "Patients need clear information and practical routes to exercise GDPR rights when the app processes their health data.",
    obligations: [
      "Explain controller, purpose, data categories, retention, and sharing.",
      "Support access, correction, deletion, restriction, portability, and objection where applicable.",
      "Explain automated decision support in plain language.",
      "Provide human review routes where significant automated effects may arise.",
    ],
    answer: "Patients need clear information and practical routes to exercise GDPR rights when the app processes their health data.",
    path: ["Patient", "Health Data", "GDPR", "Transparency", "Access Rights", "Correction", "Human Review"],
    evidence: [
      { regulation: "GDPR", reference: "Articles 12-15", concept: "Transparency and access", explanation: "Grounds the patient's ability to understand and inspect processing.", passage: "Controllers must provide transparent information and support data subject access rights." },
      { regulation: "GDPR", reference: "Articles 16-18", concept: "Rectification, erasure, restriction", explanation: "Shows correction and control rights relevant to app users.", passage: "Data subjects may request rectification, erasure, or restriction where the legal criteria are met." },
      { regulation: "GDPR", reference: "Article 22", concept: "Automated decision-making", explanation: "Flags safeguards when automated outputs materially affect patients.", passage: "Data subjects have protections concerning decisions based solely on automated processing with significant effects." },
    ],
  },
  [normalizeQuestion("What risks should be mitigated before deployment?")]: {
    domain: "Pre-deployment healthcare AI risk controls",
    regulations: ["GDPR", "HIPAA", "EU AI Act"],
    summary: "Before deployment, the system needs a risk-to-control review across clinical safety, privacy, fairness, and security.",
    obligations: [
      "Clinical safety review and human oversight.",
      "Bias, drift, and performance testing.",
      "Privacy controls for health data.",
      "Cybersecurity, access logging, and auditability.",
      "Post-deployment monitoring.",
    ],
    answer: "Before deployment, the system needs a risk-to-control review across clinical safety, privacy, fairness, and security.",
    path: ["Mental Health AI", "Risk Management", "Clinical Safety", "Bias", "Privacy Controls", "Cybersecurity", "Human Oversight", "Monitoring"],
    evidence: [
      { regulation: "EU AI Act", reference: "Articles 9 and 15", concept: "Risk management", explanation: "Connects deployment readiness to risk management, accuracy, robustness and cybersecurity.", passage: "High-risk AI systems require documented risk management and appropriate accuracy, robustness and cybersecurity measures." },
      { regulation: "GDPR", reference: "Article 35", concept: "DPIA", explanation: "Supports pre-deployment assessment for high-risk health-data processing.", passage: "A DPIA is required where processing is likely to result in high risk to individuals' rights and freedoms." },
      { regulation: "HIPAA", reference: "Security Rule", concept: "Security risk analysis", explanation: "Adds access, audit, integrity and transmission safeguards for PHI environments.", passage: "Security controls should identify and reduce risks to electronic protected health information." },
    ],
  },
  default: {
    domain: "Healthcare AI compliance",
    regulations: ["GDPR", "HIPAA", "EU AI Act"],
    summary: "This healthcare AI question needs a short evidence-backed assessment before any compliance conclusion is presented.",
    obligations: [
      "Identify the healthcare AI context and affected data.",
      "Route the question to GDPR, HIPAA, and EU AI Act where relevant.",
      "Ground the answer in regulatory sources.",
      "Keep technical detail behind the technical details view.",
    ],
    answer: "This healthcare AI question needs a short evidence-backed assessment before any compliance conclusion is presented.",
    path: ["Healthcare AI", "Health Data", "GDPR", "EU AI Act", "PHI Safeguards", "Evidence Passages"],
    evidence: [
      { regulation: "EU AI Act", reference: "High-risk AI obligations", concept: "Provider duties", explanation: "Supplies the AI-system governance frame.", passage: "Providers of high-risk AI systems must manage risk, document controls, support oversight and maintain performance safeguards." },
      { regulation: "GDPR", reference: "Health data processing", concept: "Patient data safeguards", explanation: "Supplies the personal-data and health-data frame.", passage: "Health data processing requires a lawful basis, a special category condition and appropriate safeguards." },
    ],
  },
};

if (typeof window !== "undefined") {
  window.__KEP_DEMO_TEST__ = {
    EVENT_DEMO_CONFIG,
    buildRequestPolicy,
    isDemoPresetQuestion,
    normalizeTimeoutMs,
    responseSourceLabel,
    renderMarkdown,
    renderEvidenceCard,
    normalizeBackendResponse,
    normalizeGraphNodes,
    normalizeGraphEdges,
    splitOverloadedRegulationNodes,
    graphModeLabel,
    visibleEdgeLabel,
    layoutGraph,
    formatScore,
    safeEvidenceReference,
    askQuestionWithDemoPolicy,
    fetchAnswer,
  };
}

render();
