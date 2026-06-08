const DEMO_QUESTIONS = [
  "What obligations apply to a mental health diagnostic AI system?",
  "Can patient health data be used without explicit authorisation?",
  "What must providers of high-risk AI systems document?",
  "Which GDPR rights apply to a patient using this app?",
  "What risks should be mitigated before deployment?",
];

const PIPELINE_STEPS = [
  "Question received",
  "Domain detected",
  "Regulations routed",
  "Entities matched",
  "Evidence searched",
  "Answer generated",
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
  DEMO_CACHE_ENABLED: true,
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
  charCount: document.querySelector("#char-count"),
  newSessionButton: document.querySelector("#new-session-button"),
  clearSessionButton: document.querySelector("#clear-session-button"),
  resetDemoButton: document.querySelector("#reset-demo-button"),
  demoGrid: document.querySelector("#demo-grid"),
  leftSidebar: document.querySelector("#left-sidebar"),
  sidebarToggleButton: document.querySelector("#sidebar-toggle-button"),
  demoQuestionList: document.querySelector("#demo-question-list"),
  questionCards: document.querySelector("#question-cards"),
  questionCount: document.querySelector("#question-count"),
  sessionState: document.querySelector("#session-state"),
  stepper: document.querySelector("#stepper"),
  graphStage: document.querySelector("#graph-stage"),
  technicalDetailsDrawer: document.querySelector("#technical-details-drawer"),
  answerContent: document.querySelector("#answer-content"),
  evidenceContent: document.querySelector("#evidence-content"),
  advancedGraphButton: document.querySelector("#advanced-graph-button"),
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
  if (badge) badge.textContent = composerSourceText();
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

els.advancedGraphButton.addEventListener("click", () => {
  state.ui.technicalDetailsOpen = !state.ui.technicalDetailsOpen;
  persistUiPrefs();
  render();
});

els.sidebarToggleButton.addEventListener("click", () => {
  state.ui.sidebarCollapsed = !state.ui.sidebarCollapsed;
  persistUiPrefs();
  applyLayoutPrefs();
  render();
});

els.copyAnswerButton.addEventListener("click", () => {
  const selected = getSelectedQuestion();
  if (selected?.answer) navigator.clipboard?.writeText(selected.answer);
});

document.addEventListener("click", (event) => {
  const demoButton = event.target.closest("[data-demo-question]");
  if (demoButton) {
    if (state.isLoading) return;
    els.input.value = demoButton.dataset.demoQuestion;
    submitQuestion(demoButton.dataset.demoQuestion);
    return;
  }

  const questionCard = event.target.closest("[data-question-id]");
  if (questionCard) {
    state.selectedId = questionCard.dataset.questionId;
    persistSelectedQuestion();
    render();
  }

  const technicalTab = event.target.closest("[data-technical-tab]");
  if (technicalTab) {
    state.ui.technicalDetailsTab = technicalTab.dataset.technicalTab;
    persistUiPrefs();
    renderTechnicalDetails();
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
    graph: buildGraphJourney(question, [], false),
    error: "",
    usedCache: false,
    responseSource: "live_pipeline",
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
    const responseSource = error.isTimeout ? "live_timeout" : "live_failed";
    updateQuestion(item.id, {
      status: "error",
      pipelineStep: PIPELINE_STEPS.length - 1,
      error: responseSource === "live_timeout" ? "Live retrieval timed out." : "Live retrieval failed.",
      responseSource,
      graph: buildGraphJourney(question, [], true),
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
    return normalizeDemoResponse(question, true, "cached_demo");
  }

  if (!policy.livePipelineCalled) {
    throw new Error("Live custom questions are disabled.");
  }

  try {
    const data = await fetchAnswer(question, policy.timeoutMs);
    return normalizeBackendResponse(question, data, false, "live_pipeline");
  } catch (error) {
    if (policy.cacheHit && policy.cacheEnabled && policy.isDemoPreset) {
      return normalizeDemoResponse(question, true, "fallback_cache");
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

function normalizeBackendResponse(question, data, usedCache, responseSource = "live_pipeline") {
  const evidence = (data.evidence || []).slice(0, 6).map((item, index) => {
    const regulation = regulationFromText(`${item.source_group || ""} ${item.source_document || ""} ${item.statement || ""}`);
    return {
      id: String(item.id || index + 1),
      regulation,
      reference: item.citation || item.source_document || "Evidence passage",
      concept: item.statement || "Matched compliance concept",
      explanation: explainEvidenceUse(regulation, item.statement, question),
      passage: item.evidence_text || "",
      score: normalizeScore(item.score, 0.92 - index * 0.04),
    };
  });

  const demo = findDemoResponse(question);
  const fallback = demo ? normalizeDemoResponse(question, true) : null;
  const activeEvidence = evidence.length ? evidence : fallback?.evidence || [];

  return {
    answer: data.answer || fallback?.answer || `The system generated a compliance answer for: ${question}`,
    summary: fallback?.summary || summarizeAnswer(data.answer, question),
    obligations: fallback?.obligations || obligationsForQuestion(question),
    evidence: activeEvidence,
    graph: buildGraphJourney(question, activeEvidence, false),
    debug: data.debug || {},
    domain: detectDomain(question, data.debug?.query_plan),
    regulations: inferRegulations(question, activeEvidence),
    verdict: buildVerdict(question, activeEvidence, detectDomain(question, data.debug?.query_plan)),
    metrics: {
      evidencePassages: data.metrics?.evidence_items || activeEvidence.length,
      searchTimeMs: data.metrics?.elapsed_ms || null,
      routingMs: data.metrics?.routing_ms || null,
      retrievalMs: data.metrics?.retrieval_ms || null,
      graphQueryMs: data.metrics?.graph_query_ms || null,
      rankingMs: data.metrics?.ranking_ms || null,
      generationMs: data.metrics?.generation_ms || null,
      totalMs: data.metrics?.total_ms || data.metrics?.elapsed_ms || null,
    },
    usedCache,
    responseSource,
  };
}

function normalizeDemoResponse(question, usedCache, responseSource = usedCache ? "cached_demo" : "live_pipeline") {
  const demo = findDemoResponse(question) || DEMO_CACHE.default;
  return {
    answer: demo.answer,
    summary: demo.summary,
    obligations: demo.obligations,
    evidence: demo.evidence.map((item, index) => ({
      ...item,
      id: String(index + 1),
      score: item.score || Number((0.96 - index * 0.04).toFixed(2)),
    })),
    graph: buildGraphJourney(question, demo.evidence, false, demo.path),
    debug: demo.debug || {},
    domain: demo.domain,
    regulations: demo.regulations,
    verdict: demo.verdict || buildVerdict(question, demo.evidence, demo.domain),
    metrics: { evidencePassages: demo.evidence.length, searchTimeMs: 360, routingMs: 24, retrievalMs: 140, graphQueryMs: 80, rankingMs: 48, generationMs: 108, totalMs: 390 },
    usedCache,
    responseSource,
  };
}

function buildGraphJourney(question, evidence, failed, preferredPath) {
  if (failed) {
    return {
      nodes: [
        { id: "question", label: "Question", type: "Question" },
        { id: "error", label: "Evidence search unavailable", type: "Risk" },
      ],
      edges: [{ source: "question", target: "error", label: "blocked by" }],
      advancedNodes: [],
      advancedEdges: [],
    };
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

  return { nodes, edges, advancedNodes, advancedEdges };
}

function isRiskGraphPath(question, path) {
  const text = `${question} ${path.join(" ")}`.toLowerCase();
  return text.includes("deployment") || (text.includes("risk management") && text.includes("privacy controls"));
}

function buildRiskGraphJourney() {
  const labels = [
    "Mental Health AI",
    "Risk Management",
    "Clinical Safety",
    "Bias",
    "Privacy Controls",
    "Cybersecurity",
    "Monitoring",
    "Human Oversight",
    "Articles 9 and 15",
    "Article 35 GDPR",
    "Security Rule",
  ];
  const nodes = labels.map((label, index) => ({
    id: `n${index + 1}`,
    label,
    type: nodeTypeFor(label, index),
  }));
  const byLabel = Object.fromEntries(nodes.map((node) => [node.label, node.id]));
  const edges = [
    ["Mental Health AI", "Risk Management", "applies to"],
    ["Risk Management", "Clinical Safety", "involves"],
    ["Clinical Safety", "Bias", ""],
    ["Bias", "Privacy Controls", "requires"],
    ["Cybersecurity", "Risk Management", ""],
    ["Monitoring", "Clinical Safety", ""],
    ["Risk Management", "Human Oversight", "supported by"],
    ["Clinical Safety", "Articles 9 and 15", "supported by"],
    ["Clinical Safety", "Article 35 GDPR", "supported by"],
    ["Privacy Controls", "Security Rule", "supported by"],
  ].map(([source, target, label]) => ({ source: byLabel[source], target: byLabel[target], label }));
  const advancedNodes = [
    { id: "a1", label: "Audit Logs", type: "Control" },
    { id: "a2", label: "Drift Testing", type: "Risk" },
    { id: "a3", label: "DPIA Record", type: "Evidence" },
    { id: "a4", label: "Access Controls", type: "Control" },
  ];
  const advancedEdges = [
    { source: byLabel["Privacy Controls"], target: "a4", label: "requires" },
    { source: byLabel["Monitoring"], target: "a2", label: "requires" },
    { source: byLabel["Article 35 GDPR"], target: "a3", label: "cites" },
    { source: byLabel["Security Rule"], target: "a1", label: "supported by" },
  ];
  return { nodes, edges, advancedNodes, advancedEdges };
}

function inferPath(question, evidence) {
  const regulations = inferRegulations(question, evidence);
  const concepts = evidence.map((item) => item.concept).filter(Boolean).slice(0, 3);
  return [
    "Mental Health AI",
    detectDomain(question),
    regulations.join(" + ") || "Healthcare regulation",
    ...concepts,
    "Sources Used",
    "Compliance answer",
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
  renderDemoButtons();
  renderQuestionCards();
  renderStepper();
  renderGraph();
  renderTechnicalDetails();
  renderAnswer();
  els.askButton.disabled = state.isLoading;
  updateCharacterCount();
  els.questionCount.textContent = "5 suggested questions";
  els.sessionState.textContent = state.isLoading ? "Running pipeline" : state.questions.length ? "Session active" : "Ready";
  els.advancedGraphButton.textContent = state.ui.technicalDetailsOpen ? "Hide technical details" : "Show technical details";
  els.sidebarToggleButton.setAttribute?.("aria-label", state.ui.sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar");
  els.sidebarToggleButton.textContent = state.ui.sidebarCollapsed ? "›" : "‹";
  els.composerSourceBadge = document.querySelector("#composer-source-badge");
  if (els.composerSourceBadge) els.composerSourceBadge.textContent = composerSourceText();
}

function updateCharacterCount() {
  if (!els.charCount) return;
  els.charCount.textContent = `${els.input?.value?.length || 0} / 500`;
}

function renderDemoButtons() {
  const selected = getSelectedQuestion();
  els.demoQuestionList.innerHTML = DEMO_QUESTIONS.map((question, index) => {
    const isSelected = normalizeQuestion(selected?.question || "") === normalizeQuestion(question);
    return `
    <button type="button" data-demo-question="${escapeHtml(question)}" class="${isSelected ? "selected" : ""}" title="${escapeHtml(question)}">
      <span>${index + 1}</span>
      <strong>${escapeHtml(question)}</strong>
    </button>
  `;
  }).join("");
}

function composerSourceText() {
  const question = els.input?.value?.trim() || getSelectedQuestion()?.question || "";
  const policy = buildRequestPolicy(question || "custom");
  return policy.useCache ? "Cached demo response" : "Live pipeline run";
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
      <article class="question-card ${isSelected ? "selected" : ""} ${item.status}" data-question-id="${item.id}">
        <header>
          <span>Q${index + 1}</span>
          <strong>${escapeHtml(truncate(item.question, 72))}</strong>
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
    return `
      <div class="step ${status}">
        <span>✓</span>
        <strong>${escapeHtml(label)}</strong>
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

  const graph = selected.graph || buildGraphJourney(selected.question, selected.evidence || [], selected.status === "error");
  const technicalGraph = state.ui.graphMode === "technical";
  const nodes = normalizeGraphNodes(technicalGraph ? [...graph.nodes, ...graph.advancedNodes] : graph.nodes);
  const edges = normalizeGraphEdges(technicalGraph ? [...graph.edges, ...graph.advancedEdges] : graph.edges);
  const layout = layoutGraph(nodes, selected);
  const byId = Object.fromEntries(layout.map((node) => [node.id, node]));

  const edgeSvg = edges.map((edge) => {
    const source = byId[edge.source];
    const target = byId[edge.target];
    if (!source || !target) return "";
    const midX = (source.x + target.x) / 2;
    const midY = (source.y + target.y) / 2 - 10;
    const edgeLabel = technicalGraph ? shortEdgeLabel(edge.label) : importantDemoEdgeLabel(edge.label);
    const label = edgeLabel ? `<g class="edge-label"><rect x="${midX - 42}" y="${midY - 14}" width="84" height="20" rx="6"></rect><text x="${midX}" y="${midY}">${escapeHtml(edgeLabel)}</text></g>` : "";
    return `
      <path marker-end="url(#arrowhead)" d="M ${source.x + 70} ${source.y} C ${midX} ${source.y}, ${midX} ${target.y}, ${target.x - 70} ${target.y}"></path>
      ${label}
    `;
  }).join("");

  const nodeSvg = layout.map((node, index) => {
    const isPrimary = isPrimaryGraphNode(selected, node, index, technicalGraph);
    const step = isPrimary ? `<text class="node-step" x="-44" y="-16">${primaryStepNumber(layout, selected, index)}</text>` : "";
    return `
    <g class="graph-node graph-${slug(node.type)} ${isPrimary ? "primary" : "secondary"}" transform="translate(${node.x}, ${node.y})">
      <rect x="-70" y="-34" width="140" height="68" rx="10"></rect>
      <text text-anchor="middle">${svgLines(node.label)}</text>
      ${step}
      <text class="node-type" y="55" text-anchor="middle">${escapeHtml(node.type)}</text>
    </g>
  `;
  }).join("");

  els.graphStage.innerHTML = `
    <div class="graph-meta">
      <span>${nodes.length} highlighted nodes</span>
      <span>${edges.length} highlighted links</span>
      <span>${technicalGraph ? "Technical graph" : "Demo graph"}</span>
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
      <span><i class="legend-orange"></i>Orange = risk</span>
      <span><i class="legend-green"></i>Green = control / evidence</span>
    </div>
    <svg viewBox="0 0 1060 500" role="img" aria-label="Highlighted compliance knowledge graph">
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
}

function renderTechnicalDetails() {
  if (!state.ui.technicalDetailsOpen) {
    els.technicalDetailsDrawer.hidden = false;
    els.technicalDetailsDrawer.innerHTML = `
      <button type="button" class="technical-drawer-handle" data-graph-action="toggle-details">
        <span>⌄</span>
        Show technical details
        <span>⌄</span>
      </button>
    `;
    return;
  }
  const selected = getSelectedQuestion();
  els.technicalDetailsDrawer.hidden = false;
  if (!selected) {
    els.technicalDetailsDrawer.innerHTML = `
      <button type="button" class="technical-drawer-handle" data-graph-action="toggle-details">
        <span>⌄</span>
        Show technical details
        <span>⌄</span>
      </button>
    `;
    return;
  }
  const activeTab = state.ui.technicalDetailsTab || "matched-entities";
  els.technicalDetailsDrawer.innerHTML = `
    <header>
      <div>
        <strong>Hide technical details</strong>
        <span>${escapeHtml(responseSourceLabel(selected))}</span>
      </div>
      <div class="technical-actions">
        <button type="button" data-graph-action="technical-mode">${state.ui.graphMode === "technical" ? "Use demo graph" : "Expand graph"}</button>
        <button type="button" data-graph-action="fit">Fit graph</button>
      </div>
    </header>
    <div class="technical-tabs">
      ${TECHNICAL_TABS.map((tab) => `<button type="button" data-technical-tab="${tab.id}" class="${activeTab === tab.id ? "active" : ""}">${tab.label}</button>`).join("")}
    </div>
    <div class="technical-tab-content">${renderTechnicalTabContent(selected, activeTab)}</div>
  `;
}

function renderTechnicalTabContent(selected, tabId) {
  if (tabId === "matched-entities") return renderChipList(getMatchedEntities(selected), "No matched entities available for this response.");
  if (tabId === "retrieval-scores") {
    const scores = getRetrievalScores(selected);
    if (!scores.length) return renderTechnicalEmpty("No retrieval scores available for this response.");
    return `<div class="technical-table">${scores.map((item) => `
      <div><strong>${escapeHtml(item.reference)}</strong><span>${escapeHtml(item.regulation)}</span><em>${formatScore(item.score)}</em></div>
    `).join("")}</div>`;
  }
  if (tabId === "evidence-ids") return renderChipList(getEvidenceIds(selected), "No evidence IDs available for this response.");
  if (tabId === "graph-traversal") {
    const details = getGraphTraversalDetails(selected);
    if (!details.length) return renderTechnicalEmpty("No graph traversal details available for this response.");
    return `<div class="technical-table traversal">${details.map((item) => `
      <div><strong>${escapeHtml(item.from)}</strong><span>${escapeHtml(item.relationship)}</span><strong>${escapeHtml(item.to)}</strong></div>
    `).join("")}</div>`;
  }
  if (tabId === "pipeline-timings") {
    const timings = getPipelineTimings(selected);
    if (!timings.length) return renderTechnicalEmpty("No pipeline timings available for this response.");
    return `<div class="technical-table timings">${timings.map((item) => `
      <div><strong>${escapeHtml(item.label)}</strong><em>${escapeHtml(item.value)}</em></div>
    `).join("")}</div>`;
  }
  return renderTechnicalEmpty("No technical detail available for this tab.");
}

function renderChipList(items, emptyMessage) {
  if (!items.length) return renderTechnicalEmpty(emptyMessage);
  return `<div class="entity-chips">${items.map((item) => `<span>${escapeHtml(item)}</span>`).join("")}</div>`;
}

function renderTechnicalEmpty(message) {
  return `<div class="technical-empty">${escapeHtml(message)}</div>`;
}

function getMatchedEntities(item) {
  return technicalEntityChips(item);
}

function getRetrievalScores(item) {
  return (item.evidence || []).map((entry) => ({
    reference: entry.reference || entry.id || "Evidence passage",
    regulation: entry.regulation || "Regulation",
    score: entry.score,
  }));
}

function getEvidenceIds(item) {
  return (item.evidence || []).map((entry) => `${entry.regulation || "Source"} · ${entry.reference || entry.id || "Evidence"}`);
}

function getGraphTraversalDetails(item) {
  const graph = item.graph || buildGraphJourney(item.question, item.evidence || [], item.status === "error");
  const byId = Object.fromEntries([...(graph.nodes || []), ...(graph.advancedNodes || [])].map((node) => [node.id, node.label || node.id]));
  return [...(graph.edges || []), ...(graph.advancedEdges || [])].map((edge) => ({
    from: byId[edge.source] || edge.source || "Source",
    relationship: shortEdgeLabel(edge.label) || "related to",
    to: byId[edge.target] || edge.target || "Target",
  }));
}

function getPipelineTimings(item) {
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

  els.answerContent.innerHTML = `
    <article class="answer-card">
      <div class="answer-topline">
        ${renderRegulationBadges(selected.regulations)}
        <span class="cache-badge source-${slug(selected.responseSource || "live_pipeline")}">${escapeHtml(responseSourceLabel(selected))}</span>
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
      <h3>Sources used</h3>
      <span>${selected.evidence.length} regulatory sources selected</span>
    </div>
    <div class="evidence-list">
      ${selected.evidence.map(renderEvidenceCard).join("") || renderEmptyState("No evidence passages were returned.")}
    </div>
  `;
}

function renderEvidenceCard(item) {
  return `
    <details class="evidence-card">
      <summary>
        <span class="reg-badge ${slug(item.regulation)}">${escapeHtml(item.regulation)}</span>
        <span class="source-card-copy">
          <strong>${escapeHtml(item.reference)}</strong>
          <small>${escapeHtml(sourceShortExplanation(item))}</small>
        </span>
      </summary>
      <dl>
        <div><dt>Regulation</dt><dd>${escapeHtml(item.regulation)}</dd></div>
        <div><dt>Section or article</dt><dd>${escapeHtml(item.reference)}</dd></div>
        <div><dt>Matched concept</dt><dd>${escapeHtml(item.concept)}</dd></div>
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
      <div>${renderAnswerText(full)}</div>
    </details>
  `;
}

function renderAnswerText(text) {
  const lines = String(text || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (!lines.length) return "<p>No answer generated.</p>";
  return lines.map((line) => {
    const bullet = line.match(/^[-*]\s+(.+)/);
    return bullet ? `<p class="answer-bullet">${inlineMarkdown(bullet[1])}</p>` : `<p>${inlineMarkdown(line)}</p>`;
  }).join("");
}

function renderEmptyState(message) {
  return `<div class="empty-card"><strong>Ready for demo</strong><p>${escapeHtml(message)}</p></div>`;
}

function stepDescription(label, item) {
  const descriptions = {
    "Question received": truncate(item.question, 56),
    "Domain detected": item.domain || "Healthcare AI",
    "Regulations routed": (item.regulations || []).join(", ") || "GDPR, HIPAA, EU AI Act",
    "Entities matched": relatedConcepts(item).slice(0, 2).join(", "),
    "Evidence searched": `${item.evidence?.length || 0} sources selected`,
    "Answer generated": item.status === "complete" ? "Visitor-ready assessment" : item.status === "error" ? "Error state shown" : "Preparing answer",
  };
  return descriptions[label] || "";
}

function relatedConcepts(item) {
  const concepts = (item.evidence || []).map((entry) => entry.concept).filter(Boolean).slice(0, 3);
  return concepts.length ? concepts : ["health data", "high-risk AI", "patient safeguards"];
}

function formatQuestionMeta(item) {
  if (item.status === "loading") return `Running: ${PIPELINE_STEPS[item.pipelineStep]}`;
  if (item.status === "error") return item.error;
  const time = item.metrics?.searchTimeMs ? `${item.metrics.searchTimeMs} ms` : "live backend";
  return `${item.evidence.length} regulatory sources selected · ${time}`;
}

function responseSourceLabel(item) {
  const source = item.responseSource || (item.usedCache ? "cached_demo" : "live_pipeline");
  if (source === "cached_demo") return "Cached demo response";
  if (source === "fallback_cache") return "Fallback cache shown";
  if (source === "live_failed") return "Live retrieval failed";
  if (source === "live_timeout") return "Live retrieval timed out";
  return "Live pipeline run";
}

function buildRequestPolicy(question, config = EVENT_DEMO_CONFIG) {
  const isDemoPreset = isDemoPresetQuestion(question);
  const cacheHit = Boolean(findDemoResponse(question));
  const cacheEnabled = Boolean(config.EVENT_DEMO_MODE && config.DEMO_CACHE_ENABLED);
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
    cacheHit: policy.cacheHit,
    timeoutMs: policy.timeoutMs,
    useCache: policy.useCache,
    livePipelineCalled: policy.livePipelineCalled,
  });
}

function statusLabelForQuestion(item) {
  if (item.status === "loading") return "Running";
  if (item.status === "error") return "Review needed";
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

function isPrimaryGraphNode(selected, node, index, advanced) {
  if (advanced && node.id?.startsWith("a")) return false;
  const text = selected.question.toLowerCase();
  if (text.includes("risk") || text.includes("deployment")) {
    return ["Mental Health AI", "Risk Management", "Clinical Safety", "Bias", "Privacy Controls"].includes(node.label);
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

function handleGraphAction(action) {
  if (action === "toggle-details") {
    state.ui.technicalDetailsOpen = !state.ui.technicalDetailsOpen;
    persistUiPrefs();
    render();
    return;
  }
  if (action === "fit") fitGraph();
  if (action === "zoom-in") state.ui.graphZoom = Math.min(1.8, Number((state.ui.graphZoom + 0.12).toFixed(2)));
  if (action === "zoom-out") state.ui.graphZoom = Math.max(0.72, Number((state.ui.graphZoom - 0.12).toFixed(2)));
  if (action === "reset-layout") {
    state.ui.graphZoom = 1;
  }
  if (action === "fullscreen") state.ui.graphFullscreen = !state.ui.graphFullscreen;
  if (action === "technical-mode") {
    state.ui.graphMode = state.ui.graphMode === "technical" ? "demo" : "technical";
    state.ui.graphZoom = 1;
  }
  persistUiPrefs();
  render();
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
        if (side === "left" && !state.ui.sidebarCollapsed) {
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
  els.leftSidebar?.classList?.toggle("is-collapsed", state.ui.sidebarCollapsed);
  els.graphStage?.classList?.toggle("is-fullscreen", state.ui.graphFullscreen);
}

function effectiveLeftWidth() {
  return state.ui.sidebarCollapsed ? 76 : clampNumber(Number(state.ui.leftWidth) || DEFAULT_UI_PREFS.leftWidth, 260, 520);
}

function readUiPrefs() {
  try {
    const raw = { ...DEFAULT_UI_PREFS, ...JSON.parse(localStorage.getItem(UI_STORAGE_KEY) || "{}") };
    const tabIds = TECHNICAL_TABS.map((tab) => tab.id);
    return {
      ...raw,
      sidebarCollapsed: Boolean(raw.sidebarCollapsed),
      leftWidth: clampNumber(Number(raw.leftWidth) || DEFAULT_UI_PREFS.leftWidth, 260, 520),
      rightWidth: clampNumber(Number(raw.rightWidth) || DEFAULT_UI_PREFS.rightWidth, 320, 620),
      technicalDetailsOpen: Boolean(raw.technicalDetailsOpen),
      technicalDetailsTab: tabIds.includes(raw.technicalDetailsTab) ? raw.technicalDetailsTab : "matched-entities",
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

function layoutGraph(nodes, selected) {
  if (state.ui.graphMode === "technical") {
    const columns = Math.min(5, Math.max(3, Math.ceil(Math.sqrt(nodes.length))));
    const xGap = 880 / Math.max(columns - 1, 1);
    const rows = Math.ceil(nodes.length / columns);
    const yGap = 360 / Math.max(rows - 1, 1);
    return nodes.map((node, index) => ({
      ...node,
      x: 90 + (index % columns) * xGap,
      y: 80 + Math.floor(index / columns) * yGap,
    }));
  }
  const risk = `${selected?.question || ""} ${selected?.domain || ""}`.toLowerCase().includes("risk")
    || `${selected?.question || ""}`.toLowerCase().includes("deployment");
  if (risk) {
    const positions = {
      "Mental Health AI": [105, 250],
      "Risk Management": [300, 250],
      "Clinical Safety": [500, 250],
      "Bias": [700, 250],
      "Privacy Controls": [900, 250],
      "Cybersecurity": [300, 110],
      "Monitoring": [700, 110],
      "Human Oversight": [220, 390],
      "Articles 9 and 15": [420, 390],
      "Article 35 GDPR": [630, 390],
      "Security Rule": [830, 390],
    };
    return nodes.map((node, index) => {
      const fallback = [110 + (index % 5) * 205, 115 + Math.floor(index / 5) * 145];
      const [x, y] = positions[node.label] || fallback;
      return { ...node, x, y };
    });
  }
  const columns = Math.min(Math.max(nodes.length, 2), 5);
  return nodes.map((node, index) => {
    const col = index % columns;
    const row = Math.floor(index / columns);
    return {
      ...node,
      x: 105 + col * 205,
      y: 125 + row * 150 + (col % 2 ? 18 : 0),
    };
  });
}

function shortEdgeLabel(label) {
  const text = String(label || "").toLowerCase();
  if (text.includes("applies")) return "applies to";
  if (text.includes("involves")) return "involves";
  if (text.includes("requires")) return "requires";
  if (text.includes("supported")) return "supported by";
  if (text.includes("cites")) return "cites";
  return "";
}

function importantDemoEdgeLabel(label) {
  const normalized = shortEdgeLabel(label);
  return ["applies to", "involves", "requires", "supported by"].includes(normalized) ? normalized : "";
}

function normalizeGraphNodes(nodes = []) {
  return nodes.map((node, index) => ({
    id: node.id || `node-${index + 1}`,
    label: shortGraphLabel(node.label || node.name || node.type || `Node ${index + 1}`, index),
    type: node.type || node.category || nodeTypeFor(node.label || "", index),
  })).filter((node) => node.id && node.label);
}

function normalizeGraphEdges(edges = []) {
  return edges
    .map((edge) => ({
      source: edge.source,
      target: edge.target,
      label: shortEdgeLabel(edge.label || edge.relationship || ""),
    }))
    .filter((edge) => edge.source && edge.target);
}

function formatScore(score) {
  const parsed = Number(score);
  if (!Number.isFinite(parsed)) return "n/a";
  return `${Math.round(parsed * 100)}%`;
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
  return [...new Set([...base, ...concepts])].slice(0, 8).concat("+3 more");
}

function nodeTypeFor(label, index) {
  const text = String(label).toLowerCase();
  if (text.includes("gdpr") || text.includes("hipaa") || text.includes("ai act")) return "Regulation";
  if (text.includes("evidence")) return "Evidence";
  if (text.includes("privacy") || text.includes("cyber") || text.includes("human") || text.includes("monitor") || text.includes("safeguard") || text.includes("technical")) return "Control";
  if (text.includes("provider") || text.includes("patient")) return "Actor";
  if (text.includes("risk") || text.includes("clinical") || text.includes("bias")) return "Risk";
  if (index === 0) return "System";
  return "Concept";
}

function edgeLabelFor(index) {
  return ["applies to", "involves data", "", "requires", "supported by", "", "requires"][index] || "";
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
  return truncate(text.replace(/\s+/g, " "), 24);
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

function slug(value) {
  return String(value || "other").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "other";
}

function svgLines(label) {
  const words = String(label || "").split(/\s+/);
  const lines = [];
  words.forEach((word) => {
    if (!lines.length) {
      lines.push(word);
      return;
    }
    const current = lines.at(-1) || "";
    if (`${current} ${word}`.trim().length > 15) lines.push(word);
    else lines[lines.length - 1] = `${current} ${word}`.trim();
  });
  const visibleLines = lines.slice(0, 3);
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
    fetchAnswer,
  };
}

render();
