// ============================================================================
// kep-app.js — KEP Compliance Graph-RAG front-end (vanilla, no build step).
// Talks to the existing Flask POST /ask endpoint; falls back to bundled sample
// payloads (kep-demo.js) when the endpoint is unreachable so the page also runs
// as a static file. Theme switching persists to localStorage.
// ============================================================================

(function () {
  "use strict";

  // ---- pipeline stage definitions -----------------------------------------
  const STAGES = [
    { id: "route", label: "Route", glyph: "01", desc: "Classify intent & target regulation" },
    { id: "plan", label: "Plan", glyph: "02", desc: "Decompose into sub-questions" },
    { id: "expand", label: "Expand", glyph: "03", desc: "Concept & phrase expansion" },
    { id: "query", label: "Graph Query", glyph: "04", desc: "Traverse Neo4j ontology" },
    { id: "rank", label: "Rank", glyph: "05", desc: "Score, boost & dedupe rows" },
    { id: "context", label: "Context", glyph: "06", desc: "Group evidence into context" },
    { id: "generate", label: "Generate", glyph: "07", desc: "Compose grounded answer" },
  ];

  const ACCENTS = {
    console: ["#6c7bff", "#3fd0c9", "#c98bff", "#e0b341"],
    terminal: ["#36f08a", "#2bd6d6", "#f0d850", "#ff6b6b"],
    schematic: ["#2f5fd0", "#1f8a5b", "#7a3bb5", "#b0451f"],
  };
  const ACCENT_ON = { console: "#0c0f15", terminal: "#05070a", schematic: "#ffffff" };

  // ---- tiny helpers --------------------------------------------------------
  const $ = (sel, root) => (root || document).querySelector(sel);
  const el = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstChild; };
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const ls = { get: (k, d) => { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} } };
  const hexA = (hex, a) => { const h = hex.replace("#", ""); return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${a})`; };
  const regSlug = (r) => /hipaa/i.test(r) ? "hipaa" : /ai/i.test(r) ? "aiact" : /gdpr/i.test(r) ? "gdpr" : "aiact";

  // ---- app state -----------------------------------------------------------
  const state = {
    direction: ls.get("kep_direction", "console"),
    density: ls.get("kep_density", "comfortable"),
    accent: ls.get("kep_accent", ""),
    history: [],      // {id, regulation, question, status, payload, error, demo}
    activeId: null,
    seq: 0,
    online: null,     // null=unknown, true, false
    inspectorTab: "plan",
    hotEvidence: null,
    graph: null,      // KepGraph instance
    stepTimer: null,
  };

  // ========================================================================
  // THEME
  // ========================================================================
  function applyTheme() {
    const html = document.documentElement;
    html.setAttribute("data-direction", state.direction);
    html.setAttribute("data-density", state.density);
    let accent = state.accent;
    if (!ACCENTS[state.direction].includes(accent)) accent = ACCENTS[state.direction][0];
    html.style.setProperty("--accent", accent);
    html.style.setProperty("--accent-on", ACCENT_ON[state.direction]);
    html.style.setProperty("--accent-soft", hexA(accent, 0.14));
    // reflect in header switcher + settings
    document.querySelectorAll("[data-dir]").forEach((b) => b.classList.toggle("on", b.getAttribute("data-dir") === state.direction));
    document.querySelectorAll("[data-dens]").forEach((b) => b.classList.toggle("on", b.getAttribute("data-dens") === state.density));
    renderSwatches();
  }
  function setDirection(d) { state.direction = d; if (!ACCENTS[d].includes(state.accent)) { state.accent = ACCENTS[d][0]; ls.set("kep_accent", state.accent); } ls.set("kep_direction", d); applyTheme(); if (state.graph) state.graph.setHot(state.hotEvidence); }
  function setDensity(d) { state.density = d; ls.set("kep_density", d); applyTheme(); }
  function setAccent(a) { state.accent = a; ls.set("kep_accent", a); applyTheme(); }
  function renderSwatches() {
    const wrap = $("#swatches"); if (!wrap) return;
    wrap.innerHTML = ACCENTS[state.direction].map((c) =>
      `<span class="swatch${c === ($("html").style.getPropertyValue("--accent").trim() || ACCENTS[state.direction][0]) ? " on" : ""}" style="background:${c};color:${c}" data-accent="${c}"></span>`).join("");
  }

  // ========================================================================
  // ADAPTER — backend /ask payload -> view model
  // ========================================================================
  function adapt(payload) {
    const m = payload.metrics || {};
    const plan = (payload.debug && payload.debug.query_plan) || {};
    const ev = (payload.evidence || []).map((x) => ({
      id: x.id, key: "E" + x.id, citation: x.citation || ("Evidence " + x.id),
      statement: x.statement, text: x.evidence_text, score: typeof x.score === "number" ? x.score : null,
      source_document: x.source_document, source_group: x.source_group,
    }));
    const groups = countGroups(payload);
    const stageTimes = {
      route: m.routing_ms, plan: null, expand: null, query: m.graph_query_ms,
      rank: m.ranking_ms, context: Math.max((m.retrieval_ms || 0) - (m.graph_query_ms || 0), 0) || null,
      generate: m.generation_ms,
    };
    const notes = {
      route: `${plan.category || "classified"} · ${(plan.target_source_documents || []).length || regCount(payload)} reg`,
      plan: `${(plan.sub_questions || []).length} sub-questions`,
      expand: `+${(plan.expansion_terms || []).length} expansion terms`,
      query: `${m.rows_retrieved || 0} candidate rows`,
      rank: `${m.rows_retrieved || 0} → ${m.evidence_items || 0} after dedupe`,
      context: `${groups} source group${groups === 1 ? "" : "s"}`,
      generate: `${m.evidence_items || 0} citations · ${m.model || "model"}`,
    };
    return {
      answer: payload.answer || "", evidence: ev, rows: (payload.debug && payload.debug.top_rows) || [],
      graph: payload.graph || { nodes: [], edges: [] }, plan, metrics: m, stageTimes, notes,
      totalMs: m.total_ms || m.elapsed_ms || Object.values(stageTimes).reduce((a, b) => a + (b || 0), 0),
      groups, source: payload.response_source, error: payload.error,
      determination: payload.determination || null,
    };
  }
  function countGroups(p) {
    const cov = (p.metrics && p.metrics.source_coverage) || {};
    const nz = Object.keys(cov).filter((k) => k !== "unknown" && cov[k] > 0).length;
    if (nz) return nz;
    return new Set((p.evidence || []).map((e) => e.source_document)).size || 1;
  }
  function regCount(p) { return new Set((p.evidence || []).map((e) => e.source_group)).size || 1; }

  // ========================================================================
  // MARKDOWN (answer) — minimal + [E#] citation links
  // ========================================================================
  function renderAnswerMd(md) {
    const blocks = String(md || "").split(/\n{2,}/).map((b) => b.trim()).filter(Boolean);
    return blocks.map((block) => {
      const lines = block.split("\n");
      const isUl = lines.every((l) => /^[-*]\s+/.test(l));
      const isOl = lines.every((l) => /^\d+[.)]\s+/.test(l));
      if (isUl) return "<ul>" + lines.map((l) => "<li>" + inline(l.replace(/^[-*]\s+/, "")) + "</li>").join("") + "</ul>";
      if (isOl) return "<ol>" + lines.map((l) => "<li>" + inline(l.replace(/^\d+[.)]\s+/, "")) + "</li>").join("") + "</ol>";
      return "<p>" + inline(block.replace(/\n/g, " ")) + "</p>";
    }).join("");
  }
  function inline(s) {
    let out = esc(s);
    out = out.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/\[E(\d+)\]/g, '<span class="answer-cite" data-ev="$1">E$1</span>');
    return out;
  }

  // ========================================================================
  // LEFT PANE
  // ========================================================================
  function renderLeft() {
    const left = $("#leftPane");
    const items = state.history.map((h) => {
      const slug = regSlug(h.regulation);
      const color = slug === "hipaa" ? "var(--good)" : slug === "aiact" ? "var(--accent)" : "var(--warn)";
      const on = h.id === state.activeId;
      return `<button class="session${on ? " on" : ""}" data-hid="${h.id}">
        <span class="session-top">
          <span class="session-dot" style="background:${color};box-shadow:0 0 6px ${color}"></span>
          <span class="tag reg-${slug}">${esc(h.regulation)}</span>
          ${h.status === "running" ? '<span class="stage-spin" style="margin-left:auto"></span>' : ""}
          ${h.demo ? '<span class="session-meta" style="margin-left:auto">demo</span>' : ""}
        </span>
        <span class="session-q">${esc(clip(h.question, 84))}</span>
        <span class="session-meta">${h.time}</span>
      </button>`;
    }).join("");

    left.innerHTML = `<div class="lp">
      <button class="lp-new" id="newQuery">+ New query</button>
      <div class="lp-sec-title"><span class="eyebrow">Session history</span><span class="eyebrow">${state.history.length}</span></div>
      ${state.history.length ? items : '<div class="session-meta" style="padding:8px 4px 18px;color:var(--fg-2)">No queries yet — ask one to begin.</div>'}
      <div class="lp-foot">
        <div class="metricline"><span class="k">retriever</span><span class="v">GraphRetriever</span></div>
        <div class="metricline"><span class="k">model</span><span class="v">qwen3:8b</span></div>
        <div class="metricline"><span class="k">neo4j</span><span class="v" id="neoStat" style="color:var(--fg-2)">checking…</span></div>
      </div>
    </div>`;

    $("#newQuery").addEventListener("click", newQuery);
    left.querySelectorAll("[data-hid]").forEach((b) => b.addEventListener("click", () => selectHistory(b.getAttribute("data-hid"))));
    updateNeoStat();
  }
  function updateNeoStat() {
    const elS = $("#neoStat"); if (!elS) return;
    if (state.online === true) { elS.textContent = "online"; elS.style.color = "var(--good)"; }
    else if (state.online === false) { elS.textContent = "offline · demo"; elS.style.color = "var(--warn)"; }
    else { elS.textContent = "checking…"; elS.style.color = "var(--fg-2)"; }
  }
  function clip(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  // ========================================================================
  // CENTER PANE
  // ========================================================================
  function renderComposer() {
    const examples = (window.KEP_DEMO ? window.KEP_DEMO.examples : []);
    return `<div class="composer">
      <div class="composer-box">
        <textarea id="composerInput" class="composer-input" rows="1" placeholder="Ask a compliance question across HIPAA · GDPR · EU AI Act…"></textarea>
        <div class="composer-bar">
          <span class="chip on">Graph-RAG</span>
          <span class="chip">Cite sources</span>
          <span class="spacer"></span>
          <button class="ask" id="askBtn">Ask <kbd>⌘↵</kbd></button>
        </div>
      </div>
      <div class="suggest-row" id="suggestRow">
        ${examples.map((s, i) => `<button class="suggest" data-ex="${i}"><span class="reg">${esc(s.regulation)}</span>${esc(s.question)}</button>`).join("")}
      </div>
    </div>`;
  }

  function renderCenter() {
    const center = $("#centerPane");
    const active = activeItem();
    center.innerHTML = `<div class="cp">${renderComposer()}<div id="resultArea"></div></div>`;

    const ta = $("#composerInput");
    autoGrow(ta);
    ta.addEventListener("input", () => autoGrow(ta));
    ta.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); submit(ta.value); } });
    $("#askBtn").addEventListener("click", () => submit(ta.value));
    center.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => {
      const ex = window.KEP_DEMO.examples[+b.getAttribute("data-ex")];
      submit(ex.question);
    }));

    if (active) {
      if (active.status === "running") renderRunning();
      else if (active.status === "done") renderResult(active);
      else if (active.status === "error") renderError(active);
    } else {
      $("#resultArea").innerHTML = `<div style="text-align:center;padding:48px 0;color:var(--fg-2)"><div class="mono" style="font-size:12px;letter-spacing:.04em">ask a question, or pick one above, to trace it through the pipeline</div></div>`;
    }
  }
  function autoGrow(ta) { if (!ta) return; ta.style.height = "auto"; ta.style.height = Math.min(ta.scrollHeight, 200) + "px"; }

  function stepperHtml(progress, stageTimes) {
    return `<div class="pipe">` + STAGES.map((sd, i) => {
      const done = progress.completed.has(sd.id);
      const active = progress.active === sd.id;
      const note = (done || active) && progress.notes ? progress.notes[sd.id] : "";
      const t = stageTimes && done && stageTimes[sd.id] != null ? stageTimes[sd.id] + "ms" : active ? "···" : (done ? "·" : "");
      const glyph = active ? '<span class="stage-spin"></span>' : done ? "✓" : sd.glyph;
      return `<div class="stage${done ? " done" : ""}${active ? " active" : ""}">
        <div class="stage-glyph">${glyph}</div>
        <div class="stage-body">
          <div class="stage-name">${sd.label}</div>
          <div class="stage-desc">${sd.desc}</div>
          ${note ? `<div class="stage-note">${esc(note)}</div>` : ""}
        </div>
        <div class="stage-time">${t}</div>
      </div>`;
    }).join("") + `</div>`;
  }

  function renderRunning() {
    const area = $("#resultArea");
    area.innerHTML =
      `<div class="sec-head"><span class="eyebrow">Pipeline journey</span><span class="ln"></span><span class="eyebrow">running</span></div>
       <div id="stepperMount"></div>
       <div class="sec-head"><span class="eyebrow">Answer</span><span class="ln"></span></div>
       <div class="skel skel-line" style="width:90%"></div><div class="skel skel-line" style="width:96%"></div><div class="skel skel-line" style="width:70%"></div>`;
  }

  function renderResult(item) {
    const vm = item.vm;
    const slug = regSlug(item.regulation);
    const area = $("#resultArea");

    // no-evidence / error guard
    if (vm.source === "no_evidence" || !vm.evidence.length) {
      area.innerHTML =
        finalStepperBlock(vm) +
        `<div class="sec-head"><span class="eyebrow">Answer</span><span class="ln"></span></div>
         <div class="notice warn"><div class="notice-glyph">∅</div><div><div class="notice-title">No sufficient evidence</div><div class="notice-body">${esc(vm.answer || "No regulatory evidence was found in the knowledge graph for this question.")}</div></div></div>` +
        determinationBlock(vm);
      renderInspector(vm);
      return;
    }

    const coverage = coverageChips(vm);
    area.innerHTML =
      finalStepperBlock(vm) +
      `<div class="sec-head"><span class="eyebrow">Answer</span><span class="ln"></span>${item.demo ? '<span class="demo-badge">● sample data</span>' : `<span class="tag reg-${slug}">${esc(item.regulation)}</span>`}</div>
       <div class="answer fade-in"><div class="answer-md" id="answerMd">${renderAnswerMd(vm.answer)}</div><div class="coverage">${coverage}</div></div>` +
      determinationBlock(vm) +
      `<div class="sec-head"><span class="eyebrow">Knowledge graph</span><span class="ln"></span><span class="eyebrow">${vm.graph.nodes.length} nodes</span></div>
       <div class="graph-wrap" data-om-raster id="graphMount"></div>
       <div class="sec-head"><span class="eyebrow">Evidence · ${vm.evidence.length}</span><span class="ln"></span><span class="eyebrow">grounded</span></div>
       <div class="evidence-list" id="evList">${vm.evidence.map(evidenceCard).join("")}</div>`;

    // graph
    state.graph = window.KepGraph($("#graphMount"));
    state.graph.render(vm.graph);
    state.graph.setHot(state.hotEvidence);

    // citation clicks
    area.querySelectorAll(".answer-cite").forEach((c) => c.addEventListener("click", () => setHot(+c.getAttribute("data-ev"))));
    // evidence interactions
    wireEvidence(area, vm);
    renderInspector(vm);
  }

  function finalStepperBlock(vm) {
    const progress = { completed: new Set(STAGES.map((s) => s.id)), active: null, notes: vm.notes };
    return `<div class="sec-head"><span class="eyebrow">Pipeline journey</span><span class="ln"></span><span class="eyebrow">complete · ${(vm.totalMs / 1000).toFixed(2)}s</span></div>${stepperHtml(progress, vm.stageTimes)}`;
  }

  // ---- determination (backend-derived verdict / obligations / summary) -----
  const VERDICT_META = {
    obligations_apply:         { badge: "obligation",   label: "Obligations apply" },
    permitted_with_conditions: { badge: "permit",       label: "Permitted · conditions" },
    prohibited:                { badge: "prohibit",     label: "Prohibited" },
    insufficient_evidence:     { badge: "insufficient", label: "Insufficient evidence" },
    unavailable:               { badge: "insufficient", label: "Determination unavailable" },
  };
  function determinationBlock(vm) {
    const d = vm.determination;
    if (!d || !d.verdict) return "";
    const meta = VERDICT_META[d.verdict] || { badge: "insufficient", label: d.verdict };
    const obligations = Array.isArray(d.obligations) ? d.obligations : [];
    const summary = d.summary ? `<span class="verdict-text">${inline(d.summary)}</span>` : "";
    const obs = obligations.length
      ? `<div class="answer-md determination-obs"><div class="eyebrow" style="margin-bottom:9px">Obligations · ${obligations.length}</div><ul>${obligations.map((o) => "<li>" + inline(o) + "</li>").join("")}</ul></div>`
      : "";
    return `<div class="sec-head"><span class="eyebrow">Determination</span><span class="ln"></span><span class="eyebrow">backend-derived</span></div>
       <div class="answer fade-in">
         <div class="verdict"><span class="verdict-badge ${meta.badge}">${esc(meta.label)}</span>${summary}</div>
         ${obs}
       </div>`;
  }

  function coverageChips(vm) {
    const cov = vm.metrics.source_coverage || {};
    const labels = { hipaa: "HIPAA", eu_ai_act: "EU AI Act", gdpr: "GDPR" };
    return Object.keys(labels).filter((k) => cov[k] > 0).map((k) =>
      `<span class="tag reg-${k === "eu_ai_act" ? "aiact" : k}">${labels[k]} · ${cov[k]}</span>`).join("")
      || `<span class="tag">${vm.evidence.length} evidence items</span>`;
  }

  function evidenceCard(ev, idx) {
    const open = idx === 0;
    const sc = ev.score != null ? ev.score.toFixed(2) : "—";
    const w = ev.score != null ? Math.round(ev.score * 100) : 0;
    return `<div class="ev${open ? "" : ""}" data-ev="${ev.id}">
      <div class="ev-head" data-toggle="${ev.id}">
        <span class="ev-id">${ev.key}</span>
        <span class="ev-cite">${esc(ev.citation)}</span>
        <span class="ev-statement">${open ? "" : esc(clip(ev.statement, 60))}</span>
        <span class="ev-score">${sc}<span class="score-bar"><span class="score-fill" style="width:${w}%"></span></span></span>
      </div>
      <div class="ev-body" style="${open ? "" : "display:none"}">
        <div style="font-size:12.5px;color:var(--fg-1);padding-top:12px;font-weight:500">${esc(ev.statement)}</div>
        <div class="ev-quote">"${esc(ev.text)}"</div>
        <div class="ev-meta"><span class="tag">${esc(ev.source_document)}</span>${ev.source_group ? `<span class="tag">${esc(ev.source_group)}</span>` : ""}</div>
      </div>
    </div>`;
  }
  function wireEvidence(root, vm) {
    root.querySelectorAll(".ev").forEach((card) => {
      const id = +card.getAttribute("data-ev");
      card.addEventListener("mouseenter", () => setHot(id, false));
      card.addEventListener("mouseleave", () => setHot(null, false));
      $(".ev-head", card).addEventListener("click", () => {
        const body = $(".ev-body", card);
        const stmt = $(".ev-statement", card);
        const isOpen = body.style.display !== "none";
        body.style.display = isOpen ? "none" : "";
        if (stmt) stmt.textContent = isOpen ? clip(vm.evidence.find((e) => e.id === id).statement, 60) : "";
      });
    });
  }
  function setHot(id, scroll) {
    state.hotEvidence = id;
    if (state.graph) state.graph.setHot(id);
    document.querySelectorAll("#evList .ev").forEach((c) => c.classList.toggle("hot", +c.getAttribute("data-ev") === id));
  }

  function renderError(item) {
    const area = $("#resultArea") || (renderRunning(), $("#resultArea"));
    if (area) area.innerHTML = `<div class="notice error"><div class="notice-glyph">!</div><div><div class="notice-title">Request failed</div><div class="notice-body">${esc(item.error || "The system could not retrieve evidence or generate an answer right now.")}</div></div></div>`;
  }

  // ========================================================================
  // RIGHT PANE — inspector
  // ========================================================================
  function renderRight() {
    const active = activeItem();
    const right = $("#rightPane");
    if (!active || (active.status !== "done" && active.status !== "running")) {
      right.innerHTML = `<div class="empty"><div class="empty-inner"><div class="empty-glyph">{ }</div><h3>Retrieval inspector</h3><p>Run a query to inspect the query plan, ranked rows, and retrieval statistics under the hood.</p></div></div>`;
      return;
    }
    if (active.status === "running" || !active.vm) { renderInspectorShell(); return; }
    renderInspector(active.vm);
  }
  function renderInspectorShell() {
    $("#rightPane").innerHTML = `<div class="rp"><div class="rp-tabs">
      <button class="rp-tab on">PLAN</button><button class="rp-tab">RANK</button><button class="rp-tab">STATS</button></div>
      <div class="skel skel-line" style="width:80%"></div><div class="skel skel-line" style="width:95%"></div><div class="skel skel-line" style="width:60%"></div></div>`;
  }
  function renderInspector(vm) {
    const right = $("#rightPane");
    const tabs = [["plan", "PLAN"], ["rank", "RANK"], ["metrics", "STATS"]];
    right.innerHTML = `<div class="rp"><div class="rp-tabs" id="rpTabs">
      ${tabs.map(([id, l]) => `<button class="rp-tab${state.inspectorTab === id ? " on" : ""}" data-tab="${id}">${l}</button>`).join("")}
    </div><div id="rpBody"></div></div>`;
    right.querySelectorAll("[data-tab]").forEach((b) => b.addEventListener("click", () => { state.inspectorTab = b.getAttribute("data-tab"); renderInspector(vm); }));
    const body = $("#rpBody");
    if (state.inspectorTab === "plan") body.innerHTML = planTab(vm.plan);
    else if (state.inspectorTab === "rank") body.innerHTML = rankTab(vm.rows);
    else body.innerHTML = metricsTab(vm);
  }

  function planTab(plan) {
    const pills = (arr) => `<span class="pill-row">${(arr || []).map((a) => `<span class="pill">${esc(a)}</span>`).join("") || '<span class="muted">—</span>'}</span>`;
    return `<div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Query plan</span></div><div class="kv">
        <div class="kv-row"><span class="kv-k">category</span><span class="kv-v"><span class="pill accent">${esc(plan.category || "—")}</span></span></div>
        <div class="kv-row"><span class="kv-k">intent</span><span class="kv-v">${esc(plan.intent || "—")}</span></div>
        <div class="kv-row"><span class="kv-k">actors</span><span class="kv-v">${pills(plan.detected_actors)}</span></div>
        <div class="kv-row"><span class="kv-k">objects</span><span class="kv-v">${pills(plan.detected_objects)}</span></div>
        <div class="kv-row"><span class="kv-k">domains</span><span class="kv-v">${pills(plan.detected_domains)}</span></div>
        <div class="kv-row"><span class="kv-k">target</span><span class="kv-v">${esc((plan.target_source_documents || []).join(", ") || "—")}</span></div>
        ${plan.cross_regulation ? '<div class="kv-row"><span class="kv-k">cross-reg</span><span class="kv-v"><span class="pill accent">true</span></span></div>' : ""}
      </div></div>
      <div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Sub-questions</span></div>
        ${(plan.sub_questions || []).map((s, i) => `<div class="subq"><span class="subq-n">${i + 1}</span><span>${esc(s)}</span></div>`).join("") || '<div class="muted" style="font-size:12px">—</div>'}</div>
      <div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Expansion terms · ${(plan.expansion_terms || []).length}</span></div>
        <div class="pill-row">${(plan.expansion_terms || []).map((t) => `<span class="pill">${esc(t)}</span>`).join("")}</div></div>`;
  }

  function rankTab(rows) {
    if (!rows || !rows.length) return '<div class="rp-block"><div class="muted" style="font-size:12px">No ranked rows (run with debug enabled).</div></div>';
    return `<div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Ranked rows · top ${rows.length}</span></div><div class="rank">` +
      rows.map((r) => {
        const sc = typeof r.score === "number" ? r.score.toFixed(2) : "—";
        const flags = []
          .concat((r.matched_terms || []).map((m) => `<span class="flag matched">${esc(m)}</span>`))
          .concat((r.boosts || []).map((b) => `<span class="flag boost">${esc(b)}</span>`))
          .concat((r.penalties || []).map((p) => `<span class="flag pen">${esc(p)}</span>`)).join("");
        return `<div class="rank-row"><div class="rank-top"><span class="rank-score">${sc}</span><span class="rank-stmt">${esc(r.statement || "—")}</span></div>
          <div class="rank-flags"><span class="rank-route">${esc(r.route || r.query_route || "")}</span>${flags}</div></div>`;
      }).join("") + `</div></div>`;
  }

  function metricsTab(vm) {
    const m = vm.metrics;
    const total = vm.totalMs || 1;
    const lat = STAGES.map((sd) => {
      const ms = vm.stageTimes[sd.id];
      if (ms == null) return "";
      const pct = Math.max(2, (ms / total) * 100);
      return `<div style="padding:8px 0;border-top:1px solid var(--line)"><div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:5px"><span class="mono" style="color:var(--fg-1)">${sd.label}</span><span class="mono muted">${ms}ms</span></div><div class="score-bar" style="width:100%;height:4px"><span class="score-fill" style="width:${pct}%"></span></div></div>`;
    }).join("");
    return `<div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Retrieval stats</span></div><div class="mgrid">
        <div class="mcell"><div class="mcell-v">${m.rows_retrieved || 0}</div><div class="mcell-k">rows retrieved</div></div>
        <div class="mcell"><div class="mcell-v">${m.evidence_items || 0}</div><div class="mcell-k">evidence items</div></div>
        <div class="mcell"><div class="mcell-v">${vm.groups}</div><div class="mcell-k">source groups</div></div>
        <div class="mcell"><div class="mcell-v">${(vm.plan.expansion_terms || []).length}</div><div class="mcell-k">expansion terms</div></div>
      </div></div>
      <div class="rp-block"><div class="rp-block-title"><span class="eyebrow">Latency · ${(total / 1000).toFixed(2)}s total</span></div><div class="rank">${lat}</div></div>`;
  }

  // ========================================================================
  // SUBMIT FLOW
  // ========================================================================
  function activeItem() { return state.history.find((h) => h.id === state.activeId) || null; }

  function newQuery() {
    state.activeId = null; state.hotEvidence = null; state.graph = null;
    stopStepper();
    renderLeft(); renderCenter(); renderRight();
    const ta = $("#composerInput"); if (ta) ta.focus();
  }

  function selectHistory(id) {
    const item = state.history.find((h) => h.id === id); if (!item) return;
    state.activeId = id; state.hotEvidence = null; state.inspectorTab = "plan";
    stopStepper();
    renderLeft(); renderCenter(); renderRight();
  }

  function submit(text) {
    const question = (text || "").trim();
    if (!question) return;
    const id = "q" + (++state.seq);
    const item = { id, question, regulation: guessReg(question), status: "running", time: nowLabel(), payload: null, vm: null, demo: false };
    state.history.unshift(item);
    state.activeId = id; state.hotEvidence = null; state.inspectorTab = "plan";
    renderLeft(); renderCenter(); renderRight();
    startStepper(item);
    askBackend(question).then(({ payload, demo }) => {
      if (state.history.indexOf(item) === -1) return;
      item.demo = demo;
      if (payload && payload.error && !demo) { item.status = "error"; item.error = payload.error; }
      else {
        item.payload = payload; item.vm = adapt(payload);
        item.regulation = regFromVm(item.vm) || item.regulation;
        item.status = "done";
      }
      finishStepper(item);
    }).catch((err) => {
      if (state.history.indexOf(item) === -1) return;
      item.status = "error"; item.error = String(err && err.message || err);
      stopStepper();
      if (item.id === state.activeId) { renderError(item); }
      renderLeft();
    });
  }

  function guessReg(q) { const s = q.toLowerCase(); if (/gdpr|consent|data protection/.test(s)) return "GDPR"; if (/ai|high-risk|provider/.test(s)) return "EU AI Act"; if (/hipaa|phi|covered entity/.test(s)) return "HIPAA"; return "Query"; }
  function regFromVm(vm) {
    const cov = vm.metrics.source_coverage || {};
    let best = "", n = -1;
    [["hipaa", "HIPAA"], ["eu_ai_act", "EU AI Act"], ["gdpr", "GDPR"]].forEach(([k, lbl]) => { if ((cov[k] || 0) > n) { n = cov[k] || 0; best = lbl; } });
    return n > 0 ? best : "";
  }
  function nowLabel() { const d = new Date(); return "Today · " + d.toTimeString().slice(0, 5); }

  // ---- backend call (with offline fallback) -------------------------------
  function askBackend(question) {
    return fetch("ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, show_context: true, debug: true }),
    }).then((r) => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      state.online = true; updateNeoStat();
      return r.json().then((payload) => ({ payload, demo: false }));
    }).catch(() => {
      // offline / static-file fallback → bundled sample payload
      state.online = false; updateNeoStat(); renderLeft();
      const payload = window.KEP_DEMO ? window.KEP_DEMO.match(question) : null;
      if (!payload) throw new Error("No backend and no sample data available.");
      return new Promise((res) => setTimeout(() => res({ payload: clone(payload), demo: true }), 250));
    });
  }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  // ---- stepper animation (optimistic while awaiting response) -------------
  function startStepper(item) {
    stopStepper();
    const completed = new Set();
    let i = 0;
    const notes0 = {};
    const tick = () => {
      const active = STAGES[i] ? STAGES[i].id : "generate";
      if (item.id === state.activeId) {
        const mount = $("#stepperMount");
        if (mount) mount.innerHTML = stepperHtml({ completed: new Set(completed), active, notes: notes0 }, null);
      }
      if (i < STAGES.length - 1) {
        completed.add(STAGES[i].id);
        i++;
        state.stepTimer = setTimeout(tick, 360);
      } else {
        // hold on last stage until response arrives
        state.stepTimer = null;
      }
    };
    tick();
  }
  function finishStepper(item) {
    stopStepper();
    if (item.id === state.activeId) { renderCenter(); renderRight(); }
    renderLeft();
  }
  function stopStepper() { if (state.stepTimer) { clearTimeout(state.stepTimer); state.stepTimer = null; } }

  // ========================================================================
  // HEALTH CHECK
  // ========================================================================
  function checkHealth() {
    fetch("health").then((r) => r.ok ? r.json() : Promise.reject()).then((h) => {
      state.online = (h && h.neo4j && h.neo4j.status === "ok") ? true : true;
      const dot = $("#statusDot"); if (dot) dot.style.background = "var(--good)";
      const lbl = $("#statusLbl"); if (lbl) lbl.textContent = "neo4j · " + ((h.neo4j && h.neo4j.status) || "online");
      updateNeoStat();
    }).catch(() => {
      state.online = false;
      const dot = $("#statusDot"); if (dot) { dot.style.background = "var(--warn)"; dot.style.boxShadow = "0 0 8px var(--warn)"; }
      const lbl = $("#statusLbl"); if (lbl) lbl.textContent = "offline · demo data";
      updateNeoStat();
    });
  }

  // ========================================================================
  // HEADER WIRING + INIT
  // ========================================================================
  function wireHeader() {
    document.querySelectorAll("[data-dir]").forEach((b) => b.addEventListener("click", () => setDirection(b.getAttribute("data-dir"))));
    const gear = $("#gearBtn"), pop = $("#settingsPop");
    if (gear) gear.addEventListener("click", (e) => { e.stopPropagation(); pop.classList.toggle("hidden"); });
    document.addEventListener("click", (e) => { if (pop && !pop.classList.contains("hidden") && !pop.contains(e.target) && e.target !== gear) pop.classList.add("hidden"); });
    document.querySelectorAll("[data-dens]").forEach((b) => b.addEventListener("click", () => setDensity(b.getAttribute("data-dens"))));
    const sw = $("#swatches");
    if (sw) sw.addEventListener("click", (e) => { const s = e.target.closest("[data-accent]"); if (s) setAccent(s.getAttribute("data-accent")); });
  }

  function init() {
    applyTheme();
    wireHeader();
    renderLeft(); renderCenter(); renderRight();
    checkHealth();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
