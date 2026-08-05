// ============================================================================
// kep-graph.js — vanilla SVG renderer for the /ask `graph` payload.
// Generic layered layout keyed by node `type` (no hard-coded ids), so it works
// with whatever _build_graph_payload emits. Exposes window.KepGraph(container).
//   const g = KepGraph(el); g.render(graph); g.setHot(evidenceId|null);
// ============================================================================

(function () {
  const COLS = {
    question: 0,
    entity: 1, concept: 1, ontology: 1, risk: 1,
    statement: 2, control: 2,
    evidence: 3,
    regulation: 4,
  };
  const W = 780;
  const NODE_H = 36;
  const ROW = 58;

  function KepGraph(container) {
    let graph = { nodes: [], edges: [] };
    let hotEvidence = null;
    let hoverNode = null;

    function layout() {
      const cols = {};
      graph.nodes.forEach((n) => {
        const c = COLS[n.type] != null ? COLS[n.type] : 1;
        (cols[c] = cols[c] || []).push(n);
      });
      const usedCols = Object.keys(cols).map(Number).sort((a, b) => a - b);
      const maxRows = Math.max(...usedCols.map((c) => cols[c].length), 1);
      const H = Math.max(maxRows * ROW + 50, 240);
      // distribute used columns evenly across width
      const pad = 70;
      const span = (W - pad * 2);
      const colX = {};
      usedCols.forEach((c, i) => {
        colX[c] = usedCols.length === 1 ? W / 2 : pad + (span * i) / (usedCols.length - 1);
      });
      const pos = {};
      usedCols.forEach((c) => {
        const arr = cols[c];
        const totalH = arr.length * ROW;
        const startY = (H - totalH) / 2 + ROW / 2;
        arr.forEach((n, i) => { pos[n.id] = { x: colX[c], y: startY + i * ROW, node: n }; });
      });
      return { pos, H };
    }

    function nodeWidth(type) {
      if (type === "question") return 150;
      if (type === "regulation") return 132;
      if (type === "statement") return 104;
      if (type === "evidence") return 96;
      return 128;
    }

    function relatedNodes(id) {
      const set = new Set([id]);
      graph.edges.forEach((e) => {
        if (e.source === id) set.add(e.target);
        if (e.target === id) set.add(e.source);
      });
      return set;
    }

    function hotEdge(e) {
      if (hoverNode) return e.source === hoverNode || e.target === hoverNode;
      if (hotEvidence != null) return String(e.evidence_id) === String(hotEvidence);
      return false;
    }
    function hotNodeSet() {
      if (hoverNode) return relatedNodes(hoverNode);
      if (hotEvidence != null) {
        const s = new Set();
        graph.edges.forEach((e) => {
          if (String(e.evidence_id) === String(hotEvidence)) { s.add(e.source); s.add(e.target); }
        });
        return s;
      }
      return null;
    }

    function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
    function clip(s, n) { s = String(s || ""); return s.length > n ? s.slice(0, n - 1) + "…" : s; }

    function draw() {
      if (!graph.nodes.length) {
        container.innerHTML = '<div class="graph-empty">no graph for this result</div>';
        return;
      }
      const { pos, H } = layout();
      const hotSet = hotNodeSet();
      const anyHot = !!hotSet;

      let edgesSvg = "";
      graph.edges.forEach((e) => {
        const a = pos[e.source], b = pos[e.target];
        if (!a || !b) return;
        const hot = hotEdge(e);
        const aw = nodeWidth(a.node.type), bw = nodeWidth(b.node.type);
        let ax = a.x + aw / 2, bx = b.x - bw / 2;
        // handle back-edges (target left of source)
        if (b.x < a.x) { ax = a.x - aw / 2; bx = b.x + bw / 2; }
        const mx = (ax + bx) / 2;
        const op = anyHot && !hot ? 0.18 : 1;
        const d = `M ${ax.toFixed(1)} ${a.y.toFixed(1)} C ${mx.toFixed(1)} ${a.y.toFixed(1)}, ${mx.toFixed(1)} ${b.y.toFixed(1)}, ${bx.toFixed(1)} ${b.y.toFixed(1)}`;
        edgesSvg += `<g opacity="${op}" style="transition:opacity .15s">`;
        edgesSvg += `<path class="gedge${hot ? " hot" : ""}" d="${d}" marker-end="url(#${hot ? "ar-hot" : "ar"})"></path>`;
        if (hot) edgesSvg += `<text class="gedge-label" x="${mx.toFixed(1)}" y="${((a.y + b.y) / 2 - 4).toFixed(1)}" text-anchor="middle">${esc(e.label)}</text>`;
        edgesSvg += `</g>`;
      });

      let nodesSvg = "";
      graph.nodes.forEach((n) => {
        const p = pos[n.id]; if (!p) return;
        const w = nodeWidth(n.type), h = NODE_H;
        const hot = hotSet ? hotSet.has(n.id) : false;
        const dim = anyHot && !hot;
        const col = `var(--node-${n.type}, var(--node-entity))`;
        nodesSvg += `<g class="gnode-box" data-id="${esc(n.id)}" transform="translate(${(p.x - w / 2).toFixed(1)}, ${(p.y - h / 2).toFixed(1)})" opacity="${dim ? 0.3 : 1}" style="transition:opacity .15s">`;
        nodesSvg += `<rect class="gnode-rect" width="${w}" height="${h}" rx="7" fill="var(--bg-3)" stroke="${hot ? "var(--accent)" : col}" stroke-width="${hot ? 2 : 1.3}"></rect>`;
        nodesSvg += `<rect width="4" height="${h}" rx="2" fill="${col}"></rect>`;
        nodesSvg += `<text class="gnode-label" x="${(w / 2 + 2).toFixed(1)}" y="${(h / 2 + 4).toFixed(1)}" text-anchor="middle">${esc(clip(n.label, 18))}</text>`;
        nodesSvg += `</g>`;
      });

      const legendTypes = uniqueTypes();
      const legend = legendTypes.map((t) =>
        `<div class="legend-row"><span class="legend-dot" style="background:var(--node-${t}, var(--node-entity))"></span>${t}</div>`).join("");

      container.innerHTML =
        `<svg viewBox="0 0 ${W} ${H.toFixed(0)}" width="100%" preserveAspectRatio="xMidYMid meet" style="display:block">
          <defs>
            <marker id="ar" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1 L7,4 L0,7" fill="none" stroke="var(--line-2)" stroke-width="1.3"></path></marker>
            <marker id="ar-hot" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,1 L7,4 L0,7" fill="none" stroke="var(--accent)" stroke-width="1.6"></path></marker>
          </defs>
          ${edgesSvg}${nodesSvg}
        </svg>
        <div class="graph-legend">${legend}</div>`;

      // hover wiring
      container.querySelectorAll(".gnode-box").forEach((g) => {
        g.addEventListener("mouseenter", () => { hoverNode = g.getAttribute("data-id"); draw(); });
        g.addEventListener("mouseleave", () => { hoverNode = null; draw(); });
      });
    }

    function uniqueTypes() {
      const order = ["question", "entity", "statement", "evidence", "regulation"];
      const present = new Set(graph.nodes.map((n) => (COLS[n.type] != null ? n.type : "entity")));
      // collapse to canonical buckets for the legend
      const buckets = new Set();
      present.forEach((t) => {
        if (t === "concept" || t === "ontology" || t === "risk") buckets.add("entity");
        else if (t === "control") buckets.add("statement");
        else buckets.add(t);
      });
      return order.filter((t) => buckets.has(t));
    }

    return {
      render(g) { graph = g && g.nodes ? g : { nodes: [], edges: [] }; hoverNode = null; draw(); },
      setHot(ev) { hotEvidence = ev; draw(); },
      clear() { graph = { nodes: [], edges: [] }; container.innerHTML = ""; },
    };
  }

  window.KepGraph = KepGraph;
})();
