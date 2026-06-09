from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT / "web" / "static" / "app.js"


def _run_policy_probe(expression: str) -> dict[str, object]:
    script = dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const elements = new Map();
        function element(selector) {{
          if (!elements.has(selector)) {{
            elements.set(selector, {{
              innerHTML: "",
              textContent: "",
              disabled: false,
              dataset: {{}},
              addEventListener() {{}},
            }});
          }}
          return elements.get(selector);
        }}
        const context = {{
          console: {{ info() {{}}, log() {{}}, error() {{}} }},
          window: {{}},
          document: {{
            querySelector: element,
            addEventListener() {{}},
          }},
          localStorage: {{
            getItem() {{ return "[]"; }},
            setItem() {{}},
          }},
          crypto: {{ randomUUID() {{ return "test-id"; }} }},
          setTimeout,
          clearTimeout,
          fetch() {{ throw new Error("fetch should not run in policy probe"); }},
        }};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync({json.dumps(str(APP_JS))}, "utf8"), context);
        const api = context.window.__KEP_DEMO_TEST__;
        const result = {expression};
        process.stdout.write(JSON.stringify(result));
        """
    )
    output = subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)
    return json.loads(output)


def test_demo_preset_calls_live_backend_by_default() -> None:
    result = _run_policy_probe(
        'api.buildRequestPolicy("What risks should be mitigated before deployment?")'
    )

    assert result["isDemoPreset"] is True
    assert result["cacheHit"] is True
    assert result["cacheEnabled"] is False
    assert result["fallbackEnabled"] is True
    assert result["useCache"] is False
    assert result["livePipelineCalled"] is True
    assert result["timeoutMs"] == 8000


def test_demo_preset_can_be_forced_to_use_cache_only_when_explicitly_enabled() -> None:
    result = _run_policy_probe(
        """api.buildRequestPolicy(
          "What risks should be mitigated before deployment?",
          {
            EVENT_DEMO_MODE: true,
            DEMO_CACHE_ENABLED: true,
            DEMO_FALLBACK_ENABLED: true,
            LIVE_CUSTOM_QUESTIONS_ENABLED: true,
            DEMO_PRESET_TIMEOUT_MS: 8000,
            CUSTOM_QUESTION_TIMEOUT_MS: 60000
          }
        )"""
    )

    assert result["useCache"] is True
    assert result["livePipelineCalled"] is False


def test_demo_preset_submission_calls_ask_endpoint_in_default_live_mode() -> None:
    script = dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const elements = new Map();
        function element(selector) {{
          if (!elements.has(selector)) {{
            elements.set(selector, {{
              innerHTML: "",
              textContent: "",
              disabled: false,
              dataset: {{}},
              addEventListener() {{}},
            }});
          }}
          return elements.get(selector);
        }}
        const fetchCalls = [];
        const context = {{
          console: {{ info() {{}}, log() {{}}, error() {{}} }},
          window: {{}},
          document: {{ querySelector: element, addEventListener() {{}} }},
          localStorage: {{ getItem() {{ return "[]"; }}, setItem() {{}} }},
          crypto: {{ randomUUID() {{ return "test-id"; }} }},
          AbortController,
          setTimeout,
          clearTimeout,
          fetch(url) {{
            fetchCalls.push(url);
            return Promise.resolve({{
              ok: true,
              json: () => Promise.resolve({{
                answer: "Live answer",
                evidence: [],
                debug: {{}},
                metrics: {{}},
                error: null
              }})
            }});
          }},
        }};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync({json.dumps(str(APP_JS))}, "utf8"), context);
        context.window.__KEP_DEMO_TEST__
          .askQuestionWithDemoPolicy("What risks should be mitigated before deployment?")
          .then((result) => process.stdout.write(JSON.stringify({{ fetchCalls, source: result.responseSource }})))
          .catch((error) => {{
            console.error(error);
            process.exit(1);
          }});
        """
    )

    output = subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)
    result = json.loads(output)

    assert result["fetchCalls"] == ["/ask"]
    assert result["source"] == "live"


def test_demo_preset_failure_uses_clearly_labelled_cached_fallback() -> None:
    script = dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const elements = new Map();
        function element(selector) {{
          if (!elements.has(selector)) {{
            elements.set(selector, {{
              innerHTML: "",
              textContent: "",
              disabled: false,
              dataset: {{}},
              addEventListener() {{}},
            }});
          }}
          return elements.get(selector);
        }}
        const context = {{
          console: {{ info() {{}}, log() {{}}, error() {{}} }},
          window: {{}},
          document: {{ querySelector: element, addEventListener() {{}} }},
          localStorage: {{ getItem() {{ return "[]"; }}, setItem() {{}} }},
          crypto: {{ randomUUID() {{ return "test-id"; }} }},
          AbortController,
          setTimeout,
          clearTimeout,
          fetch() {{ return Promise.reject(new Error("backend unavailable")); }},
        }};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync({json.dumps(str(APP_JS))}, "utf8"), context);
        context.window.__KEP_DEMO_TEST__
          .askQuestionWithDemoPolicy("What risks should be mitigated before deployment?")
          .then((result) => process.stdout.write(JSON.stringify({{
            source: result.responseSource,
            label: context.window.__KEP_DEMO_TEST__.responseSourceLabel(result),
            timings: result.metrics
          }})))
          .catch((error) => {{
            console.error(error);
            process.exit(1);
          }});
        """
    )

    output = subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)
    result = json.loads(output)

    assert result["source"] == "cached_fallback"
    assert result["label"] == "Pre-loaded demo response"
    assert result["timings"] == {}


def test_custom_question_never_uses_cache_and_calls_live_backend() -> None:
    result = _run_policy_probe(
        'api.buildRequestPolicy("What is a DPIA and when is it required under GDPR?")'
    )

    assert result["isDemoPreset"] is False
    assert result["cacheHit"] is False
    assert result["useCache"] is False
    assert result["livePipelineCalled"] is True
    assert result["timeoutMs"] == 60000


def test_cache_miss_does_not_trigger_timeout_path() -> None:
    result = _run_policy_probe(
        'api.buildRequestPolicy("A totally new custom compliance question")'
    )

    assert result["cacheHit"] is False
    assert result["useCache"] is False
    assert result["livePipelineCalled"] is True


def test_timeout_values_are_milliseconds_and_seconds_are_normalized() -> None:
    result = _run_policy_probe(
        "({ demo: api.normalizeTimeoutMs(8, 8000), custom: api.normalizeTimeoutMs(60, 60000), fallback: api.normalizeTimeoutMs(undefined, 60000) })"
    )

    assert result == {"demo": 8000, "custom": 60000, "fallback": 60000}


def test_response_source_labels_cover_live_and_cache_states() -> None:
    result = _run_policy_probe(
        """({
          cached: api.responseSourceLabel({ responseSource: "cached_fallback" }),
          live: api.responseSourceLabel({ responseSource: "live" }),
          failed: api.responseSourceLabel({ responseSource: "error" }),
          timeout: api.responseSourceLabel({ responseSource: "timeout" })
        })"""
    )

    assert result == {
        "cached": "Pre-loaded demo response",
        "live": "Live retrieval completed",
        "failed": "Live retrieval failed",
        "timeout": "Fallback response after timeout",
    }


def test_answer_markdown_is_rendered_not_displayed_raw() -> None:
    result = _run_policy_probe(
        'api.renderMarkdown("below: ### **GDPR Obligations**\\n1. Run a DPIA\\n- Keep **records**")'
    )

    assert "<h5><strong>GDPR Obligations</strong></h5>" in result
    assert "<ol><li>Run a DPIA</li></ol>" in result
    assert "<ul><li>Keep <strong>records</strong></li></ul>" in result
    assert "below:" not in result
    assert "###" not in result


def test_source_card_keeps_document_citation_title_and_snippet_separate() -> None:
    result = _run_policy_probe(
        """api.renderEvidenceCard({
          regulation: "GDPR",
          sourceDocument: "gdpr.pdf",
          reference: "Article 35",
          statementTitle: "Data protection impact assessment",
          concept: "DPIA",
          passage: "A DPIA is required where processing is likely to result in high risk.",
          explanation: "The question asks when a DPIA is required."
        })"""
    )

    assert "gdpr.pdf" in result
    assert "Article 35" in result
    assert "Data protection impact assessment" in result
    assert "A DPIA is required" in result
    assert "The question asks when a DPIA is required." in result


def test_source_document_is_not_rendered_as_section_or_article() -> None:
    result = _run_policy_probe(
        """({
          safe: api.safeEvidenceReference("gdpr.pdf", "gdpr.pdf"),
          card: api.renderEvidenceCard({
            regulation: "GDPR",
            sourceDocument: "gdpr.pdf",
            reference: "gdpr.pdf",
            statementTitle: "Data protection impact assessment",
            concept: "DPIA",
            passage: "DPIA evidence.",
            explanation: "Selected for DPIA evidence."
          })
        })"""
    )

    assert result["safe"] == ""
    assert "<dt>Source document</dt><dd>gdpr.pdf</dd>" in result["card"]
    assert "<dt>Section or article</dt><dd>Not available</dd>" in result["card"]
    assert "gdpr.pdf / gdpr.pdf" not in result["card"]


def test_retrieval_scores_use_qualitative_labels_not_confidence_percentages() -> None:
    result = _run_policy_probe(
        """({
          high: api.formatScore(527),
          medium: api.formatScore(0.5),
          low: api.formatScore(0.2)
        })"""
    )

    assert result == {
        "high": "High match",
        "medium": "Medium match",
        "low": "Low match",
    }


def test_graph_labeling_and_regulation_split_are_demo_safe() -> None:
    result = _run_policy_probe(
        """(() => {
          const nodes = api.normalizeGraphNodes([
            { id: "privacy", label: "Privacy Controls GDPR", type: "control" },
            { id: "article", label: "Article 35 GDPR", type: "evidence" }
          ]);
          const graph = api.splitOverloadedRegulationNodes(nodes, api.normalizeGraphEdges([
            { source: "privacy", target: "article", label: "supported by evidence" }
          ]));
          return {
            illustrative: api.graphModeLabel({ meta: { source: "illustrative_demo_graph" } }, false),
            retrieved: api.graphModeLabel({ meta: { source: "live_retrieval_rows" } }, false),
            labels: graph.nodes.map((node) => `${node.type}:${node.label}`).sort(),
            edgeLabels: graph.edges.map((edge) => edge.label).sort()
          };
        })()"""
    )

    assert result["illustrative"] == "Illustrative graph"
    assert result["retrieved"] == "Retrieved evidence graph"
    assert "regulation:GDPR" in result["labels"]
    assert "control:Privacy Controls" in result["labels"]
    assert "evidence:Article 35" in result["labels"]
    assert "requires" in result["edgeLabels"]
    assert "cites" in result["edgeLabels"]


def test_graph_semantics_type_citations_and_sources_defensibly() -> None:
    result = _run_policy_probe(
        """(() => {
          const nodes = api.normalizeGraphNodes([
            { id: "article", label: "Art. 9", type: "system" },
            { id: "gdpr", label: "gdpr.pdf", type: "concept" },
            { id: "ai", label: "Mental Health AI", type: "system" }
          ]);
          return Object.fromEntries(nodes.map((node) => [node.id, `${node.type}:${node.label}`]));
        })()"""
    )

    assert result["article"] == "evidence:Art. 9"
    assert result["gdpr"] == "regulation:GDPR"
    assert result["ai"] == "system:Mental Health AI"


def test_graph_relationship_labels_are_cleaned_for_demo_story() -> None:
    result = _run_policy_probe(
        """({
          requiresGdpr: api.visibleEdgeLabel({ label: "requires GDPR" }, { type: "regulation" }, { type: "control" }, false),
          supportsBias: api.visibleEdgeLabel({ label: "supports Bias" }, { type: "control" }, { type: "risk" }, true),
          classifiedArticle: api.visibleEdgeLabel({ label: "classified as Art. 9" }, { type: "system" }, { type: "evidence" }, false),
          cites: api.visibleEdgeLabel({ label: "cites" }, { type: "regulation" }, { type: "evidence" }, false)
        })"""
    )

    assert result == {
        "requiresGdpr": "requires",
        "supportsBias": "",
        "classifiedArticle": "classified as",
        "cites": "cites",
    }


def test_fullscreen_graph_layout_uses_available_canvas_space() -> None:
    result = _run_policy_probe(
        """(() => {
          const nodes = api.normalizeGraphNodes([
            { id: "ai", label: "Mental Health AI", type: "system" },
            { id: "data", label: "Health Data", type: "concept" },
            { id: "high", label: "High-Risk AI System", type: "concept" },
            { id: "gdpr", label: "GDPR", type: "regulation" },
            { id: "ai-act", label: "EU AI Act", type: "regulation" },
            { id: "risk", label: "Risk Management", type: "control" },
            { id: "oversight", label: "Human Oversight", type: "control" },
            { id: "docs", label: "Technical Documentation", type: "control" },
            { id: "art9", label: "Article 9", type: "evidence" }
          ]);
          const layout = api.layoutGraph(nodes, {
            question: "What safeguards are needed when an AI system generates mental health risk scores?",
            graph: { meta: { source: "illustrative_demo_graph" } }
          }, { width: 1600, height: 900, topPadding: 120, bottomPadding: 120, sidePadding: 160 });
          const xs = layout.map((node) => node.x);
          const ys = layout.map((node) => node.y);
          return {
            xSpread: Math.max(...xs) - Math.min(...xs),
            ySpread: Math.max(...ys) - Math.min(...ys),
            minX: Math.min(...xs),
            maxX: Math.max(...xs)
          };
        })()"""
    )

    assert result["xSpread"] >= 1100
    assert result["ySpread"] >= 300
    assert result["minX"] >= 140
    assert result["maxX"] <= 1460


def test_dense_graph_hides_secondary_edge_labels_by_default() -> None:
    result = _run_policy_probe(
        """({
          requires: api.visibleEdgeLabel({ label: "requires" }, { type: "regulation" }, { type: "control" }, false),
          regulated: api.visibleEdgeLabel({ label: "regulated by" }, { type: "concept" }, { type: "regulation" }, false),
          cites: api.visibleEdgeLabel({ label: "cites" }, { type: "evidence" }, { type: "regulation" }, false),
          supported: api.visibleEdgeLabel({ label: "supported by" }, { type: "control" }, { type: "evidence" }, false),
          ranked: api.visibleEdgeLabel({ label: "ranked_for_question" }, { type: "question" }, { type: "statement" }, false)
        })"""
    )

    assert result == {
        "requires": "requires",
        "regulated": "regulated by",
        "cites": "cites",
        "supported": "",
        "ranked": "",
    }


def test_timeout_state_is_scheduled_after_configured_duration() -> None:
    script = dedent(
        f"""
        const fs = require("fs");
        const vm = require("vm");
        const elements = new Map();
        function element(selector) {{
          if (!elements.has(selector)) {{
            elements.set(selector, {{
              innerHTML: "",
              textContent: "",
              disabled: false,
              dataset: {{}},
              addEventListener() {{}},
            }});
          }}
          return elements.get(selector);
        }}
        const timeoutCalls = [];
        const context = {{
          console: {{ info() {{}}, log() {{}}, error() {{}} }},
          window: {{}},
          document: {{ querySelector: element, addEventListener() {{}} }},
          localStorage: {{ getItem() {{ return "[]"; }}, setItem() {{}} }},
          crypto: {{ randomUUID() {{ return "test-id"; }} }},
          AbortController,
          setTimeout(callback, delay) {{ timeoutCalls.push(delay); return 1; }},
          clearTimeout() {{}},
          fetch() {{ return new Promise(() => {{}}); }},
        }};
        vm.createContext(context);
        vm.runInContext(fs.readFileSync({json.dumps(str(APP_JS))}, "utf8"), context);
        context.window.__KEP_DEMO_TEST__.fetchAnswer("What is a DPIA?", 60000);
        process.stdout.write(JSON.stringify(timeoutCalls));
        """
    )

    output = subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)

    assert json.loads(output) == [60000]
