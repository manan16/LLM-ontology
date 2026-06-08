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


def test_demo_preset_uses_cache_when_cache_enabled() -> None:
    result = _run_policy_probe(
        'api.buildRequestPolicy("What risks should be mitigated before deployment?")'
    )

    assert result["isDemoPreset"] is True
    assert result["cacheHit"] is True
    assert result["cacheEnabled"] is True
    assert result["useCache"] is True
    assert result["livePipelineCalled"] is False
    assert result["timeoutMs"] == 8000


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
          cached: api.responseSourceLabel({ responseSource: "cached_demo" }),
          live: api.responseSourceLabel({ responseSource: "live_pipeline" }),
          failed: api.responseSourceLabel({ responseSource: "live_failed" }),
          timeout: api.responseSourceLabel({ responseSource: "live_timeout" }),
          fallback: api.responseSourceLabel({ responseSource: "fallback_cache" })
        })"""
    )

    assert result == {
        "cached": "Cached demo response",
        "live": "Live pipeline run",
        "failed": "Live retrieval failed",
        "timeout": "Live retrieval timed out",
        "fallback": "Fallback cache shown",
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
