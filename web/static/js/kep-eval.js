// ============================================================================
// kep-eval.js — progressive enhancement for the Evaluation Report page.
// Server renders the full table; this only adds row-expand + column sorting.
// No data fetching, no metric computation.
// ============================================================================
(function () {
  "use strict";

  // --- render answers as markdown -------------------------------------------
  // Server renders the raw answer text into `.eval-answer`; convert in place.
  (function renderAnswers() {
    if (typeof marked === "undefined") return; // CDN unavailable → leave raw text
    marked.setOptions({ gfm: true, breaks: false });
    document.querySelectorAll(".eval-answer").forEach((el) => {
      const raw = el.textContent.trim();
      if (!raw || raw === "—") return;
      el.innerHTML = marked.parse(raw);
    });
  })();

  const table = document.getElementById("evalTable");
  if (!table) return;

  const tbody = table.tBodies[0];

  // --- expandable detail rows ------------------------------------------------
  function detailFor(id) {
    return tbody.querySelector('.eval-detail[data-detail="' + CSS.escape(id) + '"]');
  }
  tbody.addEventListener("click", (e) => {
    const row = e.target.closest(".eval-row");
    if (!row) return;
    const detail = detailFor(row.getAttribute("data-id"));
    if (!detail) return;
    const open = detail.hasAttribute("hidden") ? false : true;
    if (open) { detail.setAttribute("hidden", ""); row.classList.remove("open"); }
    else { detail.removeAttribute("hidden"); row.classList.add("open"); }
  });

  // --- column sorting --------------------------------------------------------
  function pairs() {
    // [{summary, detail}] preserving DOM linkage.
    const out = [];
    tbody.querySelectorAll(".eval-row").forEach((summary) => {
      out.push({ summary, detail: detailFor(summary.getAttribute("data-id")) });
    });
    return out;
  }

  let sortState = { col: -1, dir: -1 }; // dir: -1 desc, 1 asc

  function sortByColumn(th) {
    const col = th.cellIndex;
    const dir = sortState.col === col ? -sortState.dir : -1; // default first click = desc
    sortState = { col, dir };

    const rows = pairs();
    rows.sort((a, b) => {
      const av = parseFloat(a.summary.cells[col].getAttribute("data-v"));
      const bv = parseFloat(b.summary.cells[col].getAttribute("data-v"));
      const an = isNaN(av) ? -Infinity : av;
      const bn = isNaN(bv) ? -Infinity : bv;
      return (an - bn) * dir;
    });

    rows.forEach(({ summary, detail }) => {
      tbody.appendChild(summary);
      if (detail) tbody.appendChild(detail);
    });

    table.querySelectorAll("th.sortable").forEach((h) => h.classList.remove("sort-asc", "sort-desc"));
    th.classList.add(dir === 1 ? "sort-asc" : "sort-desc");
  }

  table.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => sortByColumn(th));
  });
})();
