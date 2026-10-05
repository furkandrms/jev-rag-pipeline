const viewUploadEl = document.getElementById("view-upload");
const viewChatEl = document.getElementById("view-chat");
const chatEl = document.getElementById("chat");
const composer = document.getElementById("composer");
const input = document.getElementById("query-input");
const sendBtn = document.getElementById("send-btn");
const template = document.getElementById("message-template");
const badgesEl = document.getElementById("backend-badges");
const uploadDropEl = document.getElementById("upload-drop");
const uploadLabelEl = document.getElementById("upload-label");
const uploadStatusEl = document.getElementById("upload-status");
const fileInputEl = document.getElementById("file-input");
const docChipNameEl = document.getElementById("doc-chip-name");
const docChipCountEl = document.getElementById("doc-chip-count");
const docChangeBtnEl = document.getElementById("doc-change-btn");
const chunksPanelEl = document.getElementById("chunks-panel");
const chunksTabCountEl = document.getElementById("chunks-tab-count");
const metricsStripEl = document.getElementById("metrics-strip");
const activityLogEl = document.getElementById("activity-log");
const sideStepperEl = document.getElementById("side-stepper");
const modeToggleEl = document.getElementById("mode-toggle");
const singleProviderViewEl = document.getElementById("single-provider-view");
const compareViewEl = document.getElementById("compare-view");
const compareIntroEl = document.getElementById("compare-intro");
const compareColumnsEl = document.getElementById("compare-columns");
const compareSummaryEl = document.getElementById("compare-summary");
const compareComposer = document.getElementById("compare-composer");
const compareInput = document.getElementById("compare-query-input");
const compareSendBtn = document.getElementById("compare-send-btn");
const inspTabEls = document.querySelectorAll(".insp-tab");
const inspPanels = {
  pipeline: document.getElementById("insp-panel-pipeline"),
  chunks: document.getElementById("insp-panel-chunks"),
  log: document.getElementById("insp-panel-log"),
};

let thresholds = {
  relevance_threshold: 0.5,
  sufficiency_threshold: 0.6,
  grounding_threshold: 0.5,
  grounding_min_coverage: 0.8,
  intent_threshold: 0.5,
  ambiguity_threshold: 0.6,
  caveat_threshold: 0.5,
};

const STAGES = [
  { key: "intent", label: "Intent" },
  { key: "retrieval", label: "Retrieve" },
  { key: "relevance", label: "Relevance" },
  { key: "clarify", label: "Clarify" },
  { key: "sufficiency", label: "Sufficiency" },
  { key: "generation", label: "Generate" },
  { key: "grounding", label: "Ground" },
  { key: "caveat", label: "Caveat" },
];

const ACTION_LABELS = {
  answered: "Answered",
  answered_with_caveat: "Answered — with caveat",
  insufficient_context: "Insufficient context",
  ungrounded_answer_flagged: "Ungrounded — flagged",
  clarify: "Needs clarification",
  escalate: "Escalated to a human",
  refuse: "Refused",
};

// A raw answer containing unrendered LaTeX delimiters reads as garbled text
// instead of math -- called out explicitly in compare mode's raw (guard
// off) column rather than silently rendered as if it were normal prose.
const LATEX_RE = /\\\(|\\\)|\\\[|\\\]|\$\$/;

// Real, session-local counters -- no fabricated numbers. Reset on page
// reload since they're not persisted server-side (see README).
const sessionMetrics = {
  queries: 0,
  cacheHits: 0,
  inputTokens: 0,
  outputTokens: 0,
  decisionCalls: 0,
  latencies: [],
};

// A per-browser (not per-account) session id, just enough identity to keep
// one visitor's uploaded document out of another's -- minted once and kept
// in localStorage, never sent anywhere but this app's own backend.
function getSessionId() {
  const key = "jevrag_session_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}
const sessionId = getSessionId();

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function pct(p) {
  return `${Math.round(p * 100)}%`;
}

/* --- Pipeline stage (inline bubble stepper) ------------------------------- */

function stepperHtml() {
  return STAGES.map(
    (s) => `<span class="stepper-item" data-stage="${s.key}"><span class="stepper-dot"></span>${s.label}</span>`
  ).join("");
}

/* --- Pipeline stage (sidebar timeline) ------------------------------------ */
// A fuller, vertical echo of the same stepper -- dot, name, a short result
// string (the same text the activity log prints), and timing -- so the
// current/last query's decisions are visible without scrolling back up to
// the message bubble.

function timelineRowHtml(s) {
  return `
    <div class="timeline-row" data-stage="${s.key}">
      <span class="timeline-dot"></span>
      <span class="timeline-body">
        <span class="timeline-name">${s.label}</span>
        <span class="timeline-result mono"></span>
      </span>
      <span class="timeline-ms mono"></span>
    </div>`;
}

function renderSideTimelineIdle() {
  sideStepperEl.innerHTML = STAGES.map(timelineRowHtml).join("");
}

function setSideTimelineStage(key, state, ms, resultText) {
  const el = sideStepperEl.querySelector(`.timeline-row[data-stage="${key}"]`);
  if (!el) return;
  el.className = `timeline-row ${state}`;
  el.querySelector(".timeline-result").textContent =
    resultText || (state === "active" ? "running…" : state === "skipped" ? "skipped" : "");
  el.querySelector(".timeline-ms").textContent = ms !== undefined ? `${ms}ms` : "";
}

/* --- Inspector tabs (Pipeline / Chunks / Log) ------------------------------ */

function setInspectorTab(tab) {
  inspTabEls.forEach((btn) => {
    const active = btn.dataset.tab === tab;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-selected", active ? "true" : "false");
  });
  Object.entries(inspPanels).forEach(([key, el]) => {
    el.hidden = key !== tab;
  });
}

inspTabEls.forEach((btn) => {
  btn.addEventListener("click", () => setInspectorTab(btn.dataset.tab));
});

/* --- Activity log ----------------------------------------------------------- */
// A live, timestamped record of what the system actually did -- every line
// here corresponds to a real event the backend reported, not a simulated
// status message.

function logTime() {
  const d = new Date();
  return d.toTimeString().slice(0, 8) + "." + String(d.getMilliseconds()).padStart(3, "0");
}

function logLine(icon, html) {
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="log-time">${logTime()}</span><span class="log-icon ${icon}">${
    { ok: "✓", warn: "●", fail: "✗", info: "›", cache: "⚡" }[icon] || "›"
  }</span><span class="log-text">${html}</span>`;
  activityLogEl.appendChild(line);
  activityLogEl.scrollTop = activityLogEl.scrollHeight;
}

async function loadInfo() {
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    if (data.thresholds) thresholds = data.thresholds;
    badgesEl.innerHTML = `
      <span class="backend-badge">generate: <b>${escapeHtml(data.generate_model || data.generate_fn)}</b></span>
      <span class="backend-badge">decisions: <b>${escapeHtml(data.decision_model)}</b></span>
    `;
  } catch (e) {
    badgesEl.innerHTML = `<span class="backend-badge">backend info unavailable</span>`;
  }
}

/* --- Views ---------------------------------------------------------------- */

function showUploadView() {
  viewUploadEl.hidden = false;
  viewChatEl.hidden = true;
}

function showChatView(filename, chunkCount, chunks) {
  viewUploadEl.hidden = true;
  viewChatEl.hidden = false;
  docChipNameEl.textContent = filename;
  docChipCountEl.textContent = `${chunkCount} chunks`;
  renderChunksPanel(chunks || []);
  renderMetricsStrip();
  renderSideTimelineIdle();
  setInspectorTab("pipeline");
  setMode("single");
  input.focus();
}

/* --- Mode toggle (single provider / compare) --------------------------------- */

function setMode(mode) {
  modeToggleEl.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  singleProviderViewEl.hidden = mode !== "single";
  compareViewEl.hidden = mode !== "compare";
  if (mode === "compare") compareInput.focus();
}

modeToggleEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".mode-btn");
  if (btn) setMode(btn.dataset.mode);
});

async function checkExistingDocument() {
  try {
    const res = await fetch("/api/document-status", { headers: { "X-Session-Id": sessionId } });
    const data = await res.json();
    if (data.uploaded) {
      showChatView(data.filename, data.chunk_count, data.chunks);
      logLine("info", `resumed session with <b>${escapeHtml(data.filename)}</b> · ${data.chunk_count} chunks`);
    } else {
      showUploadView();
    }
  } catch (e) {
    showUploadView();
  }
}

/* --- Empty state (no messages yet) ------------------------------------------ */

function emptyStateHtml() {
  return `
    <div class="empty-state" id="empty-state">
      <h2>Ask anything about this document</h2>
      <p>Every answer arrives with the decisions behind it: what was retrieved, what was kept, and whether each claim is supported.</p>
      <div class="suggestions" id="suggestions">
        <button type="button" class="sug" data-q="What is this document about?">What is this document about?</button>
        <button type="button" class="sug" data-q="What are the key findings or conclusions?">What are the key findings or conclusions?</button>
        <button type="button" class="sug" data-q="Summarize the main sections of this document.">Summarize the main sections of this document.</button>
      </div>
    </div>`;
}

// Event delegation on the chat container itself -- it persists across
// resets (only its innerHTML changes between documents/queries), so one
// listener covers every empty-state render without re-attaching.
chatEl.addEventListener("click", (e) => {
  const btn = e.target.closest(".sug");
  if (btn) sendQuery(btn.dataset.q);
});

/* --- Upload ----------------------------------------------------------------- */

async function uploadFile(file) {
  uploadLabelEl.textContent = `Uploading ${file.name}…`;
  uploadStatusEl.textContent = "";
  uploadStatusEl.className = "upload-status";

  try {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Session-Id": sessionId },
      body: form,
    });
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Upload failed");
    }

    // Fresh document for this session -- clear conversation + metrics so
    // old answers/stats don't sit alongside a different document's context.
    chatEl.innerHTML = emptyStateHtml();
    Object.assign(sessionMetrics, {
      queries: 0, cacheHits: 0, inputTokens: 0, outputTokens: 0, decisionCalls: 0, latencies: [],
    });
    activityLogEl.innerHTML = "";
    uploadLabelEl.textContent = `Drop a .txt / .md / .pdf here, or click to choose a file`;
    uploadStatusEl.textContent = "";
    showChatView(data.filename, data.chunk_count, data.chunks);
    renderMetricsStrip();
    logLine("ok", `document indexed: <b>${escapeHtml(data.filename)}</b> · ${data.chunk_count} chunks`);
  } catch (e) {
    uploadLabelEl.textContent = `Drop a .txt / .md / .pdf here, or click to choose a file`;
    uploadStatusEl.textContent = e.message || "Upload failed";
    uploadStatusEl.className = "upload-status error";
  }
}

fileInputEl.addEventListener("change", () => {
  if (fileInputEl.files[0]) uploadFile(fileInputEl.files[0]);
});

["dragenter", "dragover"].forEach((evt) =>
  uploadDropEl.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadDropEl.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  uploadDropEl.addEventListener(evt, (e) => {
    e.preventDefault();
    uploadDropEl.classList.remove("dragover");
  })
);
uploadDropEl.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

docChangeBtnEl.addEventListener("click", async () => {
  docChangeBtnEl.disabled = true;
  try {
    await fetch("/api/document", { method: "DELETE", headers: { "X-Session-Id": sessionId } });
  } catch (e) {
    // Even if the request fails, still drop back to the upload view --
    // the next upload will overwrite whatever's left server-side anyway.
  }
  fileInputEl.value = "";
  chatEl.innerHTML = emptyStateHtml();
  activityLogEl.innerHTML = "";
  Object.assign(sessionMetrics, {
    queries: 0, cacheHits: 0, inputTokens: 0, outputTokens: 0, decisionCalls: 0, latencies: [],
  });
  docChangeBtnEl.disabled = false;
  showUploadView();
});

/* --- Chunk panel -------------------------------------------------------------- */

function renderChunksPanel(chunks) {
  chunksTabCountEl.textContent = chunks.length ? String(chunks.length) : "";
  if (!chunks.length) {
    chunksPanelEl.innerHTML = `<div class="chunks-empty">No chunks indexed.</div>`;
    return;
  }
  chunksPanelEl.innerHTML = chunks
    .map(
      (text, i) => `
      <div class="chunk-card">
        <span class="chunk-card-index">CHUNK ${i}</span>
        <div class="chunk-card-text">${escapeHtml(text)}</div>
      </div>`
    )
    .join("");
}

/* --- Metrics strip -------------------------------------------------------------- */

function renderMetricsStrip() {
  if (sessionMetrics.queries === 0) {
    metricsStripEl.innerHTML = `<div class="metrics-empty">Ask a question to see live metrics.</div>`;
    return;
  }
  const avgLatency = sessionMetrics.latencies.length
    ? Math.round(sessionMetrics.latencies.reduce((a, b) => a + b, 0) / sessionMetrics.latencies.length)
    : 0;
  metricsStripEl.innerHTML = `
    <div class="metric-item" title="Questions asked this session"><span class="metric-value">${sessionMetrics.queries}</span><span class="metric-label">Queries</span></div>
    <div class="metric-item" title="Relevance/sufficiency/grounding judgments made by the decision model"><span class="metric-value">${sessionMetrics.decisionCalls}</span><span class="metric-label">Decision calls</span></div>
    <div class="metric-item" title="Input + output tokens used by the generation model only -- decision-model calls aren't counted"><span class="metric-value">${sessionMetrics.inputTokens + sessionMetrics.outputTokens}</span><span class="metric-label">Generation tokens</span></div>
    <div class="metric-item" title="Average end-to-end time per query, including cache hits"><span class="metric-value">${avgLatency}ms</span><span class="metric-label">Avg latency</span></div>
  `;
}

/* --- Chat messages -------------------------------------------------------------- */

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="msg-bubble"></div>`;
  div.querySelector(".msg-bubble").textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addAssistantMessage() {
  const div = document.createElement("div");
  div.className = "msg msg-assistant";
  div.innerHTML = template.content.cloneNode(true).firstElementChild.innerHTML;
  div.querySelector(".stepper").innerHTML = stepperHtml();
  div.querySelector(".msg-bubble").innerHTML = `<span class="dots">thinking</span>`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function setStageState(container, key, state, ms) {
  const el = container.querySelector(`.stepper-item[data-stage="${key}"]`);
  if (!el) return;
  el.className = `stepper-item ${state}`;
  const label = STAGES.find((s) => s.key === key).label;
  el.innerHTML = `<span class="stepper-dot"></span>${label}${
    ms !== undefined ? ` <span class="stepper-ms">${ms}ms</span>` : ""
  }`;
}

// A chunk id is "<uploaded filename>-<index>" (see pipeline_setup.py), which
// is often wider than the fixed-width label column -- showing the raw id
// truncates mid-filename and leaves nothing recognizable. The index alone
// ("Chunk 3") fits reliably; the full id is still available on hover.
function chunkShortLabel(chunkId) {
  const match = /-(\d+)$/.exec(chunkId);
  return match ? `Chunk ${match[1]}` : chunkId;
}

function buildRelevanceStage(relevance) {
  if (!relevance.length) {
    return `<div class="stage-empty">No chunks retrieved.</div>`;
  }
  return [...relevance]
    .sort((a, b) => b.probability - a.probability)
    .map((r) => {
      const kept = r.kept;
      // Kept despite scoring below the relevance threshold means this
      // chunk was force-kept outside the normal per-chunk judgment (a
      // document anchor, or a neighbor pulled in next to a high-scoring
      // chunk) -- surfaced as "pinned" rather than a plain "kept" so it
      // doesn't read as a scoring inconsistency.
      const pinned = kept && r.probability < thresholds.relevance_threshold;
      const flagClass = pinned ? "pinned" : kept ? "kept" : "dropped";
      const flagText = pinned ? "pinned" : kept ? "kept" : "dropped";
      const flagTitle = pinned
        ? `Kept despite scoring ${r.probability.toFixed(2)}, below the relevance threshold (${thresholds.relevance_threshold.toFixed(2)}) — likely force-kept as a document anchor or neighbor-expansion chunk, not a relevance judgment.`
        : "";
      const snippet = r.text.length > 90 ? r.text.slice(0, 90) + "…" : r.text;
      // escapeHtml() on chunk_id too: it's derived from the uploaded
      // filename, which is attacker-controllable -- never trust it raw
      // in HTML, even though it's just an id string.
      const safeId = escapeHtml(r.chunk_id);
      const shortLabel = escapeHtml(chunkShortLabel(r.chunk_id));
      return `
        <div>
          <div class="chunk-row">
            <span class="chunk-id" title="${safeId}">${shortLabel}</span>
            <div class="chunk-bar-track">
              <div class="chunk-bar-fill ${kept ? "kept" : "dropped"}" style="width:${pct(r.probability)}"></div>
            </div>
            <span class="chunk-prob">${r.probability.toFixed(2)}</span>
            <span class="chunk-flag ${flagClass}"${flagTitle ? ` title="${escapeHtml(flagTitle)}"` : ""}>${flagText}</span>
          </div>
          <div class="chunk-snippet">${escapeHtml(snippet)}</div>
        </div>`;
    })
    .join("");
}

function buildSufficiencyStage(sufficiency) {
  if (!sufficiency) {
    return `<div class="stage-empty">Skipped (pipeline stopped at an earlier stage).</div>`;
  }
  const pass = sufficiency.sufficient;
  const label = pass ? "sufficient" : sufficiency.partial ? "partial" : "insufficient";
  const barClass = pass ? "pass" : sufficiency.partial ? "partial" : "fail";
  return `
    <div class="suff-row">
      <div class="suff-bar-track">
        <div class="suff-threshold" style="left:${pct(thresholds.sufficiency_threshold)}" title="Threshold: ${thresholds.sufficiency_threshold.toFixed(2)} — needs to score at least this high to generate an answer"></div>
        <div class="suff-bar-fill ${barClass}" style="width:${pct(sufficiency.probability)}"></div>
      </div>
      <span class="suff-label">${label} · ${sufficiency.probability.toFixed(2)}</span>
    </div>
    ${
      sufficiency.partial
        ? `<div class="chunk-snippet" style="margin-top:8px;">Context only partially covers the question — generation was allowed to answer what it can and flag the rest, instead of refusing outright.</div>`
        : ""
    }
    ${sufficiency.reason && !sufficiency.partial ? `<div class="chunk-snippet" style="margin-top:8px;">${escapeHtml(sufficiency.reason)}</div>` : ""}
  `;
}

function buildGenerationStage(usage) {
  if (!usage || (!usage.input_tokens && !usage.output_tokens)) {
    return `<div class="stage-empty">No usage data reported by this backend.</div>`;
  }
  return `<div class="chunk-snippet">input: ${usage.input_tokens} tokens · output: ${usage.output_tokens} tokens</div>`;
}

function buildGroundingStage(grounding) {
  if (!grounding) {
    return `<div class="stage-empty">Skipped (generation didn't run).</div>`;
  }
  const rows = grounding.claims
    .map(
      (c) => `
      <div class="claim-row">
        <div class="claim-head">
          <span class="claim-flag ${c.supported ? "supported" : "unsupported"}">${c.supported ? "supported" : "unsupported"}</span>
          <span class="claim-prob">p=${c.probability.toFixed(2)}</span>
        </div>
        <div class="claim-text">${escapeHtml(c.claim)}</div>
      </div>`
    )
    .join("");
  return `
    <div class="chunk-snippet" style="margin-bottom:8px;">coverage: ${pct(grounding.coverage)} (needs ≥ ${pct(thresholds.grounding_min_coverage)})</div>
    ${rows}
  `;
}

function buildIntentStage(intent) {
  if (!intent) {
    return `<div class="stage-empty">Not evaluated for this response.</div>`;
  }
  return `
    <div class="chunk-snippet">action: ${escapeHtml(intent.action)} · p=${intent.probability.toFixed(2)}</div>
    ${intent.reason ? `<div class="chunk-snippet" style="margin-top:6px;">${escapeHtml(intent.reason)}</div>` : ""}
  `;
}

function buildClarifyStage(clarify) {
  if (!clarify) {
    return `<div class="stage-empty">Not evaluated for this response.</div>`;
  }
  return `
    <div class="chunk-snippet">needs clarification: ${clarify.needs_clarification ? "yes" : "no"} · p=${clarify.probability.toFixed(2)}</div>
    ${clarify.reason ? `<div class="chunk-snippet" style="margin-top:6px;">${escapeHtml(clarify.reason)}</div>` : ""}
  `;
}

function buildCaveatStage(caveat) {
  if (!caveat) {
    return `<div class="stage-empty">Not evaluated for this response.</div>`;
  }
  return `
    <div class="chunk-snippet">has caveat: ${caveat.has_caveat ? "yes" : "no"} · p=${caveat.probability.toFixed(2)}</div>
    ${caveat.reason ? `<div class="chunk-snippet" style="margin-top:6px;">${escapeHtml(caveat.reason)}</div>` : ""}
  `;
}

// A box citing exactly which chunks the grounding check traced each claim
// back to, shown outside the trace toggle (always visible once an answer
// is grounded) rather than buried a click away. Only renders when the
// backend actually reported supporting chunk ids for at least one claim --
// it stays empty rather than inventing a citation that wasn't made.
function buildSupportedBy(report) {
  if (!report.grounding || !report.grounding.claims) return "";
  const ids = new Set();
  report.grounding.claims.forEach((c) => (c.supporting_chunk_ids || []).forEach((id) => ids.add(id)));
  if (!ids.size) return "";
  const chunkMap = new Map((report.relevance || []).map((r) => [r.chunk_id, r.text]));
  const parts = [...ids].map((id) => {
    const label = escapeHtml(chunkShortLabel(id));
    const text = chunkMap.get(id);
    const snippet = text ? `: "${escapeHtml(text.length > 140 ? text.slice(0, 140) + "…" : text)}"` : "";
    return `${label}${snippet}`;
  });
  return `<span class="supported-by-label">SUPPORTED BY</span>${parts.join(" · ")}`;
}

function renderTracePanel(container, report) {
  const tracePanel = container.querySelector(".trace-panel");
  const relevancePass = report.relevance.some((r) => r.kept);
  const suffPass = report.sufficiency ? report.sufficiency.sufficient : null;
  const groundPass = report.grounding ? report.grounding.grounded : null;
  const intentPass = report.intent ? report.intent.action === "proceed" : null;
  const clarifyPass = report.clarify ? !report.clarify.needs_clarification : null;
  const caveatPass = report.caveat ? !report.caveat.has_caveat : null;

  tracePanel.innerHTML = `
    ${
      report.rewritten_query
        ? `<div class="trace-rewrite">Resolved as: <b>${escapeHtml(report.rewritten_query)}</b> <span class="trace-rewrite-note">(used for retrieval, since the question as asked relies on the earlier conversation)</span></div>`
        : ""
    }
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${intentPass === null ? "" : intentPass ? "pass" : "fail"}"></span>Intent</div>
      ${buildIntentStage(report.intent)}
    </div>
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${relevancePass ? "pass" : "fail"}"></span>Relevance</div>
      ${buildRelevanceStage(report.relevance)}
    </div>
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${clarifyPass === null ? "" : clarifyPass ? "pass" : "fail"}"></span>Clarify</div>
      ${buildClarifyStage(report.clarify)}
    </div>
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${suffPass === null ? "" : suffPass ? "pass" : "fail"}"></span>Sufficiency</div>
      ${buildSufficiencyStage(report.sufficiency)}
    </div>
    ${
      report.metrics && report.metrics.generation_usage
        ? `<div class="trace-stage">
             <div class="trace-stage-title"><span class="stage-dot pass"></span>Generation</div>
             ${buildGenerationStage(report.metrics.generation_usage)}
           </div>`
        : ""
    }
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${groundPass === null ? "" : groundPass ? "pass" : "fail"}"></span>Grounding</div>
      ${buildGroundingStage(report.grounding)}
    </div>
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${caveatPass === null ? "" : caveatPass ? "pass" : "fail"}"></span>Caveat</div>
      ${buildCaveatStage(report.caveat)}
    </div>
  `;

  const toggle = container.querySelector(".trace-toggle");
  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    tracePanel.classList.toggle("open");
  });
}

function finalizeMessage(container, report) {
  const bubble = container.querySelector(".msg-bubble");
  const badge = container.querySelector(".action-badge");
  const textEl = container.querySelector(".msg-text");
  const supportedByEl = container.querySelector(".supported-by");
  badge.textContent = ACTION_LABELS[report.action] || report.action;
  badge.classList.add("show", report.action);
  bubble.classList.toggle("verdict-good", report.action === "answered" || report.action === "answered_with_caveat");
  textEl.textContent = report.answer;
  if (supportedByEl) supportedByEl.innerHTML = buildSupportedBy(report);
  renderTracePanel(container, report);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function recordMetrics(report, totalMs, cached) {
  sessionMetrics.queries += 1;
  sessionMetrics.latencies.push(totalMs);
  if (cached) sessionMetrics.cacheHits += 1;
  if (report.metrics) {
    sessionMetrics.decisionCalls += report.metrics.decision_calls || 0;
    if (report.metrics.generation_usage) {
      sessionMetrics.inputTokens += report.metrics.generation_usage.input_tokens || 0;
      sessionMetrics.outputTokens += report.metrics.generation_usage.output_tokens || 0;
    }
  }
  renderMetricsStrip();
}

/* --- Streaming chat -------------------------------------------------------------- */

async function streamChat(query, onEvent) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
    body: JSON.stringify({ query }),
  });
  if (!res.ok || !res.body) {
    const data = await res.json().catch(() => ({}));
    onEvent("error", { detail: data.detail || "Request failed." });
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const eventMatch = raw.match(/^event: (.+)$/m);
      const dataMatch = raw.match(/^data: (.+)$/m);
      if (eventMatch && dataMatch) {
        onEvent(eventMatch[1], JSON.parse(dataMatch[1]));
      }
    }
  }
}

async function sendQuery(query) {
  const emptyState = document.getElementById("empty-state");
  if (emptyState) emptyState.remove();

  addUserMessage(query);
  const container = addAssistantMessage();
  renderSideTimelineIdle();

  // Mirrors every stepper update into the sidebar timeline too, so the
  // current stage is visible without scrolling back up to the message.
  const setStage = (key, state, ms, resultText) => {
    setStageState(container, key, state, ms);
    setSideTimelineStage(key, state, ms, resultText);
  };
  setStage("intent", "active");

  input.value = "";
  sendBtn.disabled = true;

  let bubbleCleared = false;
  const clearBubble = () => {
    if (!bubbleCleared) {
      container.querySelector(".msg-bubble").innerHTML = `<div class="action-badge"></div><div class="msg-text"></div><div class="supported-by"></div>`;
      bubbleCleared = true;
    }
  };

  logLine("info", `query: <b>${escapeHtml(query)}</b>`);

  try {
    await streamChat(query, (event, data) => {
      if (event === "error") {
        container.querySelector(".msg-bubble").textContent = data.detail;
        sideStepperEl.innerHTML = `<div class="stepper-empty">Last query failed.</div>`;
        logLine("fail", escapeHtml(data.detail));
        return;
      }
      if (event === "rewrite") {
        logLine(
          "info",
          `follow-up resolved to: <b>${escapeHtml(data.rewritten_query)}</b>`
        );
        return;
      }
      if (event === "cached") {
        const cacheHtml = `<span class="cache-badge">⚡ served from cache</span>`;
        container.querySelector(".stepper").innerHTML = cacheHtml;
        clearBubble();
        finalizeMessage(container, data);
        recordMetrics(data, data.total_elapsed_ms, true);
        logLine("cache", `cache hit — served in ${data.total_elapsed_ms}ms, pipeline skipped`);
        return;
      }
      if (STAGES.some((s) => s.key === event)) {
        setStage(event, "done", data.elapsed_ms, stageLogDetail(event, data));
        logLine("ok", `<b>${event}</b> ${stageLogDetail(event, data)} (${data.elapsed_ms}ms)`);

        // Mark the next stage "active" (pulsing) so the stepper visibly
        // progresses even though SSE only tells us about completions.
        const idx = STAGES.findIndex((s) => s.key === event);
        const next = STAGES[idx + 1];
        if (next) setStage(next.key, "active");

        if (event === "sufficiency" && !data.sufficiency.sufficient) {
          setStage("generation", "skipped");
          setStage("grounding", "skipped");
          logLine("warn", "generation + grounding skipped — context judged insufficient");
        }
        if (event === "intent" && data.intent.action !== "proceed") {
          STAGES.filter((s) => s.key !== "intent").forEach((s) => setStage(s.key, "skipped"));
          logLine("warn", `pipeline stopped — query ${data.intent.action}d before retrieval`);
        }
        if (event === "clarify" && data.clarify.needs_clarification) {
          setStage("sufficiency", "skipped");
          setStage("generation", "skipped");
          setStage("grounding", "skipped");
          setStage("caveat", "skipped");
          logLine("warn", "generation skipped — question needs clarification");
        }
        return;
      }
      if (event === "done") {
        clearBubble();
        finalizeMessage(container, data.report);
        recordMetrics(data.report, data.total_elapsed_ms, false);
        logLine(
          data.report.action === "answered" ? "ok" : "warn",
          `done — <b>${escapeHtml(data.report.action)}</b> (${data.total_elapsed_ms}ms total)`
        );
      }
    });
  } catch (e) {
    container.querySelector(".msg-bubble").textContent = "Something went wrong talking to the backend.";
    logLine("fail", "request failed");
  } finally {
    sendBtn.disabled = false;
  }
}

function stageLogDetail(event, data) {
  switch (event) {
    case "intent":
      return `action=${data.intent.action} p=${data.intent.probability.toFixed(2)}`;
    case "retrieval":
      return `fetched ${data.chunk_count} chunks`;
    case "relevance": {
      const kept = data.relevance.filter((r) => r.kept).length;
      return `kept ${kept}/${data.relevance.length} chunks`;
    }
    case "clarify":
      return `p=${data.clarify.probability.toFixed(2)} → ${data.clarify.needs_clarification ? "needs clarification" : "clear"}`;
    case "sufficiency":
      return `p=${data.sufficiency.probability.toFixed(2)} → ${data.sufficiency.sufficient ? "sufficient" : "insufficient"}`;
    case "generation":
      return data.usage && (data.usage.input_tokens || data.usage.output_tokens)
        ? `${data.usage.input_tokens}→${data.usage.output_tokens} tokens`
        : "answer generated";
    case "grounding": {
      const supported = data.grounding.claims.filter((c) => c.supported).length;
      return `${supported}/${data.grounding.claims.length} claims supported`;
    }
    case "caveat":
      return `p=${data.caveat.probability.toFixed(2)} → ${data.caveat.has_caveat ? "has caveat" : "clear"}`;
    default:
      return "";
  }
}

composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  sendQuery(q);
});

/* --- Compare mode: same question, one column per embedding provider,  --------
   raw (JEV-off) vs JEV-protected answer in each -- see /api/compare. */

function renderCompareColumn(colEl, data) {
  const jev = data.jev;
  colEl.innerHTML = `
    <div class="compare-col-header">
      <span class="compare-provider-name">${escapeHtml(data.provider_name)}</span>
      <span class="compare-chunk-count">${data.chunk_count} chunks retrieved</span>
    </div>
    <div class="compare-card compare-card-raw">
      <div class="compare-card-label">Raw answer <span class="compare-card-sublabel">— JEV off</span></div>
      <div class="compare-card-text"></div>
    </div>
    <div class="compare-arrow">↓ same retrieval, now guarded by JEV ↓</div>
    <div class="compare-card compare-card-jev">
      <div class="compare-card-label-row">
        <span class="compare-card-label">JEV-protected answer</span>
        <span class="action-badge show ${jev.action}">${escapeHtml(ACTION_LABELS[jev.action] || jev.action)}</span>
      </div>
      <div class="compare-card-text"></div>
      <button class="trace-toggle" type="button"><span class="chevron">›</span> Full reasoning trace</button>
      <div class="trace-panel"></div>
    </div>
  `;
  const rawTextEl = colEl.querySelector(".compare-card-raw .compare-card-text");
  rawTextEl.textContent = data.raw.answer;
  // A raw (guard-off) answer sometimes leaks unrendered LaTeX delimiters
  // straight from the model -- flagged rather than left to read as
  // garbled prose, without altering the backend's actual output.
  if (LATEX_RE.test(data.raw.answer || "")) {
    rawTextEl.classList.add("latex-raw");
    const note = document.createElement("div");
    note.className = "compare-latex-note";
    note.textContent = "Unrendered LaTeX leaked into the raw output";
    colEl.querySelector(".compare-card-raw").appendChild(note);
  }
  colEl.querySelector(".compare-card-jev .compare-card-text").textContent = jev.answer;
  // renderTracePanel() only needs a container with a .trace-panel and a
  // .trace-toggle child -- the same function single-provider mode uses.
  renderTracePanel(colEl.querySelector(".compare-card-jev"), jev);
}

function renderCompareSummary(providers) {
  const rows = providers
    .map((p) => {
      const jev = p.jev;
      const coverage = jev.grounding ? pct(jev.grounding.coverage) : "—";
      const suffP = jev.sufficiency ? jev.sufficiency.probability.toFixed(2) : "—";
      return `
        <tr>
          <td>${escapeHtml(p.provider_name)}</td>
          <td><span class="action-badge show ${jev.action}">${escapeHtml(ACTION_LABELS[jev.action] || jev.action)}</span></td>
          <td>${coverage}</td>
          <td>${suffP}</td>
        </tr>`;
    })
    .join("");
  compareSummaryEl.innerHTML = `
    <div class="compare-summary-table-wrap">
      <table class="compare-summary-table">
        <thead>
          <tr><th>Provider</th><th>JEV decision</th><th>Grounding coverage</th><th>Sufficiency p</th></tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

async function runCompare(query) {
  compareInput.value = "";
  compareSendBtn.disabled = true;
  compareIntroEl.hidden = true;
  compareColumnsEl.innerHTML = `<div class="compare-loading"><span class="dots">Running both providers in parallel — two full retrieval + JEV passes, slower than a single query. First comparison on a large document also builds its per-provider index, which can take a while.</span></div>`;
  compareSummaryEl.innerHTML = "";

  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();
    if (!res.ok) {
      compareColumnsEl.innerHTML = `<div class="compare-error">${escapeHtml(data.detail || "Request failed.")}</div>`;
      return;
    }
    compareColumnsEl.innerHTML = "";
    data.providers.forEach((p) => {
      const col = document.createElement("div");
      col.className = "compare-col";
      compareColumnsEl.appendChild(col);
      renderCompareColumn(col, p);
    });
    renderCompareSummary(data.providers);
  } catch (e) {
    compareColumnsEl.innerHTML = `<div class="compare-error">Something went wrong talking to the backend.</div>`;
  } finally {
    compareSendBtn.disabled = false;
  }
}

compareComposer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = compareInput.value.trim();
  if (!q) return;
  runCompare(q);
});

checkExistingDocument();
loadInfo();
