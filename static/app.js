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
const docChipEl = document.getElementById("doc-chip");
const docChangeBtnEl = document.getElementById("doc-change-btn");
const chunksPanelEl = document.getElementById("chunks-panel");
const metricsStripEl = document.getElementById("metrics-strip");
const activityLogEl = document.getElementById("activity-log");

let thresholds = {
  relevance_threshold: 0.5,
  sufficiency_threshold: 0.6,
  grounding_threshold: 0.5,
  grounding_min_coverage: 0.8,
};

const STAGES = [
  { key: "retrieval", label: "Retrieve" },
  { key: "relevance", label: "Relevance" },
  { key: "sufficiency", label: "Sufficiency" },
  { key: "generation", label: "Generate" },
  { key: "grounding", label: "Ground" },
];

const ACTION_LABELS = {
  answered: "Answered",
  insufficient_context: "Insufficient context",
  ungrounded_answer_flagged: "Ungrounded — flagged",
};

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
  docChipEl.textContent = `📄 ${filename} · ${chunkCount} chunks`;
  renderChunksPanel(chunks || []);
  renderMetricsStrip();
  input.focus();
}

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
    chatEl.innerHTML = `
      <div class="intro">
        <p>Ask something about the document you uploaded.</p>
        <p class="intro-hint">Watch the pipeline run live -- retrieval, relevance, sufficiency, generation, and grounding, each with real timing and token counts.</p>
      </div>
    `;
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

docChangeBtnEl.addEventListener("click", () => {
  fileInputEl.value = "";
  showUploadView();
});

/* --- Chunk panel -------------------------------------------------------------- */

function renderChunksPanel(chunks) {
  document.getElementById("chunks-title").textContent = `Document chunks (${chunks.length})`;
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
    <div class="metric-item"><span class="metric-value">${sessionMetrics.queries}</span><span class="metric-label">Queries</span></div>
    <div class="metric-item"><span class="metric-value">${sessionMetrics.decisionCalls}</span><span class="metric-label">Decision calls</span></div>
    <div class="metric-item"><span class="metric-value">${sessionMetrics.inputTokens + sessionMetrics.outputTokens}</span><span class="metric-label">Tokens used</span></div>
    <div class="metric-item"><span class="metric-value">${avgLatency}ms</span><span class="metric-label">Avg latency</span></div>
    <div class="metric-item"><span class="metric-value">${sessionMetrics.cacheHits}</span><span class="metric-label">Cache hits</span></div>
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
  const stepperEl = div.querySelector(".stepper");
  stepperEl.innerHTML = STAGES.map(
    (s) => `<span class="stepper-item" data-stage="${s.key}"><span class="stepper-dot"></span>${s.label}</span>`
  ).join("");
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

function buildRelevanceStage(relevance) {
  if (!relevance.length) {
    return `<div class="stage-empty">No chunks retrieved.</div>`;
  }
  return relevance
    .map((r) => {
      const kept = r.kept;
      const snippet = r.text.length > 90 ? r.text.slice(0, 90) + "…" : r.text;
      // escapeHtml() on chunk_id too: it's derived from the uploaded
      // filename, which is attacker-controllable -- never trust it raw
      // in HTML, even though it's just an id string.
      const safeId = escapeHtml(r.chunk_id);
      return `
        <div>
          <div class="chunk-row">
            <span class="chunk-id" title="${safeId}">${safeId}</span>
            <div class="chunk-bar-track">
              <div class="chunk-bar-fill ${kept ? "kept" : "dropped"}" style="width:${pct(r.probability)}"></div>
            </div>
            <span class="chunk-prob">${r.probability.toFixed(2)}</span>
          </div>
          <div class="chunk-snippet">${kept ? "✓ kept — " : "dropped — "}${escapeHtml(snippet)}</div>
        </div>`;
    })
    .join("");
}

function buildSufficiencyStage(sufficiency) {
  const pass = sufficiency.sufficient;
  return `
    <div class="suff-row">
      <div class="suff-bar-track">
        <div class="suff-threshold" style="left:${pct(thresholds.sufficiency_threshold)}"></div>
        <div class="suff-bar-fill ${pass ? "pass" : "fail"}" style="width:${pct(sufficiency.probability)}"></div>
      </div>
      <span class="suff-label">${pass ? "sufficient" : "insufficient"} · ${sufficiency.probability.toFixed(2)}</span>
    </div>
    ${sufficiency.reason ? `<div class="chunk-snippet" style="margin-left:0;margin-top:6px;">${escapeHtml(sufficiency.reason)}</div>` : ""}
  `;
}

function buildGenerationStage(usage) {
  if (!usage || (!usage.input_tokens && !usage.output_tokens)) {
    return `<div class="stage-empty">No usage data reported by this backend.</div>`;
  }
  return `<div class="chunk-snippet" style="margin-left:0;">input: ${usage.input_tokens} tokens · output: ${usage.output_tokens} tokens</div>`;
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
    <div class="chunk-snippet" style="margin-left:0;margin-bottom:8px;">coverage: ${pct(grounding.coverage)} (needs ≥ ${pct(thresholds.grounding_min_coverage)})</div>
    ${rows}
  `;
}

function renderTracePanel(container, report) {
  const tracePanel = container.querySelector(".trace-panel");
  const relevancePass = report.relevance.some((r) => r.kept);
  const suffPass = report.sufficiency.sufficient;
  const groundPass = report.grounding ? report.grounding.grounded : null;

  tracePanel.innerHTML = `
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${relevancePass ? "pass" : "fail"}"></span>Relevance</div>
      ${buildRelevanceStage(report.relevance)}
    </div>
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${suffPass ? "pass" : "fail"}"></span>Sufficiency</div>
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
  `;

  const toggle = container.querySelector(".trace-toggle");
  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    tracePanel.classList.toggle("open");
  });
}

function finalizeMessage(container, report) {
  const badge = container.querySelector(".action-badge");
  const textEl = container.querySelector(".msg-text");
  badge.textContent = ACTION_LABELS[report.action] || report.action;
  badge.classList.add("show", report.action);
  textEl.textContent = report.answer;
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
  addUserMessage(query);
  const container = addAssistantMessage();
  setStageState(container, "retrieval", "active");

  input.value = "";
  sendBtn.disabled = true;

  let bubbleCleared = false;
  const clearBubble = () => {
    if (!bubbleCleared) {
      container.querySelector(".msg-bubble").innerHTML = `<div class="action-badge"></div><div class="msg-text"></div>`;
      bubbleCleared = true;
    }
  };

  logLine("info", `query: <b>${escapeHtml(query)}</b>`);

  try {
    await streamChat(query, (event, data) => {
      if (event === "error") {
        container.querySelector(".msg-bubble").textContent = data.detail;
        logLine("fail", escapeHtml(data.detail));
        return;
      }
      if (event === "cached") {
        container.querySelector(".stepper").innerHTML = `<span class="cache-badge">⚡ served from cache</span>`;
        clearBubble();
        finalizeMessage(container, data);
        recordMetrics(data, data.total_elapsed_ms, true);
        logLine("cache", `cache hit -- served in ${data.total_elapsed_ms}ms, pipeline skipped`);
        return;
      }
      if (STAGES.some((s) => s.key === event)) {
        setStageState(container, event, "done", data.elapsed_ms);
        logLine("ok", `<b>${event}</b> ${stageLogDetail(event, data)} (${data.elapsed_ms}ms)`);

        // Mark the next stage "active" (pulsing) so the stepper visibly
        // progresses even though SSE only tells us about completions.
        const idx = STAGES.findIndex((s) => s.key === event);
        const next = STAGES[idx + 1];
        if (next) setStageState(container, next.key, "active");

        if (event === "sufficiency" && !data.sufficiency.sufficient) {
          setStageState(container, "generation", "skipped");
          setStageState(container, "grounding", "skipped");
          logLine("warn", "generation + grounding skipped -- context judged insufficient");
        }
        return;
      }
      if (event === "done") {
        clearBubble();
        finalizeMessage(container, data.report);
        recordMetrics(data.report, data.total_elapsed_ms, false);
        logLine(
          data.report.action === "answered" ? "ok" : "warn",
          `done -- <b>${escapeHtml(data.report.action)}</b> (${data.total_elapsed_ms}ms total)`
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
    case "retrieval":
      return `fetched ${data.chunk_count} chunks`;
    case "relevance": {
      const kept = data.relevance.filter((r) => r.kept).length;
      return `kept ${kept}/${data.relevance.length} chunks`;
    }
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

checkExistingDocument();
loadInfo();
