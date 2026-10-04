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

let thresholds = {
  relevance_threshold: 0.5,
  sufficiency_threshold: 0.6,
  grounding_threshold: 0.5,
  grounding_min_coverage: 0.8,
};

const ACTION_LABELS = {
  answered: "Answered",
  insufficient_context: "Insufficient context",
  ungrounded_answer_flagged: "Ungrounded — flagged",
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

async function loadInfo() {
  try {
    const res = await fetch("/api/info");
    const data = await res.json();
    if (data.thresholds) thresholds = data.thresholds;
    badgesEl.innerHTML = `
      <span class="backend-badge">generate_fn: <b>${escapeHtml(data.generate_fn)}</b></span>
      <span class="backend-badge">decisions: <b>${escapeHtml(data.decision_model)}</b></span>
    `;
  } catch (e) {
    badgesEl.innerHTML = `<span class="backend-badge">backend info unavailable</span>`;
  }
}

function showUploadView() {
  viewUploadEl.hidden = false;
  viewChatEl.hidden = true;
}

function showChatView(filename, chunkCount) {
  viewUploadEl.hidden = true;
  viewChatEl.hidden = false;
  docChipEl.textContent = `📄 ${filename} · ${chunkCount} chunks`;
  input.focus();
}

async function checkExistingDocument() {
  try {
    const res = await fetch("/api/document-status", { headers: { "X-Session-Id": sessionId } });
    const data = await res.json();
    if (data.uploaded) {
      showChatView(data.filename, data.chunk_count);
    } else {
      showUploadView();
    }
  } catch (e) {
    showUploadView();
  }
}

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

    // Fresh document for this session -- clear any previous conversation
    // so old answers don't sit alongside a different document's context.
    chatEl.innerHTML = `
      <div class="intro">
        <p>Ask something about the document you uploaded.</p>
        <p class="intro-hint">Every answer includes a reasoning trace -- click <strong>Why this answer?</strong> to see it.</p>
      </div>
    `;
    uploadLabelEl.textContent = `Drop a .txt / .md / .pdf here, or click to choose a file`;
    uploadStatusEl.textContent = "";
    showChatView(data.filename, data.chunk_count);
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

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="msg-bubble"></div>`;
  div.querySelector(".msg-bubble").textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addPendingAssistant() {
  const div = document.createElement("div");
  div.className = "msg msg-assistant pending";
  div.innerHTML = `<div class="msg-bubble"><span class="dots">thinking</span></div>`;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function pct(p) {
  return `${Math.round(p * 100)}%`;
}

function buildRelevanceStage(relevance) {
  if (!relevance.length) {
    return `<div class="stage-empty">No chunks retrieved.</div>`;
  }
  const rows = relevance
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
  return rows;
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

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderAssistantMessage(container, report) {
  container.classList.remove("pending");
  const badge = container.querySelector(".action-badge");
  const textEl = container.querySelector(".msg-text");
  const tracePanel = container.querySelector(".trace-panel");
  const toggle = container.querySelector(".trace-toggle");

  badge.textContent = ACTION_LABELS[report.action] || report.action;
  badge.classList.add("show", report.action);
  textEl.textContent = report.answer;

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
    <div class="trace-stage">
      <div class="trace-stage-title"><span class="stage-dot ${groundPass === null ? "" : groundPass ? "pass" : "fail"}"></span>Grounding</div>
      ${buildGroundingStage(report.grounding)}
    </div>
  `;

  toggle.addEventListener("click", () => {
    toggle.classList.toggle("open");
    tracePanel.classList.toggle("open");
  });

  chatEl.scrollTop = chatEl.scrollHeight;
}

async function sendQuery(query) {
  addUserMessage(query);
  const pending = addPendingAssistant();

  input.value = "";
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-Id": sessionId },
      body: JSON.stringify({ query }),
    });
    const report = await res.json();

    if (!res.ok) {
      pending.querySelector(".msg-bubble").textContent = report.detail || "Request failed.";
      return;
    }

    pending.innerHTML = template.content.cloneNode(true).firstElementChild.innerHTML;
    renderAssistantMessage(pending, report);
  } catch (e) {
    pending.querySelector(".msg-bubble").textContent = "Something went wrong talking to the backend.";
  } finally {
    sendBtn.disabled = false;
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
