# jev-rag-pipeline

A small, real RAG system used to test
[rag-guard](https://github.com/furkandrms/jev-rag-guard) the way an actual
external user would: installed via `pip`, not as an editable path into its
own repo, wired to a real vector store and a real LLM.

This repo has no logic of its own beyond retrieval, chunking, and a web
app shell -- everything about relevance filtering, the sufficiency gate,
and grounding checks comes from rag-guard.

## What's here

Two separate entry points, sharing the same pipeline wiring:

- **`app.py` + `static/` -- the web app.** Upload your own document
  (`.txt` / `.md` / `.pdf`) and chat against it. There is no built-in demo
  corpus here on purpose -- the whole point is proving rag-guard + Jev
  work on a document neither has ever seen, chunked and embedded at
  request time, not a pre-built example. Every assistant reply has a
  **"Why this answer?"** toggle that expands into a visual reasoning
  trace: which retrieved chunks were kept vs. dropped and at what
  probability, the sufficiency gate's probability against its threshold,
  and a per-claim grounding breakdown (supported/unsupported, with
  probability). The point is to make rag-guard's decisions inspectable,
  not just its final answer.
- **`chat.py` -- a terminal REPL**, separate from the web app, used to
  smoke-test rag-guard's pipeline wiring against a fixed corpus
  (`corpus.py`: ~35 documents simulating a fictional company's internal
  engineering docs, deliberately messier than trivia facts so relevance
  filtering and the sufficiency gate have real work to do). This is a
  developer tool, not part of the product -- if you just want to try the
  app, use the web UI.
- **`pipeline_setup.py`** -- shared setup (Chroma client, chunking, file
  text extraction, backend selection) both entry points build on, so they
  can't drift apart on how retrieval or generation works.
- **`pipeline_logging.py`** -- structured per-stage console logging for
  `app.py` (see "Server-side logging" below) -- what actually went in and
  out of rag-guard and the LLM, independent of what any one browser
  session's UI shows.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY or ANTHROPIC_API_KEY, and optionally TYPESAFE_API_KEY

uvicorn app:app --reload   # web app at http://127.0.0.1:8000
# or
python chat.py             # terminal REPL against the fixed demo corpus
```

`.env` is loaded automatically (via `python-dotenv`, in `pipeline_setup.py`) --
no manual `export` step needed. If you exported these as real shell/CI
environment variables instead, that still works too; `.env` never
overrides a variable that's already set.

## Web app

`app.py` serves a single-page app (no build step -- plain HTML/CSS/JS in
`static/`) with two views: an upload screen, and a chat screen that
appears once a document is uploaded for your session. Reloading the page
goes straight back to the chat screen if your session still has a
document (`GET /api/document-status`); there's an "Upload a different
document" link in the chat view to replace it, and a "View chunks" toggle
that shows exactly how your document was split before anything gets
retrieved from it.

### Live pipeline, not a spinner

Each reply streams in stage by stage over Server-Sent Events, as the real
pipeline actually runs -- a small stepper lights up **Retrieve → Relevance
→ Sufficiency → Generate → Ground** live, each with its own real elapsed
time. If sufficiency fails, Generate/Ground visibly gray out as skipped
instead of running pointlessly. This isn't a simulated animation: `app.py`
calls rag-guard's own stage functions (`filter_relevant_chunks`,
`check_sufficiency`, `check_grounding` -- the same ones `RagGuard.run()`
calls internally) directly instead of through that one wrapper, so each
stage's completion can be observed and pushed to the client the moment it
actually finishes.

A per-session metrics strip tracks real numbers across the conversation:
queries asked, total decision-model calls made, tokens used (from the
actual OpenAI/Anthropic response's `usage` field), average latency, and
cache hits -- nothing here is fabricated or simulated.

### Response cache

Asking the exact same question again (for the same document) skips the
pipeline entirely and returns the previous result in under a millisecond,
flagged with a **⚡ served from cache** badge instead of the stepper. The
cache is keyed on `(session_id, normalized query)`, in memory, and is
cleared for a session the moment it uploads a new document -- a cached
answer is only ever valid for the document it was computed against.

### API

- `GET /api/info` -- active backends, the real model name in use
  (`gpt-4o-mini` / `claude-haiku-4-5` / Jev), and the `RagGuard` thresholds
  currently in effect.
- `GET /api/document-status` -- whether this session already has an
  uploaded document, its filename/chunk count, and the full chunk list.
- `POST /api/upload` -- multipart file upload (`.txt`/`.md`/`.pdf`, 5 MB
  max). The file is chunked (`pipeline_setup.chunk_text`: paragraph-packed
  up to 800 chars, hard-split with overlap only for a single paragraph
  longer than that) and embedded into a Chroma collection scoped to the
  caller's `X-Session-Id` header, and the chunk list is returned so the UI
  can show it. Uploading again replaces that session's previous document
  rather than accumulating -- one document per session, always the current
  one -- and clears that session's response cache.
- `POST /api/chat` -- plain JSON version: `{"query": "..."}` in, the full
  `GuardReport` trace plus real metrics out. Blocks until done; used by
  curl/tests. Fails with a clear 400 if no document has been uploaded yet.
- `POST /api/chat/stream` -- same pipeline, as Server-Sent Events: one
  event per stage (`retrieval`, `relevance`, `sufficiency`, `generation`,
  `grounding`, each with `elapsed_ms`) as it actually completes, then a
  final `done` event with the full report -- or a single `cached` event
  on a cache hit. This is what the web UI uses.

### Session isolation

There's no login -- the frontend mints a random `X-Session-Id`
(`crypto.randomUUID()`) on first load and persists it in `localStorage`.
That id is the only thing scoping one visitor's uploaded document to them:
a request with a different (or missing) session id can never retrieve
another session's chunks, since each session gets its own Chroma
collection (`session_<id>`), created fresh on upload and looked up by
exact id on every query. Verified directly: uploading under one session
id and then querying with a different one returns a clean "no document
uploaded yet" error, not someone else's data.

This holds regardless of how many server instances are running, *as a
privacy guarantee* -- a session's data lives on whichever instance
handled its upload, so a different instance simply has no collection for
that id (same "no document uploaded yet" response) rather than ever
returning another session's content. The tradeoff is availability, not
correctness: running with more than one instance (or one that restarts)
means an uploaded document can become temporarily unreachable if a later
request lands elsewhere. For a single-instance deployment (e.g. one
Cloud Run instance, `min-instances=1 max-instances=1`) this doesn't come
up at all.

The frontend never hides a decision behind a plain "here's your answer" --
every reply has a **Why this answer?** toggle. Opening it shows three
stages, each with a pass/fail indicator dot: **Relevance** (a bar per
retrieved chunk, green for kept and gray for dropped, with its probability
and a text snippet), **Sufficiency** (a probability bar against the
configured threshold), and **Grounding** (each claim in the answer, tagged
supported/unsupported with its own probability). The goal is that a
non-technical reader can look at a flagged or rejected answer and see
*which* stage and *which specific evidence* drove that outcome, not just
trust a black box.

Every chunk id shown in the trace is HTML-escaped before being inserted
into the page (`static/app.js`'s `escapeHtml()`) -- chunk ids are derived
from the uploaded filename, which is attacker-controllable, so this isn't
optional hardening.

### Two different points where a bad answer gets caught

rag-guard can stop a hallucinated answer in two different places, and
they're not interchangeable:

- **Sufficiency** (before generation) -- the decision model looks at the
  retrieved chunks and decides whether to let the LLM answer *at all*. A
  "no" here means the LLM never runs; you get `insufficient_context` and
  never see what it would have said.
- **Grounding** (after generation) -- the LLM already answered, and the
  decision model checks each claim in that answer against the retrieved
  chunks. An unsupported claim means `ungrounded_answer_flagged` -- you
  still see the answer, just visibly marked as not fully backed by the
  document.

`TOP_K` (`pipeline_setup.py`, 8 chunks per query) and `sufficiency_threshold`
(0.45, lower than rag-guard's 0.6 default) are both tuned toward the second
behavior on purpose: let generation run more often, and lean on grounding
-- not sufficiency -- as the primary catch. A stricter sufficiency gate is
more defensible in an unattended production pipeline (it never pays for a
generation call it doesn't need), but this app exists partly to *show*
rag-guard's checks working, and a hard sufficiency block before generation
never runs means you never see grounding do anything. 4 was tuned against
the short, narrow Nimbus demo corpus and turned out too thin a slice for a
longer, broader uploaded document -- a summary-style question ("what is
this document about?") against a 30+ chunk document needs more than 4
chunks of coverage to be fairly judged as sufficient or not.

The LLM/decision-model backend selection happens once at process startup
(`pipeline_setup.setup()`), not per-request; per-session document
metadata (filename, chunk count) is an in-memory dict in `app.py` that
mirrors the in-memory Chroma client it describes -- both reset on process
restart, which is fine for this scope (see "Known limitations" below).

### Server-side logging

The browser's activity log only tells *you* what happened in your own
session. `pipeline_logging.py` prints the same kind of per-stage detail
to the server console, for every request, independent of the UI --
useful when `uvicorn app:app --reload` is the thing you're actually
watching:

```
17:52:05 INFO    [a3f9e21c] upload      'paper.pdf' -> 24 chunks indexed
17:52:05 INFO    [a3f9e21c] query       "What is this paper about?"
17:52:05 INFO    [a3f9e21c] retrieval   8 chunks fetched (102.9ms)
17:52:07 INFO    [a3f9e21c] relevance   kept 7/8 chunks (threshold=0.5) (2508.7ms)
17:52:08 INFO    [a3f9e21c] sufficiency p=0.83 (threshold=0.45) -> sufficient (300.1ms)
17:52:10 INFO    [a3f9e21c] generation  gpt-4o-mini | in=634 out=83 tokens (2206.8ms) -> 'This paper is about...'
17:52:10 INFO    [a3f9e21c] grounding   coverage=1.00 (2/2 claims supported, min_coverage=0.8) -> grounded (607.5ms)
17:52:10 INFO    [a3f9e21c] done        action=answered, decision_calls=11, tokens=634in/83out, total=5730.8ms
```

An insufficient or ungrounded result logs at `WARNING` instead of `INFO`
(so `grep WARNING` on the server log finds every case the pipeline
blocked or flagged something, without reading every line), and includes
the reason inline rather than just the outcome. Set `LOG_LEVEL=DEBUG` for
a line per chunk/claim -- the actual snippet and probability behind every
keep/drop and supported/unsupported decision, not just the aggregate:

```bash
LOG_LEVEL=DEBUG uvicorn app:app --reload
```
```
17:52:27 DEBUG   [a3f9e21c] relevance     [drop] p=0.45  paper.pdf-22: "Grace Ifechukwu holds a..."
17:52:27 DEBUG   [a3f9e21c] relevance     [KEEP] p=0.97  paper.pdf-4: "This work is organized around..."
```

## Example session

```
> How do I return an electric bike?
  retrieved: ['aurora_policy.txt-0']
  kept (relevant): ['aurora_policy.txt-0']
  sufficiency: True (p=0.94)
  action: answered
  grounding coverage: 1.00

You have 14 days to return an electric bike.
```

## Known limitations / before deploying publicly

- **No rate limiting or auth.** Every `/api/*` call makes a real,
  billed API call (OpenAI/Anthropic and/or Jev). Fine for local use;
  add rate limiting (per-IP, or a simple allowlist) before exposing this
  publicly, or API costs become an open-ended liability.
- **No session TTL.** Session collections, metadata, and the response
  cache all live in memory for the life of the process; a long-running
  public instance should add expiry/cleanup for stale sessions.
- **PDF parsing has no resource limit.** `pypdf` on a maliciously crafted
  PDF could consume excessive CPU/memory; low risk for personal/internal
  use, worth hardening before a public upload endpoint.
- **Uploaded content goes straight into the LLM prompt** as retrieval
  context -- a document (or question) crafted to manipulate the model is
  a known, generally-unsolved class of risk for any RAG system, not
  specific to this one.

## Why this repo exists

`jev-rag-guard`'s own test suite and `examples/real_pipeline.py` prove the
library works in isolation. They don't prove it's pleasant or even
correct to depend on from a project that didn't write it -- import paths,
the `requirements.txt` install story, and whether the documented API
actually matches what you get after a real `pip install` are all things
that only show up once something *outside* that repo depends on it. This
repo is that outside consumer.
