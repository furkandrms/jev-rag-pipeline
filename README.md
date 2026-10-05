# jev-rag-pipeline

A small, real RAG system used to test
[rag-guard](https://github.com/furkandrms/jev-rag-guard) the way an actual
external user would: installed via `pip`, not as an editable path into its
own repo, wired to a real vector store and a real LLM.

This repo has no logic of its own beyond retrieval, chunking, and a web
app shell -- everything about relevance filtering, the sufficiency gate,
and grounding checks comes from rag-guard.

## What's here

Two separate entry points, sharing the same pipeline wiring, plus the
shared setup/logging modules and dev-only evaluation tooling:

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
  text extraction, backend selection, embedding provider selection) both
  entry points build on, so they can't drift apart on how retrieval or
  generation works.
- **`pipeline_logging.py`** -- structured per-stage console logging for
  `app.py` (see "Server-side logging" below) -- what actually went in and
  out of rag-guard and the LLM, independent of what any one browser
  session's UI shows.
- **`eval_rag_guard.py` / `eval_multi_embedding.py`** -- golden-set
  evaluation scripts, not part of the running app. `eval_rag_guard.py`
  runs a hand-built set of query/expected-decision cases (transcribed from
  real test documents in `test_pdf/`) against the real pipeline and
  reports recall@k and pass rate per case group. `eval_multi_embedding.py`
  reruns the same golden set through OpenAI's and Gemini's embeddings in
  parallel to check whether rag-guard reaches the same decision regardless
  of which provider's retrieval fed it -- see "Embedding provider
  comparison" below.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY or ANTHROPIC_API_KEY, and optionally TYPESAFE_API_KEY
# GOOGLE_API_KEY additionally enables Gemini as an embedding provider and
# is required for Compare mode (see "Embedding provider comparison" below)

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
pipeline actually runs -- a small stepper lights up **Intent → Retrieve →
Relevance → Clarify → Sufficiency → Generate → Ground** live, each with its
own real elapsed time. A query stopped by the intent gate (escalate/refuse)
skips retrieval entirely; a query stopped by the clarify gate skips
straight to asking a clarifying question instead of generating. If
sufficiency fails, Generate/Ground visibly gray out as skipped instead of
running pointlessly. This isn't a simulated animation: `app.py` calls
rag-guard's own stage functions (`check_intent`, `filter_relevant_chunks`,
`check_ambiguity`, `check_sufficiency`, `check_grounding`, `check_caveat`
-- the same ones `RagGuard.run()` calls internally) directly instead of
through that one wrapper, so each stage's completion can be observed and
pushed to the client the moment it actually finishes.

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
  event per stage (`intent`, `retrieval`, `relevance`, `clarify`,
  `sufficiency`, `generation`, `grounding`, `caveat`, each with
  `elapsed_ms`) as it actually completes, then a final `done` event with
  the full report -- or a single `cached` event on a cache hit. The
  `intent` and `clarify` events can themselves end the turn early (with
  `action` = `"escalate"`/`"refuse"`/`"clarify"` in the final report)
  without the later stages ever running. This is what the web UI uses.
- `POST /api/compare` -- `{"query": "..."}` in, both providers' raw +
  JEV-protected results out (see "Embedding provider comparison" below).
  Blocks until both providers finish running in parallel.

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
every reply has a **Why this answer?** toggle. Opening it shows every gate
rag-guard ran, each with a pass/fail indicator dot: **Intent** (the
proceed/escalate/refuse classification and its probability), **Relevance**
(a bar per retrieved chunk, green for kept and gray for dropped, with its
probability and a text snippet), **Clarify** (whether the question needed
more information to answer safely), **Sufficiency** (a probability bar
against the configured threshold), **Grounding** (each claim in the
answer, tagged supported/unsupported with its own probability), and
**Caveat** (whether the answer depends on a condition or exception worth
surfacing). A gate that didn't run for a given turn (e.g. Grounding when
the intent gate already stopped the pipeline) shows no dot rather than a
false pass/fail. The goal is that a non-technical reader can look at a
flagged or rejected answer and see *which* stage and *which specific
evidence* drove that outcome, not just trust a black box.

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

`TOP_K` (`pipeline_setup.py`, 12 chunks per query by default) and
`sufficiency_threshold` (0.45, lower than rag-guard's 0.6 default) are both
tuned toward the second behavior on purpose: let generation run more
often, and lean on grounding -- not sufficiency -- as the primary catch. A
stricter sufficiency gate is more defensible in an unattended production
pipeline (it never pays for a generation call it doesn't need), but this
app exists partly to *show* rag-guard's checks working, and a hard
sufficiency block before generation never runs means you never see
grounding do anything.

`TOP_K` itself scales with document size (`_effective_top_k`):
`max(TOP_K, round(chunk_count * 0.08))`, capped at 30. A fixed `TOP_K=12`
means a 250-chunk document only ever offers ~5% of itself as retrieval
candidates per query -- not enough for broad questions ("what datasets does
this survey use?") that are legitimately answered by content spread across
many chunks. The cap exists so relevance-check cost (one decision-model
call per candidate chunk) doesn't grow unbounded on very large documents.

Retrieval also widens past the raw top-k result in two targeted ways:
up to the first 2 chunks of the document are force-kept as candidates even
if they don't naturally score high enough (`ANCHOR_CHUNK_COUNT` in
`app.py`) -- real documents often open with dense author/affiliation text
that dilutes the embedding of the sentence actually stating the paper's
contribution, which otherwise never surfaces in the top-k at all. And any
kept chunk scoring above 0.7 pulls in the next chunk immediately after it
(`NEIGHBOR_EXPANSION_THRESHOLD`) -- chunking is blind to section
boundaries, so a high-scoring chunk cut off mid-heading
("...4 OPEN PROBLEMS") otherwise leaves its actual content in the very
next chunk, which was never a retrieval candidate at all.

The LLM/decision-model backend selection happens once at process startup
(`pipeline_setup.setup()`), not per-request; per-session document
metadata (filename, chunk count) is an in-memory dict in `app.py` that
mirrors the in-memory Chroma client it describes -- both reset on process
restart, which is fine for this scope (see "Known limitations" below).

### Embedding provider comparison ("Compare" mode)

The chat view has a **Single provider / Compare (2)** mode toggle. Compare
mode runs the exact same query through two complete, independent
pipelines that differ in exactly one variable -- which embedding provider
produced the retrieval candidates (OpenAI `text-embedding-3-small` vs.
Google `gemini-embedding-001`) -- while generation model, rag-guard
thresholds, and chunk boundaries are held constant. For each provider it
shows both the raw answer (JEV off, context handed straight to the LLM)
and the JEV-protected answer with its full reasoning trace, side by side,
plus a summary table of each provider's final decision, grounding
coverage, and sufficiency probability. The point is to make retrieval
quality differences visible, not just assert them -- e.g. a broad
"what's this paper's main contribution?" query can come back fully
grounded under one provider's embedding and flagged `ungrounded_answer_flagged`
under the other's, on the same document, same chunks, same thresholds.

`POST /api/compare` (`{"query": "..."}`) is the backing endpoint. The two
providers' comparison collections are built lazily, the first time a
session opens compare mode (not on every upload -- most sessions never
use it, and building both costs two full embedding API passes over the
document for nothing if they're never needed), and in parallel rather
than one after another so a large document's first comparison doesn't pay
for both providers' embedding time twice over. Retrieval and the two full
`RagGuard.run()` passes also run in parallel (`ThreadPoolExecutor`), one
thread per provider.

Two real provider-specific constraints showed up building this and are
handled in `pipeline_setup.py`: Gemini's `embedContent` batch endpoint
rejects more than 100 inputs per call (`_add_chunks_in_batches` chunks any
`collection.add()` into batches of 90), and its free tier separately
rate-limits embedding requests per minute (generic retry with backoff,
since chromadb wraps every provider's error as a plain `ValueError` with
no typed "retryable" distinction). A per-session lock around building the
comparison collections also prevents two `/api/compare` calls racing on
the same session from each deciding the other's in-progress collection
"doesn't exist yet," deleting it, and crashing the other's in-flight
`collection.add()` call.

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
