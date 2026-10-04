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
document" link in the chat view to replace it.

JSON API the frontend calls:

- `GET /api/info` -- which backends are active (`generate_fn`,
  `decision_model`) and the `RagGuard` thresholds currently in effect.
- `GET /api/document-status` -- whether this session already has an
  uploaded document, and its filename/chunk count if so.
- `POST /api/upload` -- multipart file upload (`.txt`/`.md`/`.pdf`, 5 MB
  max). The file is chunked (`pipeline_setup.chunk_text`: paragraph-packed
  up to 800 chars, hard-split with overlap only for a single paragraph
  longer than that) and embedded into a Chroma collection scoped to the
  caller's `X-Session-Id` header. Uploading again replaces that session's
  previous document rather than accumulating -- one document per session,
  always the current one.
- `POST /api/chat` -- `{"query": "..."}` in (plus the `X-Session-Id`
  header), the full `GuardReport` trace out (per-chunk relevance,
  sufficiency, per-claim grounding, answer, action). Fails with a clear
  400 if no document has been uploaded yet for that session.

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

The LLM/decision-model backend selection happens once at process startup
(`pipeline_setup.setup()`), not per-request; per-session document
metadata (filename, chunk count) is an in-memory dict in `app.py` that
mirrors the in-memory Chroma client it describes -- both reset on process
restart, which is fine for this scope (see "Known limitations" below).

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
- **No session TTL.** Session collections and metadata live in memory for
  the life of the process; a long-running public instance should add
  expiry/cleanup for stale sessions.
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
