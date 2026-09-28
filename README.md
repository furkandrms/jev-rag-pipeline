# jev-rag-pipeline

A small, real RAG system used to test
[rag-guard](https://github.com/furkandrms/jev-rag-guard) the way an actual
external user would: installed via `pip`, not as an editable path into its
own repo, wired to a real vector store, a real LLM, and a corpus rag-guard
has never seen during its own development.

This repo has no logic of its own beyond retrieval + a chat loop --
everything about relevance filtering, the sufficiency gate, and grounding
checks comes from rag-guard.

## What's here

- `corpus.py` -- ~35 documents simulating internal engineering docs for a
  fictional company, Nimbus (onboarding, API references, runbooks, incident
  postmortems, policies). Deliberately messier than trivia facts: several
  runbooks share near-identical wording for different services, and a few
  plausible questions genuinely aren't answered anywhere in the corpus --
  both are meant to stress relevance filtering and the sufficiency gate,
  not just confirm the easy case.
- `pipeline_setup.py` -- shared setup (vector store, retriever, backend
  selection) used by both entry points below, so they can't drift apart.
- `chat.py` -- an interactive terminal REPL: type a question, see rag-guard's
  full trace (what was kept/dropped, the sufficiency probability, the
  action, grounding coverage) and the final answer, as plain text.
- `app.py` + `static/` -- a web chat UI serving the same pipeline over
  FastAPI, with two modes: chat against the built-in Nimbus corpus, or
  **upload your own document** (.txt/.md/.pdf) and chat against that
  instead -- this is the actual point of the whole repo: proving rag-guard
  + Jev work on a document neither has ever seen, chunked and embedded at
  request time, not just a pre-built demo corpus. Every assistant reply has
  a **"Why this answer?"** toggle that expands into a visual reasoning
  trace: which retrieved chunks were kept vs. dropped and at what
  probability, the sufficiency gate's probability against its threshold,
  and a per-claim grounding breakdown (supported/unsupported, with
  probability). The point is to make rag-guard's decisions inspectable,
  not just its final answer.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY or ANTHROPIC_API_KEY, and optionally TYPESAFE_API_KEY

python chat.py        # terminal REPL
# or
uvicorn app:app --reload   # web UI at http://127.0.0.1:8000
```

`.env` is loaded automatically (via `python-dotenv`, in `pipeline_setup.py`) --
no manual `export` step needed. If you exported these as real shell/CI
environment variables instead, that still works too; `.env` never
overrides a variable that's already set.

## Example session

```
> How do I restart the ingest service?
  retrieved: ['runbook-ingest-2', 'runbook-query-1', 'runbook-ingest-1', 'api-ingest-1']
  kept (relevant): ['runbook-ingest-2']  dropped: ['runbook-query-1', 'runbook-ingest-1', 'api-ingest-1']
  sufficiency: True (p=0.91)
  action: answered
  grounding coverage: 1.00

Use `kubectl rollout restart deployment/ingest -n prod`. This is a rolling
restart, so no downtime is expected.

> what's nimbus's data retention policy for customer PII?
  retrieved: ['policy-security-1', 'faq-4', 'policy-security-2', 'onboard-1']
  kept (relevant): []
  sufficiency: False (p=0.00)
  action: insufficient_context

I don't have enough information in the retrieved context to answer this
confidently.
```

(The second example is intentional -- the corpus has no PII-specific
retention policy, only a general Kafka topic retention note. That's the
sufficiency gate doing its job instead of the LLM guessing.)

## Web UI

`app.py` serves a single-page chat UI (no build step -- plain HTML/CSS/JS
in `static/`) plus three JSON endpoints:

- `GET /api/info` -- which backends are active (`generate_fn`,
  `decision_model`) and the `RagGuard` thresholds currently in effect.
- `POST /api/upload` -- multipart file upload (`.txt`/`.md`/`.pdf`, 5 MB
  max). The file is chunked (`pipeline_setup.chunk_text`: paragraph-packed
  up to 800 chars, hard-split with overlap only for a single paragraph
  longer than that) and embedded into a Chroma collection scoped to the
  caller's `X-Session-Id` header. Uploading again replaces that session's
  previous document rather than accumulating -- one document per session,
  always the current one.
- `POST /api/chat` -- `{"query": "...", "mode": "demo" | "upload"}` in
  (plus `X-Session-Id` when `mode` is `"upload"`), the full `GuardReport`
  trace out (per-chunk relevance, sufficiency, per-claim grounding,
  answer, action).

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
request lands elsewhere. For a demo/single-instance deployment (e.g. one
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

The vector store and backend selection happen once at process startup
(`pipeline_setup.setup()`), not per-request.

## Why this repo exists

`jev-rag-guard`'s own test suite and `examples/real_pipeline.py` prove the
library works in isolation. They don't prove it's pleasant or even
correct to depend on from a project that didn't write it -- import paths,
the `requirements.txt` install story, and whether the documented API
actually matches what you get after a real `pip install` are all things
that only show up once something *outside* that repo depends on it. This
repo is that outside consumer.
