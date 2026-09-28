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
- `chat.py` -- an interactive REPL: type a question, see rag-guard's full
  trace (what was kept/dropped, the sufficiency probability, the action,
  grounding coverage) and the final answer.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in OPENAI_API_KEY or ANTHROPIC_API_KEY, and optionally TYPESAFE_API_KEY
export $(grep -v '^#' .env | xargs)

python chat.py
```

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

## Why this repo exists

`jev-rag-guard`'s own test suite and `examples/real_pipeline.py` prove the
library works in isolation. They don't prove it's pleasant or even
correct to depend on from a project that didn't write it -- import paths,
the `requirements.txt` install story, and whether the documented API
actually matches what you get after a real `pip install` are all things
that only show up once something *outside* that repo depends on it. This
repo is that outside consumer.
