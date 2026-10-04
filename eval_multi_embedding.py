"""Phase 3 of the multi-embedding hallucination test: does rag-guard's
decision layer catch a retrieval failure the same way regardless of which
embedding provider produced it?

This reuses eval_rag_guard.py's golden set verbatim (same documents, same
cases, same expected decisions/sections) and runs it through two parallel
retrieval pipelines that differ in exactly one variable -- the embedding
provider -- while everything else (chunk boundaries, the generation model,
rag-guard's thresholds) is held constant, per the original spec:

    Anthropic, OpenAI, Gemini -> OpenAI, Gemini only.

Anthropic has no embeddings API of its own; their own docs point to Voyage
AI as the recommended third-party provider instead
(https://docs.anthropic.com/en/docs/build-with-claude/embeddings). Rather
than silently substituting a different vendor for what the original plan
called "Anthropic", this comparison is scoped to the two providers that
actually have a first-party embeddings endpoint. Re-add a third leg (Voyage
or otherwise) by writing one more `_build_*_collection` function and adding
it to PROVIDERS below -- the comparison loop itself is provider-agnostic.

Run:
    python eval_multi_embedding.py

Needs OPENAI_API_KEY and GOOGLE_API_KEY in .env (the latter is also used
for Jev if TYPESAFE_API_KEY isn't set -- see pipeline_setup.setup()), plus
the same document(s) as eval_rag_guard.py (DOCS, imported from it).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.types import EmbeddingFunction

from eval_rag_guard import DOCS, ROOT, Doc, _answered_set, _recall_hit
from pipeline_setup import (
    _add_chunks_in_batches,
    chunk_text,
    extract_text,
    get_chroma_client,
    make_retriever,
    setup,
)

@dataclass
class Provider:
    key: str  # short id, e.g. "openai" -- used in collection names and the report
    name: str  # display name
    build_embedding_function: Callable[[], EmbeddingFunction]


def _openai_embedding_function():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
    )


def _gemini_embedding_function():
    return embedding_functions.GoogleGenaiEmbeddingFunction(
        model_name="gemini-embedding-001", api_key_env_var="GOOGLE_API_KEY"
    )


PROVIDERS = [
    Provider("openai", "OpenAI (text-embedding-3-small)", _openai_embedding_function),
    Provider("gemini", "Google (gemini-embedding-001)", _gemini_embedding_function),
]


def load_collection_for_provider(doc: Doc, provider: Provider):
    """Same chunking as eval_rag_guard.load_collection() -- same chunk
    boundaries across every provider is the whole point (see module
    docstring) -- just embedded through a different provider's function
    into its own, separately-named collection.
    """
    client = get_chroma_client()
    collection_name = f"{doc.collection_name}_{provider.key}"
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        collection_name, embedding_function=provider.build_embedding_function()
    )
    text = extract_text(doc.filename, (ROOT / doc.filename).read_bytes())
    if doc.strip_section:
        start_marker, end_marker = doc.strip_section
        start = text.find(start_marker)
        end = text.find(end_marker)
        if start != -1 and end != -1 and end > start:
            text = text[:start] + text[end:]
    chunks = chunk_text(text)
    _add_chunks_in_batches(collection, [f"{doc.filename}-{i}" for i in range(len(chunks))], chunks)
    return collection, len(chunks)


def main() -> int:
    print("Setting up pipeline (generation model + Jev decision model -- held constant across providers)...")
    guard, _, _, make_upload_generate_fn, _, backend_info, _ = setup()
    generate_fn = make_upload_generate_fn()
    print(f"  backend: {backend_info}\n")

    # provider -> label -> [total, passed]; provider -> recall [hits, total]
    group_totals: dict[str, dict[str, list[int]]] = {p.key: defaultdict(lambda: [0, 0]) for p in PROVIDERS}
    recall: dict[str, list[int]] = {p.key: [0, 0] for p in PROVIDERS}
    # Per-case action agreement across providers -- the actual research
    # question: does Jev reach the same decision regardless of which
    # provider's retrieval fed it?
    agreement_total = 0
    agreement_hits = 0

    for doc in DOCS:
        print(f"=== {doc.name} ({doc.filename}) ===")
        retrievers = {}
        for provider in PROVIDERS:
            collection, n_chunks = load_collection_for_provider(doc, provider)
            retrievers[provider.key] = make_retriever(collection)
            print(f"  [{provider.key}] indexed {n_chunks} chunks")
        print()

        for case in doc.cases:
            actions_by_provider: dict[str, str] = {}
            for provider in PROVIDERS:
                chunks = retrievers[provider.key](case.query)
                hit = _recall_hit(case, chunks)
                if hit is not None:
                    recall[provider.key][1] += 1
                    recall[provider.key][0] += int(hit)

                report = guard.run(case.query, chunks, generate_fn)
                ok = case.expect == "either" or report.action in _answered_set(case.expect)
                totals = group_totals[provider.key][case.label]
                totals[1] += 1
                totals[0] += int(ok)
                actions_by_provider[provider.key] = report.action

            agreement_total += 1
            if len(set(actions_by_provider.values())) == 1:
                agreement_hits += 1
            else:
                mismatch = ", ".join(f"{k}={v}" for k, v in actions_by_provider.items())
                print(f"  [DIVERGED] ({case.label}) {mismatch}\n           Q: {case.query}")

        print()

    print("=== Recall@k by provider ===")
    for provider in PROVIDERS:
        hits, total = recall[provider.key]
        pct = 100 * hits / total if total else 0.0
        print(f"  {provider.name}: {hits}/{total} ({pct:.0f}%)")

    print("\n=== Pass rate by provider ===")
    for provider in PROVIDERS:
        all_totals = group_totals[provider.key]
        total = sum(t for _, t in all_totals.values())
        passed = sum(p for p, _ in all_totals.values())
        print(f"  {provider.name}: {passed}/{total} ({100 * passed / total:.0f}%)" if total else f"  {provider.name}: no cases")

    print(
        f"\nDecision agreement across providers: {agreement_hits}/{agreement_total} "
        f"({100 * agreement_hits / agreement_total:.0f}%) -- same final action regardless of "
        f"embedding provider. Divergences are logged above as they're found; each one is a case "
        f"where embedding quality, not rag-guard's decision logic, determined the outcome."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
