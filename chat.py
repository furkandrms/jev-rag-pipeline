"""Interactive REPL for testing rag-guard as an externally-installed package.

This repo exists to answer one question: does rag-guard actually work when
pulled in the way a real user would pull it in (`pip install git+https://...`),
against a real vector store, a real LLM, and a corpus it has never seen
during rag-guard's own development? `corpus.py` is an internal-engineering-
docs simulation for a fictional company (Nimbus) -- deliberately messier
than trivia facts, with overlapping runbooks and a few real gaps, so
relevance/sufficiency/grounding all get exercised, not just the easy case.

Usage:
    python chat.py
    > How do I restart the ingest service?
    > what's nimbus's data retention policy for raw events?
    > exit
"""

from __future__ import annotations

import os
import sys

from corpus import DOCUMENTS

from rag_guard import Chunk, RagGuard
from rag_guard.decision_model import (
    AnthropicDecisionModel,
    DecisionModel,
    JevDecisionModel,
    OpenAIDecisionModel,
)
from rag_guard.pipeline import GenerateFn

COLLECTION_NAME = "nimbus_docs"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a support assistant for Nimbus, a fictional data platform. "
    "Answer the question using only the context below. If the context "
    "doesn't contain the answer, say so plainly -- do not guess."
)


def _build_vector_store():
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        ids, texts = zip(*DOCUMENTS)
        collection.add(ids=list(ids), documents=list(texts))
    return collection


def _make_retriever(collection):
    def retrieve(query: str, k: int = TOP_K) -> list[Chunk]:
        results = collection.query(query_texts=[query], n_results=k)
        ids = results["ids"][0]
        texts = results["documents"][0]
        distances = results["distances"][0]
        return [
            Chunk(id=chunk_id, text=text, metadata={"distance": distance})
            for chunk_id, text, distance in zip(ids, texts, distances)
        ]

    return retrieve


def _make_generate_fn() -> tuple[str, GenerateFn, DecisionModel]:
    """Pick the LLM that generates answers, and the decision model it also backs by default."""
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            return response.choices[0].message.content

        return "openai", generate, OpenAIDecisionModel(client=client)

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

        return "anthropic", generate, AnthropicDecisionModel(client=client)

    print(
        "chat.py needs a real LLM to generate answers: set OPENAI_API_KEY or "
        "ANTHROPIC_API_KEY (see .env.example).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def _make_decision_model(fallback: DecisionModel) -> tuple[str, DecisionModel]:
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev", JevDecisionModel()
    return "same as generate_fn", fallback


def _print_trace(query: str, report) -> None:
    kept_ids = [c.id for c in report.kept_chunks]
    dropped_ids = [r.chunk.id for r in report.relevance if not r.kept]
    print(f"  retrieved: {[r.chunk.id for r in report.relevance]}")
    print(f"  kept (relevant): {kept_ids}" + (f"  dropped: {dropped_ids}" if dropped_ids else ""))
    print(
        f"  sufficiency: {report.sufficiency.sufficient} "
        f"(p={report.sufficiency.probability:.2f})"
    )
    print(f"  action: {report.action}")
    if report.grounding is not None:
        print(f"  grounding coverage: {report.grounding.coverage:.2f}")
        for claim in report.grounding.claims:
            if not claim.supported:
                print(f"    UNSUPPORTED ({claim.probability:.2f}): {claim.claim}")


def main() -> None:
    generate_backend, generate_fn, fallback_model = _make_generate_fn()
    decision_backend, model = _make_decision_model(fallback_model)
    collection = _build_vector_store()
    retrieve = _make_retriever(collection)
    guard = RagGuard(model=model)

    print("rag-guard REPL against the Nimbus docs corpus")
    print(f"generate_fn backend: {generate_backend}  |  decision model backend: {decision_backend}")
    print("Type a question, or 'exit' to quit.\n")

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        chunks = retrieve(query)
        report = guard.run(query, chunks, generate_fn)

        _print_trace(query, report)
        print(f"\n{report.answer}\n")


if __name__ == "__main__":
    main()
