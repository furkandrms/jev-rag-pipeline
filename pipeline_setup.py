"""Shared setup for both chat.py (REPL) and app.py (web UI): vector store,
retriever, and backend selection (generate_fn + DecisionModel), so the two
entry points never drift apart on how the pipeline gets wired.
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


def build_vector_store():
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        ids, texts = zip(*DOCUMENTS)
        collection.add(ids=list(ids), documents=list(texts))
    return collection


def make_retriever(collection):
    def retrieve(query: str, k: int = TOP_K) -> list[Chunk]:
        results = collection.query(query_texts=[query], n_results=k)
        ids = results["ids"][0]
        texts = results["documents"][0]
        distances = results["distances"][0]
        return [
            # float(): chromadb returns numpy.float32 distances, which
            # json.dumps() can't serialize -- the web UI needs plain floats.
            Chunk(id=chunk_id, text=text, metadata={"distance": float(distance)})
            for chunk_id, text, distance in zip(ids, texts, distances)
        ]

    return retrieve


def make_generate_fn() -> tuple[str, GenerateFn, DecisionModel]:
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
        "Needs a real LLM to generate answers: set OPENAI_API_KEY or "
        "ANTHROPIC_API_KEY (see .env.example).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def make_decision_model(fallback: DecisionModel) -> tuple[str, DecisionModel]:
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev", JevDecisionModel()
    return "same as generate_fn", fallback


def setup() -> tuple[RagGuard, "callable", dict[str, str]]:
    """Build everything chat.py / app.py need: (guard, retrieve, backend_info)."""
    generate_backend, generate_fn, fallback_model = make_generate_fn()
    decision_backend, model = make_decision_model(fallback_model)
    collection = build_vector_store()
    retrieve = make_retriever(collection)
    guard = RagGuard(model=model)
    backend_info = {"generate_fn": generate_backend, "decision_model": decision_backend}
    return guard, retrieve, generate_fn, backend_info
