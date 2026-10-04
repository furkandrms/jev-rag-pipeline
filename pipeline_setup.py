"""Shared setup for both chat.py (REPL) and app.py (web UI): vector store,
retriever, and backend selection (generate_fn + DecisionModel), so the two
entry points never drift apart on how the pipeline gets wired.
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from corpus import DOCUMENTS

from rag_guard import Chunk, RagGuard
from rag_guard.decision_model import (
    AnthropicDecisionModel,
    DecisionModel,
    JevDecisionModel,
    OpenAIDecisionModel,
)
from rag_guard.pipeline import GenerateFn

# Loads .env into the process environment automatically, so `python chat.py`
# / `uvicorn app:app` work right after `cp .env.example .env` -- no manual
# `export $(...)` step to forget. Does nothing (safely) if .env is absent;
# never overrides a variable already set in the real environment.
load_dotenv()

COLLECTION_NAME = "nimbus_docs"
# Broad/summary-style questions ("what is this about?") need coverage across
# more of the document than a narrow factual lookup does -- 4 was tuned for
# short, focused corpora (the Nimbus demo docs) and was too thin a slice for
# longer uploaded documents (e.g. a 37-chunk paper), where 4/37 chunks often
# didn't include enough of the document for sufficiency to fairly judge it.
TOP_K = 8
UPLOAD_COLLECTION_PREFIX = "session_"

SYSTEM_PROMPT = (
    "You are a support assistant for Nimbus, a fictional data platform. "
    "Answer the question using only the context below. If the context "
    "doesn't contain the answer, say so plainly -- do not guess."
)

UPLOAD_SYSTEM_PROMPT = (
    "Answer the question using only the context below, which comes from a "
    "document the user uploaded. If the context doesn't contain the "
    "answer, say so plainly -- do not guess."
)

GENERATE_MODEL_NAMES = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"}

_chroma_client = None


def get_chroma_client():
    """One in-memory chromadb client shared by the whole process.

    Both the built-in Nimbus collection and every per-session upload
    collection live in this same client -- a fresh `chromadb.Client()` per
    call would give each its own isolated (and immediately-orphaned)
    in-memory store instead of a shared one.
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb

        _chroma_client = chromadb.Client()
    return _chroma_client


def build_vector_store():
    client = get_chroma_client()
    collection = client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() == 0:
        ids, texts = zip(*DOCUMENTS)
        collection.add(ids=list(ids), documents=list(texts))
    return collection


def session_collection_name(session_id: str) -> str:
    return f"{UPLOAD_COLLECTION_PREFIX}{session_id}"


def replace_session_document(session_id: str, filename: str, chunks: list[str]):
    """Replace whatever this session previously uploaded with a fresh set of chunks.

    One document per session, by design -- re-uploading swaps it out rather
    than accumulating, so a session's retrieval results are always "the
    current document," never a growing mix of everything ever uploaded.
    """
    client = get_chroma_client()
    name = session_collection_name(session_id)
    try:
        client.delete_collection(name)
    except Exception:
        pass  # nothing to delete yet
    collection = client.create_collection(name)
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks)
    return collection


def get_session_collection(session_id: str):
    """Returns the session's upload collection, or None if it has none yet."""
    client = get_chroma_client()
    name = session_collection_name(session_id)
    try:
        collection = client.get_collection(name)
    except Exception:
        return None
    return collection if collection.count() > 0 else None


def remove_session_document(session_id: str) -> None:
    """Delete this session's document outright (not a replace -- nothing takes its place)."""
    client = get_chroma_client()
    try:
        client.delete_collection(session_collection_name(session_id))
    except Exception:
        pass  # nothing to delete


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Paragraph-aware chunking: pack whole paragraphs up to chunk_size,
    and only hard-split a paragraph that's longer than chunk_size on its own
    (with overlap, so a sentence spanning a hard split isn't fully lost).
    Not a claim to sophistication -- a deliberately simple, inspectable
    baseline for testing rag-guard's checks against real chunked content.
    """
    paragraphs = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            flush()
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - overlap
            continue
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            flush()
            current = para
    flush()
    return chunks


def extract_text(filename: str, content: bytes) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("txt", "md"):
        return content.decode("utf-8", errors="replace")
    if ext == "pdf":
        import io

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: .{ext or '?'} (supported: .txt, .md, .pdf)")


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


def make_client() -> tuple[str, object]:
    """Pick and construct the one LLM client used for both generation and,
    unless Jev is configured, the decision model too -- shared rather than
    built twice, since it's the same account/credentials either way.
    """
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        return "openai", OpenAI()

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        return "anthropic", anthropic.Anthropic()

    print(
        "Needs a real LLM to generate answers: set OPENAI_API_KEY or "
        "ANTHROPIC_API_KEY (see .env.example).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_generate_fn(
    backend: str, client, system_prompt: str, usage_sink: dict | None = None
) -> GenerateFn:
    """Build the generation closure for `backend`.

    `usage_sink`, if given, gets the real token usage from the underlying
    API response written into it as a side effect (`{"input_tokens": ...,
    "output_tokens": ...}`) -- the only way to surface it, since `GenerateFn`
    itself must return a plain string to satisfy RagGuard's contract.
    """
    if backend == "openai":

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.chat.completions.create(
                model=GENERATE_MODEL_NAMES["openai"],
                messages=[
                    {
                        "role": "user",
                        "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            if usage_sink is not None and response.usage is not None:
                usage_sink["input_tokens"] = response.usage.prompt_tokens
                usage_sink["output_tokens"] = response.usage.completion_tokens
            return response.choices[0].message.content

        return generate

    if backend == "anthropic":

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.messages.create(
                model=GENERATE_MODEL_NAMES["anthropic"],
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}",
                    }
                ],
            )
            if usage_sink is not None and response.usage is not None:
                usage_sink["input_tokens"] = response.usage.input_tokens
                usage_sink["output_tokens"] = response.usage.output_tokens
            return "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            )

        return generate

    raise ValueError(f"unknown backend: {backend}")


def make_decision_model_for_client(backend: str, client) -> DecisionModel:
    if backend == "openai":
        return OpenAIDecisionModel(client=client)
    if backend == "anthropic":
        return AnthropicDecisionModel(client=client)
    raise ValueError(f"unknown backend: {backend}")


def make_decision_model(fallback: DecisionModel) -> tuple[str, DecisionModel]:
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev", JevDecisionModel()
    return "same as generate_fn", fallback


def setup():
    """Build everything chat.py / app.py need.

    Returns (guard, retrieve, generate_fn, make_upload_generate_fn, backend_info).
    `generate_fn` answers using the Nimbus-support system prompt against the
    demo corpus (chat.py only). `make_upload_generate_fn(usage_sink=None)` is
    a factory, not a fixed function -- app.py calls it fresh per request with
    a new `usage_sink` dict so it can read back that specific call's real
    token usage afterward, without building a whole new client each time.
    """
    client_backend, client = make_client()
    generate_fn = build_generate_fn(client_backend, client, SYSTEM_PROMPT)

    def make_upload_generate_fn(usage_sink: dict | None = None) -> GenerateFn:
        return build_generate_fn(client_backend, client, UPLOAD_SYSTEM_PROMPT, usage_sink=usage_sink)

    fallback_model = make_decision_model_for_client(client_backend, client)
    decision_backend, model = make_decision_model(fallback_model)
    collection = build_vector_store()
    retrieve = make_retriever(collection)
    # Lower sufficiency_threshold than RagGuard's 0.6 default: the intent
    # for this app is to let generation run more often and rely on the
    # grounding check (strict defaults, untouched) to catch an answer that
    # overreaches, rather than relying on sufficiency to pre-empt every
    # borderline case before generation ever gets a chance to run.
    guard = RagGuard(model=model, sufficiency_threshold=0.45)
    backend_info = {
        "generate_fn": client_backend,
        "generate_model": GENERATE_MODEL_NAMES[client_backend],
        "decision_model": decision_backend,
    }
    return guard, retrieve, generate_fn, make_upload_generate_fn, backend_info
