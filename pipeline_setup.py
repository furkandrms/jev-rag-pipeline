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
# 8 still missed cases where a table gets hard-split across chunks (e.g. a
# price row lands in one chunk but the column header -- the only thing that
# disambiguates "this number is the monthly fee" -- ranks just outside top-8
# in a different chunk); 12 gives disambiguating neighbor chunks like that a
# real chance to be retrieved too.
TOP_K = 12
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


def _word_boundary_end(text: str, start: int, chunk_size: int) -> int:
    """Pick a hard-split end index that falls on whitespace, not mid-word.

    Searches backward from `start + chunk_size` for the nearest whitespace,
    within the last 20% of the window so a single abnormally long "word"
    (a URL, a table with no spaces) can't shrink the chunk to near-nothing.
    Falls back to the raw offset if no whitespace is found in that range.
    """
    ideal_end = start + chunk_size
    if ideal_end >= len(text):
        return len(text)
    search_floor = max(start + 1, ideal_end - chunk_size // 5)
    boundary = text.rfind(" ", search_floor, ideal_end)
    if boundary == -1:
        boundary = text.rfind("\n", search_floor, ideal_end)
    return boundary if boundary != -1 else ideal_end


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
                end = _word_boundary_end(para, start, chunk_size)
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                # `end - overlap` is a raw offset too -- just as likely to
                # land mid-word as the old hard `end` cut was (e.g. restart
                # inside "word100" as "d100"), since overlap is a fixed
                # character count with no idea where a word starts. Snap
                # forward to the next space so the *next* chunk also only
                # ever starts on a word boundary.
                next_start = end - overlap
                space = para.find(" ", next_start, end)
                start = space + 1 if space != -1 else next_start
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
        from pypdf.errors import PdfReadError

        try:
            reader = PdfReader(io.BytesIO(content))
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
        except PdfReadError as exc:
            # A malformed/corrupt/non-PDF file named .pdf raises pypdf's own
            # exception type here, not ValueError -- without this, it was an
            # unhandled 500 instead of the clean 400 the upload endpoint
            # expects to be able to catch and show the user.
            raise ValueError(f"Could not read this file as a PDF: {exc}") from exc
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


QUERY_REWRITE_SYSTEM_PROMPT = (
    "Rewrite the follow-up question into a fully standalone question that "
    "makes sense without the conversation it came from -- resolve pronouns "
    "and implicit references (e.g. \"it\", \"that\", \"so\") using the "
    "conversation below. Keep the original language and intent. If the "
    "follow-up is already standalone, return it unchanged. Reply with only "
    "the rewritten question and nothing else -- no preamble, no quotes."
)

CLARIFY_SYSTEM_PROMPT = (
    "The retrieved context below gives different, conflicting answers to "
    "the user's question depending on details they haven't provided (such "
    "as product/vehicle variant, plan tier, or mileage). Write ONE short "
    "question back to the user that asks for exactly the missing detail "
    "needed to pick the right branch of the context. Reply with only that "
    "question and nothing else -- no preamble, no explanation. Match the "
    "language of the user's original question."
)


def build_rewrite_fn(backend: str, client):
    """Build a closure that turns a context-dependent follow-up ("What's
    its purpose, then?") into a standalone question a similarity search can
    actually retrieve against, using the last few turns of this session's
    conversation. Retrieval embeds the query in isolation -- it has no way
    to resolve a pronoun or an implicit back-reference to the previous
    answer, so an unresolved follow-up routinely scores low on every chunk
    and gets judged insufficient even when the document does cover it.

    Returns the query unchanged (no LLM call) when there's no history yet,
    since a first turn is standalone by definition.
    """

    def rewrite(query: str, history: list[dict]) -> str:
        if not history:
            return query
        convo = "\n".join(f"Q: {h['query']}\nA: {h['answer']}" for h in history)
        prompt = (
            f"{QUERY_REWRITE_SYSTEM_PROMPT}\n\nConversation so far:\n{convo}\n\n"
            f"Follow-up question: {query}\n\nStandalone question:"
        )
        if backend == "openai":
            response = client.chat.completions.create(
                model=GENERATE_MODEL_NAMES["openai"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )
            rewritten = (response.choices[0].message.content or "").strip()
        else:
            response = client.messages.create(
                model=GENERATE_MODEL_NAMES["anthropic"],
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            rewritten = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
        # Guard against a malformed/empty rewrite silently breaking retrieval.
        return rewritten or query

    return rewrite


def build_clarify_fn(backend: str, client):
    """Build a closure that writes the actual clarifying-question text when
    rag_guard's clarify stage flags a query as ambiguous.

    rag_guard itself only returns a boolean + probability (see
    `rag_guard.clarify.check_ambiguity`) -- it deliberately doesn't generate
    free text, so this lives in the host app instead, using the same
    generation backend/client as everything else.
    """

    def clarify(query: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
        prompt = f"{CLARIFY_SYSTEM_PROMPT}\n\nRetrieved context:\n{context}\n\nQuestion: {query}"
        if backend == "openai":
            response = client.chat.completions.create(
                model=GENERATE_MODEL_NAMES["openai"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
            )
            text = (response.choices[0].message.content or "").strip()
        else:
            response = client.messages.create(
                model=GENERATE_MODEL_NAMES["anthropic"],
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
        return text or "Could you clarify your question?"

    return clarify


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

    Returns (guard, retrieve, generate_fn, make_upload_generate_fn,
    rewrite_query_fn, backend_info, clarify_fn). `generate_fn` answers using
    the Nimbus-support system prompt against the demo corpus (chat.py only).
    `make_upload_generate_fn(usage_sink=None)` is a factory, not a fixed
    function -- app.py calls it fresh per request with a new `usage_sink`
    dict so it can read back that specific call's real token usage
    afterward, without building a whole new client each time.
    `rewrite_query_fn(query, history)` turns a context-dependent follow-up
    into a standalone one for retrieval (app.py only -- chat.py's REPL has
    no multi-turn history to rewrite against).
    `clarify_fn(query, chunks)` writes the actual clarifying-question text
    when rag_guard's clarify stage flags a query as ambiguous (app.py only).
    """
    client_backend, client = make_client()
    generate_fn = build_generate_fn(client_backend, client, SYSTEM_PROMPT)

    def make_upload_generate_fn(usage_sink: dict | None = None) -> GenerateFn:
        return build_generate_fn(client_backend, client, UPLOAD_SYSTEM_PROMPT, usage_sink=usage_sink)

    rewrite_query_fn = build_rewrite_fn(client_backend, client)
    clarify_fn = build_clarify_fn(client_backend, client)

    fallback_model = make_decision_model_for_client(client_backend, client)
    decision_backend, model = make_decision_model(fallback_model)
    collection = build_vector_store()
    retrieve = make_retriever(collection)
    # Lower sufficiency_threshold than RagGuard's 0.6 default: the intent
    # for this app is to let generation run more often and rely on the
    # grounding check (strict defaults, untouched) to catch an answer that
    # overreaches, rather than relying on sufficiency to pre-empt every
    # borderline case before generation ever gets a chance to run.
    # rerank_by_relevance: hand the generator the most relevant chunks
    # first rather than raw retrieval order -- cheap and reduces the odds
    # of a weak/off-topic chunk near the front nudging the answer astray.
    # partial_sufficiency_threshold: a mixed question (part answerable, part
    # not) was previously judged fully "insufficient" and silently refused
    # end to end -- this lets generation attempt the answerable part and
    # say what it can't cover, instead of going silent on the whole thing.
    # 0.5 here is a normal yes/no cutoff on its own dedicated question
    # (rag_guard.sufficiency.PARTIAL_QUESTION) -- not a low number tuned
    # against the main sufficiency probability, which doesn't carry this
    # signal at all (see that module's comment on why).
    guard = RagGuard(
        model=model,
        sufficiency_threshold=0.45,
        partial_sufficiency_threshold=0.5,
        rerank_by_relevance=True,
    )
    backend_info = {
        "generate_fn": client_backend,
        "generate_model": GENERATE_MODEL_NAMES[client_backend],
        "decision_model": decision_backend,
    }
    return guard, retrieve, generate_fn, make_upload_generate_fn, rewrite_query_fn, backend_info, clarify_fn
