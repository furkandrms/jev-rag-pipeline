"""Shared setup for both chat.py (REPL) and app.py (web UI): vector store,
retriever, and backend selection (generate_fn + DecisionModel), so the two
entry points never drift apart on how the pipeline gets wired.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

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
    "answer, say so plainly -- do not guess. If the context gives more than "
    "one number or value for what the question asks (e.g. results for "
    "different model variants, configurations, or conditions), do not pick "
    "one silently -- name which variant/configuration each number belongs "
    "to, or ask which one the user means if that's not already clear from "
    "the question."
)

GENERATE_MODEL_NAMES = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
    # Ollama exposes an OpenAI-compatible /v1 API, so the "ollama" backend
    # reuses every "openai"-shaped call below as-is -- only the model name
    # and base_url differ. Whatever tag was pulled in the Colab notebook
    # (see cloudflare-ollama-bridge), matched here via the same MODEL_NAME
    # env var that notebook's .env uses.
    "ollama": os.environ.get("MODEL_NAME", "qwen3:8b"),
}

_chroma_client = None
_embedding_function = None


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


# Display names for the comparison-mode UI (app.py's /api/compare) --
# every provider this app can build a real embedding function for, keyed
# the same way session collection name suffixes and API responses are.
EMBEDDING_PROVIDERS = {
    "openai": "OpenAI (text-embedding-3-small)",
    "gemini": "Google (gemini-embedding-001)",
}

_embedding_functions_by_provider: dict[str, object] = {}


def get_embedding_function_for(provider: str):
    """Build (and cache) the real embedding function for one named provider.

    Used directly by comparison mode, which needs both providers available
    at once regardless of which one `get_embedding_function()` below picked
    as this app's single default.
    """
    if provider in _embedding_functions_by_provider:
        return _embedding_functions_by_provider[provider]

    from chromadb.utils import embedding_functions

    if provider == "openai":
        fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"], model_name="text-embedding-3-small"
        )
    elif provider == "gemini":
        fn = embedding_functions.GoogleGenaiEmbeddingFunction(
            model_name="gemini-embedding-001", api_key_env_var="GOOGLE_API_KEY"
        )
    else:
        raise ValueError(f"unknown embedding provider: {provider}")
    _embedding_functions_by_provider[provider] = fn
    return fn


def get_embedding_function():
    """Pick the best embedding available instead of chromadb's silent
    default (a small local sentence-transformers model, all-MiniLM-L6-v2).

    That default measurably under-retrieves on real documents: a real-world
    eval against the same golden set (eval_multi_embedding.py) found it
    missed the correct chunk ~40% of the time at this app's TOP_K, while
    OpenAI's and Gemini's embedding APIs both hit 100%. A broad "what is
    this document about" query is exactly the shape of question that
    exposes the gap -- e.g. a paper's one sentence actually stating its
    contribution can simply never make the top-k candidate list under the
    default embedding, no amount of relevance/sufficiency tuning downstream
    can recover a chunk retrieval never surfaced in the first place.

    Prefers OpenAI (already a required credential for generation in the
    common case) over Gemini over the local default, which still works --
    just worse -- so the app runs with no embedding-specific config at all.
    """
    global _embedding_function
    if _embedding_function is not None:
        return _embedding_function

    from chromadb.utils import embedding_functions

    if os.environ.get("OPENAI_API_KEY"):
        _embedding_function = get_embedding_function_for("openai")
    elif os.environ.get("GOOGLE_API_KEY"):
        _embedding_function = get_embedding_function_for("gemini")
    else:
        print(
            "No OPENAI_API_KEY or GOOGLE_API_KEY set -- falling back to chromadb's "
            "local default embedding, which retrieves noticeably worse on real "
            "documents (see get_embedding_function()'s docstring).",
            file=sys.stderr,
        )
        _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_function


# Gemini's embedContent batch endpoint rejects more than 100 inputs in one
# call ("at most 100 requests can be in one batch") -- verified live against
# a 151-chunk real PDF, which failed `collection.add()` outright with every
# chunk in one call. OpenAI's limit is far higher, but batching unconditionally
# here is simpler and safer than branching on which provider is active.
_ADD_BATCH_SIZE = 90

# Gemini's free tier additionally rate-limits to 100 embed requests/minute
# (also verified live -- a 429 RESOURCE_EXHAUSTED mid-upload, not just a
# too-large-batch error). chromadb wraps every provider's error as a plain
# ValueError, so there's no typed "retry-able" exception to catch -- retry
# generically on any failure a couple of times with backoff rather than
# fail an upload outright over a transient, minute-scale limit.
_ADD_RETRY_ATTEMPTS = 3
_ADD_RETRY_BASE_DELAY_SECONDS = 20.0

_embedding_provider_broken = False


def _is_auth_error(exc: Exception) -> bool:
    """True for a non-retryable authentication/permission failure (HTTP 401
    or 403), whether raised directly (OpenAI/Google SDK exceptions carry a
    `.status_code`) or wrapped by chromadb, which sometimes re-raises as a
    plain ValueError with the original exception chained via `__cause__`.

    A bad or revoked key never recovers by waiting -- retrying it
    `_ADD_RETRY_ATTEMPTS` times with up to a minute of backoff (meant for
    transient failures, see above) only delays an inevitable failure.
    """
    seen: BaseException | None = exc
    for _ in range(3):
        if getattr(seen, "status_code", None) in (401, 403):
            return True
        seen = seen.__cause__ or seen.__context__
        if seen is None:
            return False
    return False


def _fall_back_to_local_embedding(reason: Exception) -> None:
    """Switch every future `get_embedding_function()` call to chromadb's
    local default embedding, and say why.

    Called once an OpenAI/Gemini embedding call has actually failed (bad or
    revoked key, persistent API error) instead of letting that exception
    crash the whole app -- app.py calls `build_vector_store()` at import
    time, before the server even starts listening, so an unhandled failure
    here previously took the whole process down before a single request
    could be served. Remembered globally (not just for this one call) so a
    later per-session upload (`replace_session_document`) doesn't pay the
    same failing-provider retry cost again.
    """
    global _embedding_function, _embedding_provider_broken
    from chromadb.utils import embedding_functions

    print(
        f"Embedding provider failed ({reason!r}) -- falling back to "
        "chromadb's local default embedding for the rest of this run "
        "instead of crashing. Check OPENAI_API_KEY/GOOGLE_API_KEY if this "
        "is unexpected; retrieval quality will be noticeably worse on real "
        "documents (see get_embedding_function()'s docstring).",
        file=sys.stderr,
    )
    _embedding_function = embedding_functions.DefaultEmbeddingFunction()
    _embedding_provider_broken = True


def _add_chunks_in_batches(collection, ids: list[str], documents: list[str]) -> None:
    for start in range(0, len(ids), _ADD_BATCH_SIZE):
        end = start + _ADD_BATCH_SIZE
        batch_ids, batch_docs = ids[start:end], documents[start:end]
        for attempt in range(_ADD_RETRY_ATTEMPTS):
            try:
                collection.add(ids=batch_ids, documents=batch_docs)
                break
            except Exception as exc:
                if _is_auth_error(exc) or attempt == _ADD_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_ADD_RETRY_BASE_DELAY_SECONDS * (attempt + 1))


def _add_chunks_with_fallback(
    client, collection, collection_name: str, ids: list[str], documents: list[str], *, recreate
):
    """Add `ids`/`documents` to the already-created `collection`, falling
    back to local embedding and retrying once if the configured provider
    fails.

    `recreate(embedding_function)` must (re)create and return the named
    collection with that embedding function, used only for the fallback
    rebuild -- `build_vector_store()` and `replace_session_document()` each
    create a collection slightly differently (get_or_create vs. create), so
    that part stays with the caller.
    """
    try:
        _add_chunks_in_batches(collection, ids, documents)
    except Exception as exc:
        _fall_back_to_local_embedding(exc)
        client.delete_collection(collection_name)
        collection = recreate(get_embedding_function())
        _add_chunks_in_batches(collection, ids, documents)
    return collection


def build_vector_store():
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        COLLECTION_NAME, embedding_function=get_embedding_function()
    )
    if collection.count() == 0:
        ids, texts = zip(*DOCUMENTS)
        collection = _add_chunks_with_fallback(
            client,
            collection,
            COLLECTION_NAME,
            list(ids),
            list(texts),
            recreate=lambda fn: client.get_or_create_collection(
                COLLECTION_NAME, embedding_function=fn
            ),
        )
    return collection


# --- Sample documents (optional, served from GCS) ---------------------------
# A small fixed catalog of real papers for a "try a sample document" picker,
# so a visitor can see the full pipeline run on a real document without
# having one of their own to upload. Entirely optional: only usable when
# SAMPLE_DOCS_BUCKET is set (e.g. in the Cloud Run deployment) -- local dev
# without GCS credentials just doesn't advertise the feature (see
# sample_docs_available(), which app.py's /api/samples returns verbatim).
SAMPLE_DOCS = [
    {"id": "attention", "filename": "01_nlp_attention_is_all_you_need.pdf", "title": "Attention Is All You Need", "topic": "NLP / Transformers"},
    {"id": "climate-ml", "filename": "02_climate_ml_review.pdf", "title": "Climate Modeling with Machine Learning", "topic": "Climate science"},
    {"id": "quantum-cloud", "filename": "03_quantum_cloud_review.pdf", "title": "Quantum Cloud Computing: A Review", "topic": "Quantum computing"},
    {"id": "inflation-fewnet", "filename": "04_inflation_forecasting_fewnet.pdf", "title": "Inflation Forecasting with FewNet", "topic": "Economics"},
    {"id": "autonomous-driving", "filename": "05_autonomous_driving_e2e.pdf", "title": "End-to-End Autonomous Driving", "topic": "Robotics / autonomous vehicles"},
    {"id": "resnet", "filename": "06_resnet.pdf", "title": "Deep Residual Learning (ResNet)", "topic": "Computer vision"},
    {"id": "intrusion-detection", "filename": "07_intrusion_detection_survey.pdf", "title": "Intrusion Detection: A Survey", "topic": "Cybersecurity"},
    {"id": "crispr-grna", "filename": "08_crispr_grna_ai.pdf", "title": "AI for CRISPR Guide RNA Design", "topic": "Bioinformatics"},
    {"id": "solar-power", "filename": "09_solar_power_prediction.pdf", "title": "Solar Power Prediction", "topic": "Renewable energy"},
    {"id": "covid-imaging", "filename": "10_covid_medical_imaging_dl.pdf", "title": "COVID Medical Imaging with Deep Learning", "topic": "Medical imaging"},
]
_SAMPLE_DOCS_BY_ID = {d["id"]: d for d in SAMPLE_DOCS}

_gcs_client = None


def _get_gcs_client():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage

        _gcs_client = storage.Client()
    return _gcs_client


def sample_docs_available() -> list[dict]:
    """The sample-doc catalog (id/title/topic only, no filenames leaked),
    or an empty list if no bucket is configured. The frontend hides the
    picker entirely when this is empty rather than showing a broken one.
    """
    if not os.environ.get("SAMPLE_DOCS_BUCKET"):
        return []
    return [{"id": d["id"], "title": d["title"], "topic": d["topic"]} for d in SAMPLE_DOCS]


def fetch_sample_doc(sample_id: str) -> tuple[str, bytes]:
    """Downloads one sample document's raw bytes from GCS.

    Raises KeyError for an unknown id (caller turns that into a 404) and
    lets any GCS client exception (missing bucket/object, auth failure)
    propagate -- the caller turns that into a 502, since it's an upstream
    fetch failure, not a client error.
    """
    doc = _SAMPLE_DOCS_BY_ID[sample_id]
    bucket_name = os.environ["SAMPLE_DOCS_BUCKET"]
    bucket = _get_gcs_client().bucket(bucket_name)
    blob = bucket.blob(doc["filename"])
    return doc["filename"], blob.download_as_bytes()


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
    collection = client.create_collection(name, embedding_function=get_embedding_function())
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    return _add_chunks_with_fallback(
        client,
        collection,
        name,
        ids,
        chunks,
        recreate=lambda fn: client.create_collection(name, embedding_function=fn),
    )


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
    remove_compare_collections(session_id)


def compare_collection_name(session_id: str, provider: str) -> str:
    return f"{UPLOAD_COLLECTION_PREFIX}{session_id}_cmp_{provider}"


def get_compare_collection(session_id: str, provider: str):
    """Returns this session's comparison-mode collection for `provider`, or
    None if it hasn't been built yet for the current document.
    """
    client = get_chroma_client()
    name = compare_collection_name(session_id, provider)
    try:
        collection = client.get_collection(name)
    except Exception:
        return None
    return collection if collection.count() > 0 else None


# Guards ensure_compare_collections() per session so two /api/compare
# requests racing on the same session can't both see "not built yet",
# each delete-and-recreate the other's in-progress collection, and crash
# with a chromadb NotFoundError on the loser's collection.add() call --
# verified live, see the compare-mode hang/crash investigation.
_compare_build_locks: dict[str, threading.Lock] = {}
_compare_build_locks_guard = threading.Lock()


def _compare_lock_for(session_id: str) -> threading.Lock:
    with _compare_build_locks_guard:
        return _compare_build_locks.setdefault(session_id, threading.Lock())


def _build_one_compare_collection(session_id: str, filename: str, chunks: list[str], provider: str) -> None:
    client = get_chroma_client()
    name = compare_collection_name(session_id, provider)
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = client.create_collection(name, embedding_function=get_embedding_function_for(provider))
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    _add_chunks_in_batches(collection, ids, chunks)


def ensure_compare_collections(session_id: str, filename: str, chunks: list[str]) -> None:
    """Build this session's per-provider comparison collections if they
    don't already exist, embedding the same chunks the single-provider
    collection uses.

    Built lazily on first use of compare mode, not on every upload -- most
    sessions never open compare mode, and building both costs two full
    embedding API passes over the document for nothing if it's never used.

    Runs under a per-session lock (a second concurrent call just waits and
    then finds everything already built) and builds whichever providers are
    missing in parallel rather than one after another -- on a large document,
    embedding through two providers sequentially (each with its own batching
    and rate-limit retries) is what was stretching a single compare request
    past two minutes.
    """
    with _compare_lock_for(session_id):
        missing = [p for p in EMBEDDING_PROVIDERS if get_compare_collection(session_id, p) is None]
        if not missing:
            return
        with ThreadPoolExecutor(max_workers=len(missing)) as pool:
            list(
                pool.map(
                    lambda provider: _build_one_compare_collection(session_id, filename, chunks, provider),
                    missing,
                )
            )


def remove_compare_collections(session_id: str) -> None:
    client = get_chroma_client()
    for provider in EMBEDDING_PROVIDERS:
        try:
            client.delete_collection(compare_collection_name(session_id, provider))
        except Exception:
            pass  # nothing to delete
    with _compare_build_locks_guard:
        _compare_build_locks.pop(session_id, None)


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


TOP_K_FRACTION = 0.08  # of the collection's total chunk count
TOP_K_MAX = 30  # relevance-check cost scales with k (one decision call per
# candidate, parallelized up to 8 at a time -- see rag_guard/relevance.py),
# so this caps how far a huge document can push it up


def _effective_top_k(collection) -> int:
    """TOP_K scaled to the document's size.

    A fixed TOP_K=12 means a 50-chunk document gets ~24% of itself as
    retrieval candidates per query, but a 250-chunk one gets ~5% -- on a
    large document a broad question (e.g. "what are the open problems")
    routinely has its one relevant section simply not make the candidate
    list at all, which no amount of relevance-threshold tuning downstream
    can recover from. Scaling with `collection.count()` keeps candidate
    coverage roughly proportional instead of collapsing on large docs,
    capped at TOP_K_MAX so relevance-check cost doesn't grow unbounded.
    """
    total_chunks = collection.count()
    return min(TOP_K_MAX, max(TOP_K, round(total_chunks * TOP_K_FRACTION)))


def make_retriever(collection):
    def retrieve(query: str, k: int | None = None) -> list[Chunk]:
        if k is None:
            k = _effective_top_k(collection)
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

    # Free path: a model server from cloudflare-ollama-bridge (Ollama on a
    # Colab GPU, exposed through a Cloudflare tunnel). Ollama's /v1 API is
    # OpenAI-compatible, so the OpenAI SDK itself is the client here too --
    # just pointed at the tunnel instead of api.openai.com. The API key is
    # a required-but-unchecked placeholder; Ollama ignores it.
    if os.environ.get("MODEL_SERVER_URL"):
        from openai import OpenAI

        base_url = os.environ["MODEL_SERVER_URL"].rstrip("/") + "/v1"
        return "ollama", OpenAI(base_url=base_url, api_key="ollama")

    print(
        "Needs a real LLM to generate answers: set OPENAI_API_KEY, "
        "ANTHROPIC_API_KEY, or MODEL_SERVER_URL for a free Ollama backend "
        "(see .env.example and cloudflare-ollama-bridge).",
        file=sys.stderr,
    )
    raise SystemExit(1)


def build_generate_fn(
    backend: str,
    client,
    system_prompt: str,
    usage_sink: dict | None = None,
    model: str | None = None,
) -> GenerateFn:
    """Build the generation closure for `backend`.

    `usage_sink`, if given, gets the real token usage from the underlying
    API response written into it as a side effect (`{"input_tokens": ...,
    "output_tokens": ...}`) -- the only way to surface it, since `GenerateFn`
    itself must return a plain string to satisfy RagGuard's contract.

    `model`, if given, overrides `GENERATE_MODEL_NAMES[backend]` -- used for
    a per-request custom Ollama server (a visitor's own tunnel/model, not
    the one this process started with), see `build_custom_ollama_client`.
    """
    if backend in ("openai", "ollama"):
        # Same call shape for both: Ollama's /v1 API is OpenAI-compatible.
        model_name = model or GENERATE_MODEL_NAMES[backend]

        def generate(query: str, chunks: list[Chunk]) -> str:
            context = "\n\n".join(f"[{c.id}] {c.text}" for c in chunks)
            response = client.chat.completions.create(
                model=model_name,
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
        if backend in ("openai", "ollama"):
            response = client.chat.completions.create(
                model=GENERATE_MODEL_NAMES[backend],
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
        if backend in ("openai", "ollama"):
            response = client.chat.completions.create(
                model=GENERATE_MODEL_NAMES[backend],
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
    if backend == "ollama":
        # Same OpenAI-compatible JSON-mode call as the "openai" case above,
        # just against a smaller local model -- see make_client()'s comment
        # on the tradeoff (JSON mode is less reliable on small models than
        # on gpt-4o-mini).
        return OpenAIDecisionModel(model=GENERATE_MODEL_NAMES["ollama"], client=client)
    raise ValueError(f"unknown backend: {backend}")


def make_decision_model(fallback: DecisionModel) -> tuple[str, DecisionModel]:
    if os.environ.get("TYPESAFE_API_KEY"):
        return "jev", JevDecisionModel()
    return "same as generate_fn", fallback


# --- Custom per-request Ollama backend ---------------------------------------
# Lets a single visitor use their own free Colab+cloudflare-ollama-bridge
# tunnel for just their own requests, instead of (or when there is no)
# whatever OPENAI_API_KEY/ANTHROPIC_API_KEY/MODEL_SERVER_URL this process
# started with -- app.py's /api/chat(/stream) accept an optional
# model_server_url/model_name and, if given, build this per-request instead
# of using the module-level default. Same shape as the "Use Laya" toggle
# (a per-request override), not a server-wide setting.
class InvalidCustomBackend(ValueError):
    pass


# Same allowlist as cloudflare-ollama-bridge's own demo/app.py: a visitor
# types this URL and we make a server-side request to it on their behalf --
# a classic SSRF shape if left unvalidated (they could otherwise point this
# server at an internal service or a cloud metadata endpoint). Restricted to
# what this feature is actually for: a Cloudflare quick-tunnel host, or a
# local Ollama instance.
_CUSTOM_OLLAMA_HOST_RE = re.compile(r"^([a-z0-9-]+\.trycloudflare\.com|localhost|127\.0\.0\.1)$")
_CUSTOM_OLLAMA_MODEL_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


def validate_custom_ollama_target(server_url: str, model_name: str) -> tuple[str, str]:
    """Validate and normalize a visitor-supplied tunnel URL + model name.

    Raises InvalidCustomBackend with a message safe to show the visitor
    directly (no internals) if either is missing or outside the allowlist.
    """
    server_url = (server_url or "").strip().rstrip("/")
    model_name = (model_name or "").strip()

    if not server_url:
        raise InvalidCustomBackend("No tunnel URL given.")

    parsed = urlparse(server_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or not _CUSTOM_OLLAMA_HOST_RE.match(host):
        raise InvalidCustomBackend(
            "Refused: tunnel URL must be a *.trycloudflare.com tunnel, or "
            f"localhost/127.0.0.1 -- got {host or '(none)'!r}."
        )
    if not model_name or not _CUSTOM_OLLAMA_MODEL_RE.match(model_name):
        raise InvalidCustomBackend("Refused: model name is missing or contains unexpected characters.")
    return server_url, model_name


def build_custom_ollama_backend(server_url: str, model_name: str):
    """Build a one-off (generate_fn_factory, RagGuard) pair for a
    visitor-supplied tunnel URL + model name, validated against the same
    allowlist as `validate_custom_ollama_target`.

    `generate_fn_factory(usage_sink=None)` mirrors `make_upload_generate_fn`
    from `setup()` -- a factory, not a fixed function, so app.py can build a
    fresh one per request with a new usage_sink dict.
    """
    server_url, model_name = validate_custom_ollama_target(server_url, model_name)
    from openai import OpenAI

    client = OpenAI(base_url=f"{server_url}/v1", api_key="ollama")

    def generate_fn_factory(usage_sink: dict | None = None) -> GenerateFn:
        return build_generate_fn(
            "ollama", client, UPLOAD_SYSTEM_PROMPT, usage_sink=usage_sink, model=model_name
        )

    guard = RagGuard(
        model=OpenAIDecisionModel(model=model_name, client=client),
        sufficiency_threshold=0.45,
        partial_sufficiency_threshold=0.5,
        rerank_by_relevance=True,
    )
    return generate_fn_factory, guard


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


# --- Laya: an opt-in, self-hosted decision-model backend --------------------
# A second RagGuard, identical to setup()'s except for which DecisionModel
# backs it, built lazily (only if/when a request actually opts in via
# ChatRequest.use_laya) so a visitor who never touches the toggle never
# pays Laya's checkpoint download/load cost. See laya_decision_model.py for
# what this trades off against the default OpenAI/Anthropic-backed guard.
_laya_guard_instance: RagGuard | None = None
_laya_guard_lock = threading.Lock()


def get_laya_guard() -> RagGuard:
    global _laya_guard_instance
    if _laya_guard_instance is None:
        with _laya_guard_lock:
            if _laya_guard_instance is None:
                from laya_decision_model import LayaDecisionModel

                _laya_guard_instance = RagGuard(
                    model=LayaDecisionModel(),
                    sufficiency_threshold=0.45,
                    partial_sufficiency_threshold=0.5,
                    rerank_by_relevance=True,
                )
    return _laya_guard_instance
