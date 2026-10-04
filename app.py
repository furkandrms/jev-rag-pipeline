"""FastAPI web app: upload your own document, then chat against it through rag-guard.

There is no example/demo corpus in this app -- every answer is grounded in
whatever document the visitor uploaded themselves, chunked and embedded at
request time into a collection scoped to their browser session. (The
Nimbus demo corpus still exists for chat.py, the terminal REPL used to
smoke-test rag-guard's pipeline wiring -- it's a separate entry point and
intentionally not part of this web app.)

JSON/SSE API the frontend calls:

  GET  /api/document-status -- whether this session already has an
                                uploaded document, its chunk list, and
                                chunk count (so a page reload can skip
                                straight back to the chat view).
  POST /api/upload           -- chunk an uploaded .txt/.md/.pdf into this
                                session's Chroma collection. A new upload
                                replaces the previous one for that session
                                and clears its response cache (see below).
  DELETE /api/document        -- remove this session's document outright
                                (no replacement) and clear its cache, so
                                the UI can offer a real "remove" action,
                                not just "upload a different one".
  POST /api/chat             -- plain JSON request/response version of the
                                pipeline (used by curl/tests); blocks until
                                the full result is ready.
  POST /api/chat/stream      -- Server-Sent Events version of the same
                                pipeline: emits one event per rag-guard
                                stage (retrieval, relevance, sufficiency,
                                generation, grounding) as it actually
                                completes, each with that stage's real
                                elapsed time, finishing with a `done` event
                                carrying the full report. This is what the
                                web UI uses for the live-progress view.
  GET  /api/info             -- active backends, model names, and
                                RagGuard's thresholds.

Sessions are scoped by an `X-Session-Id` header the frontend mints itself
(crypto.randomUUID(), persisted in localStorage) -- there's no login, just
enough identity to keep one browser's uploaded document out of another's.
See pipeline_setup.py's module docstring for the isolation guarantee this
relies on.

Run locally:
    uvicorn app:app --reload

Both endpoints run the *same* stage-by-stage orchestration
(`_run_pipeline`, a generator) -- it mirrors exactly what `RagGuard.run()`
does internally (relevance -> sufficiency -> generate -> grounding), just
broken into visible steps instead of one opaque call, so real per-stage
timing can be measured and streamed. `/api/chat` simply drains the
generator and returns its last event; `/api/chat/stream` relays every
event over SSE as it's produced.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline_logging import stage_log
from pipeline_setup import (
    chunk_text,
    extract_text,
    get_session_collection,
    make_retriever,
    remove_session_document,
    replace_session_document,
    setup,
)
from rag_guard import Chunk
from rag_guard.grounding import check_grounding
from rag_guard.relevance import filter_relevant_chunks
from rag_guard.sufficiency import check_sufficiency

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB -- generous for text/markdown/small PDFs

app = FastAPI(title="rag-guard document chat")

# setup() also builds the Nimbus demo corpus and its generate_fn, since
# chat.py (a separate entry point) needs them -- this app only uses the
# pieces relevant to the upload flow (_guard, _make_upload_generate_fn).
_guard, _, _, _make_upload_generate_fn, _backend_info = setup()

# Per-session state, in-memory (mirrors the in-memory Chroma client it
# describes -- both reset on process restart, see README's "known
# limitations" section):
#   _session_meta: filename/chunk list/chunk count, for /api/document-status
#   _response_cache: (session_id, normalized query) -> last full report,
#     so re-asking the same question skips the pipeline entirely. Cleared
#     for a session whenever that session uploads a new document, since a
#     cached answer is only valid for the document it was computed against.
_session_meta: dict[str, dict] = {}
_response_cache: dict[tuple[str, str], dict] = {}


class ChatRequest(BaseModel):
    query: str


def _elapsed_ms(start: float) -> float:
    return round((time.monotonic() - start) * 1000, 1)


def _clear_session_cache(session_id: str) -> None:
    for key in [k for k in _response_cache if k[0] == session_id]:
        del _response_cache[key]


def _serialize_relevance(relevance) -> list[dict]:
    return [
        {
            "chunk_id": r.chunk.id,
            "text": r.chunk.text,
            "probability": r.probability,
            "kept": r.kept,
        }
        for r in relevance
    ]


def _serialize_grounding(grounding) -> dict | None:
    if grounding is None:
        return None
    return {
        "grounded": grounding.grounded,
        "coverage": grounding.coverage,
        "claims": [asdict(c) for c in grounding.claims],
    }


def _run_pipeline(query: str, session_id: str):
    """Generator yielding (event_name, data) as each real pipeline stage completes.

    Mirrors RagGuard.run()'s own stage order exactly (see
    rag_guard/pipeline.py) -- this isn't a reimplementation of the
    decision logic, just the same public calls RagGuard.run() makes
    internally, called directly instead of through that one wrapper, so
    each stage's completion (and real elapsed time) can be observed and
    streamed as it happens instead of only seeing the final result.
    """
    t_total = time.monotonic()
    stage_log(session_id, "query", f'"{query}"')

    collection = get_session_collection(session_id)
    if collection is None:
        stage_log(session_id, "error", "no document uploaded yet", level=logging.WARNING)
        yield "error", {"detail": "No document uploaded yet for this session"}
        return

    cache_key = (session_id, query.strip().lower())
    cached = _response_cache.get(cache_key)
    if cached is not None:
        elapsed = _elapsed_ms(t_total)
        stage_log(session_id, "cache_hit", f"served in {elapsed}ms, pipeline skipped")
        yield "cached", {**cached, "total_elapsed_ms": elapsed}
        return

    t = time.monotonic()
    chunks = make_retriever(collection)(query)

    # Broad "what is this document about?" queries often have little
    # lexical/semantic overlap with the document's own title or opening
    # paragraph, so pure similarity search can miss it even though it's
    # usually the single most useful chunk for exactly that kind of
    # question -- anchor it in unconditionally rather than leaving it to
    # chance, same as a human skimming a document would start at the top.
    meta = _session_meta.get(session_id)
    anchor_added = False
    if meta and meta.get("chunks") and not any(c.id.endswith("-0") for c in chunks):
        anchor = Chunk(
            id=f"{meta['filename']}-0", text=meta["chunks"][0], metadata={"anchor": True}
        )
        chunks = [anchor, *chunks]
        anchor_added = True

    elapsed = _elapsed_ms(t)
    stage_log(
        session_id,
        "retrieval",
        f"{len(chunks)} chunks fetched ({elapsed}ms)"
        + (" [+anchor: doc opening chunk]" if anchor_added else ""),
    )
    yield "retrieval", {"chunk_count": len(chunks), "elapsed_ms": elapsed}

    t = time.monotonic()
    relevance = filter_relevant_chunks(
        query, chunks, _guard.model, threshold=_guard.relevance_threshold
    )
    kept_chunks = [r.chunk for r in relevance if r.kept]
    elapsed = _elapsed_ms(t)
    stage_log(
        session_id,
        "relevance",
        f"kept {len(kept_chunks)}/{len(relevance)} chunks "
        f"(threshold={_guard.relevance_threshold}) ({elapsed}ms)",
    )
    for r in relevance:
        stage_log(
            session_id,
            "relevance",
            f"  [{'KEEP' if r.kept else 'drop'}] p={r.probability:.2f}  {r.chunk.id}: "
            f"{r.chunk.text[:80]!r}",
            level=logging.DEBUG,
        )
    yield "relevance", {"relevance": _serialize_relevance(relevance), "elapsed_ms": elapsed}

    t = time.monotonic()
    sufficiency = check_sufficiency(
        query, kept_chunks, _guard.model, threshold=_guard.sufficiency_threshold
    )
    elapsed = _elapsed_ms(t)
    stage_log(
        session_id,
        "sufficiency",
        f"p={sufficiency.probability:.2f} (threshold={_guard.sufficiency_threshold}) -> "
        f"{'sufficient' if sufficiency.sufficient else 'INSUFFICIENT'} ({elapsed}ms)",
        level=logging.INFO if sufficiency.sufficient else logging.WARNING,
    )
    yield "sufficiency", {"sufficiency": asdict(sufficiency), "elapsed_ms": elapsed}

    if not sufficiency.sufficient:
        total = _elapsed_ms(t_total)
        stage_log(
            session_id,
            "done",
            f"action=insufficient_context, generation skipped, total={total}ms",
            level=logging.WARNING,
        )
        report = {
            "query": query,
            "action": "insufficient_context",
            "answer": _guard.insufficient_context_message,
            "relevance": _serialize_relevance(relevance),
            "sufficiency": asdict(sufficiency),
            "grounding": None,
            "metrics": {"decision_calls": len(relevance) + 1, "generation_usage": None},
        }
        _response_cache[cache_key] = report
        yield "done", {"report": report, "total_elapsed_ms": total}
        return

    usage_sink: dict = {}
    t = time.monotonic()
    generate_fn = _make_upload_generate_fn(usage_sink=usage_sink)
    answer = generate_fn(query, kept_chunks)
    elapsed = _elapsed_ms(t)
    model_name = _backend_info.get("generate_model", _backend_info["generate_fn"])
    tokens_in = usage_sink.get("input_tokens", "?")
    tokens_out = usage_sink.get("output_tokens", "?")
    stage_log(
        session_id,
        "generation",
        f"{model_name} | in={tokens_in} out={tokens_out} tokens ({elapsed}ms) -> {answer[:100]!r}",
    )
    yield "generation", {"answer": answer, "usage": usage_sink, "elapsed_ms": elapsed}

    grounding = None
    decision_calls = len(relevance) + 1
    if _guard.check_grounding_enabled:
        t = time.monotonic()
        grounding = check_grounding(
            answer,
            kept_chunks,
            _guard.model,
            threshold=_guard.grounding_threshold,
            min_coverage=_guard.grounding_min_coverage,
        )
        decision_calls += len(grounding.claims)
        elapsed = _elapsed_ms(t)
        supported = sum(1 for c in grounding.claims if c.supported)
        stage_log(
            session_id,
            "grounding",
            f"coverage={grounding.coverage:.2f} ({supported}/{len(grounding.claims)} claims "
            f"supported, min_coverage={_guard.grounding_min_coverage}) -> "
            f"{'grounded' if grounding.grounded else 'NOT GROUNDED -- flagging'} ({elapsed}ms)",
            level=logging.INFO if grounding.grounded else logging.WARNING,
        )
        for c in grounding.claims:
            if not c.supported:
                stage_log(
                    session_id,
                    "grounding",
                    f"  [UNSUPPORTED] p={c.probability:.2f}  {c.claim!r}",
                    level=logging.WARNING,
                )
        yield "grounding", {
            "grounding": _serialize_grounding(grounding),
            "elapsed_ms": elapsed,
        }
        action = "answered" if grounding.grounded else "ungrounded_answer_flagged"
    else:
        action = "answered"

    total = _elapsed_ms(t_total)
    stage_log(
        session_id,
        "done",
        f"action={action}, decision_calls={decision_calls}, "
        f"tokens={tokens_in}in/{tokens_out}out, total={total}ms",
        level=logging.INFO if action == "answered" else logging.WARNING,
    )
    report = {
        "query": query,
        "action": action,
        "answer": answer,
        "relevance": _serialize_relevance(relevance),
        "sufficiency": asdict(sufficiency),
        "grounding": _serialize_grounding(grounding),
        "metrics": {"decision_calls": decision_calls, "generation_usage": usage_sink or None},
    }
    _response_cache[cache_key] = report
    yield "done", {"report": report, "total_elapsed_ms": total}


@app.get("/api/info")
def info() -> dict:
    return {
        **_backend_info,
        "thresholds": {
            "relevance_threshold": _guard.relevance_threshold,
            "sufficiency_threshold": _guard.sufficiency_threshold,
            "grounding_threshold": _guard.grounding_threshold,
            "grounding_min_coverage": _guard.grounding_min_coverage,
        },
    }


@app.get("/api/document-status")
def document_status(x_session_id: str = Header(..., alias="X-Session-Id")) -> dict:
    meta = _session_meta.get(x_session_id)
    if meta is None or get_session_collection(x_session_id) is None:
        return {"uploaded": False}
    return {"uploaded": True, **meta}


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    x_session_id: str = Header(..., alias="X-Session-Id"),
) -> dict:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")

    try:
        text = extract_text(file.filename or "upload.txt", content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(400, "No extractable text found in that file")

    replace_session_document(x_session_id, file.filename or "upload", chunks)
    _clear_session_cache(x_session_id)  # a new document invalidates cached answers from the old one

    meta = {"filename": file.filename, "chunk_count": len(chunks), "chunks": chunks}
    _session_meta[x_session_id] = meta
    stage_log(x_session_id, "upload", f"{file.filename!r} -> {len(chunks)} chunks indexed")
    return meta


@app.delete("/api/document")
def delete_document(x_session_id: str = Header(..., alias="X-Session-Id")) -> dict:
    remove_session_document(x_session_id)
    _clear_session_cache(x_session_id)
    _session_meta.pop(x_session_id, None)
    stage_log(x_session_id, "remove", "document removed")
    return {"removed": True}


@app.post("/api/chat")
def chat(request: ChatRequest, x_session_id: str = Header(..., alias="X-Session-Id")) -> dict:
    query = request.query.strip()
    if not query:
        return {"error": "empty query"}

    event, data = None, None
    for event, data in _run_pipeline(query, x_session_id):
        pass  # drain to the final event; intermediate stages are for /stream only

    if event == "error":
        raise HTTPException(400, data["detail"])
    return data.get("report", data)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest, x_session_id: str = Header(..., alias="X-Session-Id")):
    query = request.query.strip()

    def sse():
        if not query:
            yield f"event: error\ndata: {json.dumps({'detail': 'empty query'})}\n\n"
            return
        for event, data in _run_pipeline(query, x_session_id):
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
