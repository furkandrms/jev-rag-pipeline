"""FastAPI web app: upload your own document, then chat against it through rag-guard.

There is no example/demo corpus in this app -- every answer is grounded in
whatever document the visitor uploaded themselves, chunked and embedded at
request time into a collection scoped to their browser session. (The
Nimbus demo corpus still exists for chat.py, the terminal REPL used to
smoke-test rag-guard's pipeline wiring -- it's a separate entry point and
intentionally not part of this web app.)

JSON API the frontend calls:

  GET  /api/document-status -- whether this session already has an
                                uploaded document (so a page reload can
                                skip straight back to the chat view).
  POST /api/upload           -- chunk an uploaded .txt/.md/.pdf into this
                                session's Chroma collection. A new upload
                                replaces the previous one for that session.
  POST /api/chat             -- runs RagGuard.run() against this session's
                                document and returns the full decision
                                trace (relevance per chunk, sufficiency,
                                grounding per claim) alongside the answer,
                                so the UI can show not just *what*
                                rag-guard answered but *why*.
  GET  /api/info             -- active backends + RagGuard's thresholds.

Sessions are scoped by an `X-Session-Id` header the frontend mints itself
(crypto.randomUUID(), persisted in localStorage) -- there's no login, just
enough identity to keep one browser's uploaded document out of another's.
See pipeline_setup.py's module docstring for the isolation guarantee this
relies on.

Run locally:
    uvicorn app:app --reload

The LLM/decision-model backend selection happens once at import time
(module-level globals below), not per-request -- re-resolving API clients
on every chat message would be wasteful. Per-session document state
(`_session_meta`) is an in-memory dict, matching the in-memory Chroma
client it mirrors -- both reset on process restart, by design for this
scope (see README for the production-hardening list this implies).
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline_setup import (
    chunk_text,
    extract_text,
    get_session_collection,
    make_retriever,
    replace_session_document,
    setup,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB -- generous for text/markdown/small PDFs

app = FastAPI(title="rag-guard document chat")

# setup() also builds the Nimbus demo corpus and its generate_fn, since
# chat.py (a separate entry point) needs them -- this app only uses the
# pieces relevant to the upload flow (_guard, _upload_generate_fn).
_guard, _, _, _upload_generate_fn, _backend_info = setup()

# filename/chunk_count per session, for GET /api/document-status -- Chroma
# itself doesn't track this metadata, so it's kept alongside the in-memory
# vector store (see pipeline_setup.get_chroma_client).
_session_meta: dict[str, dict] = {}


class ChatRequest(BaseModel):
    query: str


def _serialize_report(report) -> dict:
    return {
        "query": report.query,
        "action": report.action,
        "answer": report.answer,
        "relevance": [
            {
                "chunk_id": r.chunk.id,
                "text": r.chunk.text,
                "probability": r.probability,
                "kept": r.kept,
            }
            for r in report.relevance
        ],
        "sufficiency": asdict(report.sufficiency),
        "grounding": (
            {
                "grounded": report.grounding.grounded,
                "coverage": report.grounding.coverage,
                "claims": [asdict(c) for c in report.grounding.claims],
            }
            if report.grounding is not None
            else None
        ),
    }


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
    meta = {"filename": file.filename, "chunk_count": len(chunks)}
    _session_meta[x_session_id] = meta
    return meta


@app.post("/api/chat")
def chat(request: ChatRequest, x_session_id: str = Header(..., alias="X-Session-Id")) -> dict:
    query = request.query.strip()
    if not query:
        return {"error": "empty query"}

    collection = get_session_collection(x_session_id)
    if collection is None:
        raise HTTPException(400, "No document uploaded yet for this session")

    chunks = make_retriever(collection)(query)
    report = _guard.run(query, chunks, _upload_generate_fn)
    return _serialize_report(report)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
