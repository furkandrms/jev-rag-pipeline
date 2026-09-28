"""FastAPI web UI for the same rag-guard pipeline chat.py runs as a REPL.

Serves a single-page chat UI (static/) and the JSON API the frontend calls:

  POST /api/upload -- chunk an uploaded .txt/.md/.pdf into a session-scoped
                       Chroma collection (one document per session; a new
                       upload replaces the previous one for that session).
  POST /api/chat    -- runs RagGuard.run() against either the demo Nimbus
                       corpus or the caller's uploaded document (mode field)
                       and returns the full decision trace (relevance per
                       chunk, sufficiency, grounding per claim) alongside
                       the answer, so the UI can show not just *what*
                       rag-guard answered but *why*.
  GET  /api/info    -- active backends + RagGuard's thresholds.

Sessions are scoped by an `X-Session-Id` header the frontend mints itself
(crypto.randomUUID(), persisted in localStorage) -- there's no login, just
enough identity to keep one browser's uploaded document out of another's.

Run locally:
    uvicorn app:app --reload

The vector store + backend selection happens once at import time (module-
level globals below), not per-request -- rebuilding Chroma collections or
re-resolving API clients on every chat message would be wasteful.
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

app = FastAPI(title="rag-guard chat")

_guard, _retrieve, _generate_fn, _upload_generate_fn, _backend_info = setup()


class ChatRequest(BaseModel):
    query: str
    mode: str = "demo"  # "demo" (Nimbus corpus) or "upload" (this session's document)


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
    return {"filename": file.filename, "chunk_count": len(chunks)}


@app.post("/api/chat")
def chat(request: ChatRequest, x_session_id: str | None = Header(None, alias="X-Session-Id")) -> dict:
    query = request.query.strip()
    if not query:
        return {"error": "empty query"}

    if request.mode == "upload":
        if not x_session_id:
            raise HTTPException(400, "X-Session-Id header required for mode=upload")
        collection = get_session_collection(x_session_id)
        if collection is None:
            raise HTTPException(400, "No document uploaded yet for this session")
        chunks = make_retriever(collection)(query)
        generate_fn = _upload_generate_fn
    else:
        chunks = _retrieve(query)
        generate_fn = _generate_fn

    report = _guard.run(query, chunks, generate_fn)
    return _serialize_report(report)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
