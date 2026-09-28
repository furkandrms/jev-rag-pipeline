"""FastAPI web UI for the same rag-guard pipeline chat.py runs as a REPL.

Serves a single-page chat UI (static/) and one JSON API the frontend calls:
POST /api/chat -- runs RagGuard.run() and returns the full decision trace
(relevance per chunk, sufficiency, grounding per claim) alongside the
answer, so the UI can show not just *what* rag-guard answered but *why*.

Run locally:
    uvicorn app:app --reload

The vector store + backend selection happens once at import time (module-
level `_STATE`), not per-request -- rebuilding the Chroma collection or
re-resolving API clients on every chat message would be wasteful and, for
the in-memory Chroma client here, unnecessary.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline_setup import setup

app = FastAPI(title="rag-guard chat")

_guard, _retrieve, _generate_fn, _backend_info = setup()


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


@app.post("/api/chat")
def chat(request: ChatRequest) -> dict:
    query = request.query.strip()
    if not query:
        return {"error": "empty query"}

    chunks = _retrieve(query)
    report = _guard.run(query, chunks, _generate_fn)
    return _serialize_report(report)


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("static/index.html")
