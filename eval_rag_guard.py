"""Standalone test harness: does rag-guard actually reduce hallucination risk?

This is not a demo -- it's the test this whole repo exists to run. It loads
two unrelated real documents, asks each a mix of in-scope, out-of-scope, and
cross-document questions through the *real* pipeline (real embeddings, real
generation model, real Jev/LLM decision model -- whatever `pipeline_setup.setup()`
wires up), and checks that rag-guard's three stages behave the way they're
supposed to:

  - relevance should drop chunks that don't actually help the question
  - sufficiency should refuse to generate when the kept context can't
    support a confident answer (this is the main hallucination-prevention
    lever) instead of guessing
  - grounding should catch/flag an answer that drifts beyond what the
    context actually supports

Run:
    python eval_rag_guard.py

Needs the same .env as app.py (OPENAI_API_KEY and/or ANTHROPIC_API_KEY,
optionally TYPESAFE_API_KEY) and the two PDFs already in this repo:
JEV_RAG_Bilgi_Dokumani.pdf and vw_bakim_kapsamlari_2025.pdf.
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

from pipeline_setup import chunk_text, extract_text, get_chroma_client, make_retriever, setup
from rag_guard.types import GuardReport

ROOT = pathlib.Path(__file__).parent


@dataclass
class Case:
    query: str
    expect: str  # "answered" | "insufficient_context" | "either" (don't assert action)
    label: str  # short tag for the report, e.g. "in-scope", "out-of-scope", "cross-doc"


@dataclass
class Doc:
    name: str
    filename: str
    collection_name: str
    cases: list[Case]


DOCS = [
    Doc(
        name="JEV/RAG knowledge document",
        filename="JEV_RAG_Bilgi_Dokumani.pdf",
        collection_name="eval_jev_doc",
        cases=[
            Case("What is the purpose of the JEV 2.3 decision mechanism?", "answered", "in-scope"),
            Case(
                "At which stages of the RAG pipeline is JEV used?",
                "answered",
                "in-scope",
            ),
            Case(
                "According to this document, how often should a Volkswagen EV be serviced?",
                "insufficient_context",
                "cross-doc",
            ),
            Case(
                "According to this document, what will the weather be like tomorrow?",
                "insufficient_context",
                "out-of-scope",
            ),
        ],
    ),
    Doc(
        name="VW maintenance scope",
        filename="vw_bakim_kapsamlari_2025.pdf",
        collection_name="eval_vw_doc",
        cases=[
            # The doc gives different brake-pad-inspection intervals depending
            # on powertrain (electric vs. internal combustion) -- with the new
            # clarify stage, this is correctly flagged as needing the user to
            # specify which vehicle type they mean, rather than guessing.
            Case("How often is the brake pad inspection performed?", "clarify", "in-scope"),
            Case("Is the windshield washer fluid level check part of the maintenance scope?", "answered", "in-scope"),
            Case(
                "According to this document, what are the threshold values of the JEV 2.3 decision mechanism?",
                "insufficient_context",
                "cross-doc",
            ),
            Case(
                "What is the price of an oil change for this vehicle?",
                "insufficient_context",
                "out-of-scope",
            ),
        ],
    ),
]


def load_collection(doc: Doc):
    client = get_chroma_client()
    try:
        client.delete_collection(doc.collection_name)
    except Exception:
        pass
    collection = client.create_collection(doc.collection_name)
    text = extract_text(doc.filename, (ROOT / doc.filename).read_bytes())
    chunks = chunk_text(text)
    collection.add(ids=[f"{doc.filename}-{i}" for i in range(len(chunks))], documents=chunks)
    print(f"  loaded {doc.filename}: {len(chunks)} chunks")
    return collection


def report_line(report: GuardReport, case: Case) -> tuple[bool, str]:
    kept = sum(1 for r in report.relevance if r.kept)
    ok = case.expect == "either" or report.action in (
        case.expect,
        "ungrounded_answer_flagged" if case.expect == "answered" else case.expect,
    )
    status = "PASS" if ok else "FAIL"
    detail = (
        f"[{status}] ({case.label}) action={report.action} "
        f"kept={kept}/{len(report.relevance)} chunks "
        f"sufficiency_p={report.sufficiency.probability:.2f}"
    )
    if report.grounding is not None:
        detail += f" grounding_coverage={report.grounding.coverage:.2f}"
    detail += f"\n         Q: {case.query}"
    detail += f"\n         A: {(report.answer or '')[:160]}"
    return ok, detail


def main() -> int:
    print("Setting up pipeline (real retrieval + generation + decision model)...")
    guard, _, _, make_upload_generate_fn, _, backend_info, _ = setup()
    generate_fn = make_upload_generate_fn()
    print(f"  backend: {backend_info}\n")

    total = 0
    passed = 0
    for doc in DOCS:
        print(f"=== {doc.name} ({doc.filename}) ===")
        collection = load_collection(doc)
        retrieve = make_retriever(collection)
        for case in doc.cases:
            chunks = retrieve(case.query)
            report = guard.run(case.query, chunks, generate_fn)
            ok, detail = report_line(report, case)
            total += 1
            passed += int(ok)
            print(detail)
        print()

    print(f"{passed}/{total} cases behaved as expected.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
