"""Interactive REPL for testing rag-guard as an externally-installed package.

This repo exists to answer one question: does rag-guard actually work when
pulled in the way a real user would pull it in (`pip install git+https://...`),
against a real vector store, a real LLM, and a corpus it has never seen
during rag-guard's own development? `corpus.py` is an internal-engineering-
docs simulation for a fictional company (Nimbus) -- deliberately messier
than trivia facts, with overlapping runbooks and a few real gaps, so
relevance/sufficiency/grounding all get exercised, not just the easy case.

For the web UI version of this same pipeline, see app.py.

Usage:
    python chat.py
    > How do I restart the ingest service?
    > what's nimbus's data retention policy for raw events?
    > exit
"""

from __future__ import annotations

from pipeline_setup import setup


def _print_trace(report) -> None:
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
    guard, retrieve, generate_fn, _upload_generate_fn, _rewrite_query_fn, backend_info, _clarify_fn = (
        setup()
    )

    print("rag-guard REPL against the Nimbus docs corpus")
    print(
        f"generate_fn backend: {backend_info['generate_fn']}  |  "
        f"decision model backend: {backend_info['decision_model']}"
    )
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

        _print_trace(report)
        print(f"\n{report.answer}\n")


if __name__ == "__main__":
    main()
