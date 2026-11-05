"""Structured console logging for the pipeline.

uvicorn's own log only tells you an HTTP request happened and what status
code it returned -- it says nothing about what actually went in or out of
rag-guard: which chunks got retrieved, kept, or dropped; what probability
each decision actually came back with; how many real tokens a generation
call used; whether the answer ended up grounded. This module is that
layer, printed to the same console uvicorn is already logging to.

Default level (INFO) shows one line per stage: counts, probabilities,
token usage, and real elapsed time -- enough to answer "what did the
pipeline actually do for this query" at a glance. Set LOG_LEVEL=DEBUG for
a line per chunk/claim (full relevance snippets, full claim text) when
you need to see exactly what went into a judgment, not just the outcome.

    LOG_LEVEL=DEBUG uvicorn app:app --reload
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("ragpipeline")
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s [%(session)s] %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(_handler)
    # Don't also propagate to the root logger -- uvicorn configures that
    # one itself, and without this every line would print twice.
    logger.propagate = False


def stage_log(session_id: str, stage: str, message: str, level: int = logging.INFO) -> None:
    """One line: timestamp, level, an 8-char session prefix, a left-aligned
    stage name, and the message. `stage` is a fixed vocabulary (query,
    retrieval, relevance, sufficiency, generation, grounding, done,
    cache_hit, upload, remove, error) -- kept short and consistent so the
    log reads as a column, not a wall of prose.
    """
    logger.log(level, f"{stage:<11} {message}", extra={"session": session_id[:8]})
