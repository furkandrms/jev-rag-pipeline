"""Optional decision-model backend: convaiinnovations/laya, a self-hosted,
open-weight (Apache 2.0) typed-decision model -- an alternative to the
OpenAI/Anthropic-backed DecisionModel this app defaults to. Opt-in per
request via ChatRequest.use_laya (see app.py); pipeline_setup.get_laya_guard()
is what actually wires one of these into a RagGuard.

Tradeoffs, measured locally before wiring this in:
- Needs torch>=2.5 + transformers + the ~800MB-1.6GB `laya` checkpoint,
  downloaded on first use and cached for the life of the process.
- On a GPU (CUDA) or Apple's MPS, one decision call measured ~50-60ms,
  comparable to a native typed-decision API. On CPU -- which is what
  Cloud Run has, no GPU/MPS -- the same call measured ~2.2s. rag-guard's
  pipeline makes many calls per query (one per retrieved chunk for
  relevance, one per claim for grounding, plus intent/clarify/
  sufficiency/caveat), so a single Laya-backed query on Cloud Run can
  take tens of seconds to a few minutes. That's a known, accepted
  tradeoff for this opt-in toggle, not a bug to fix here.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping

from rag_guard.decision_model import DecisionModel

_MODEL_ID = "convaiinnovations/laya"


def _pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LayaDecisionModel(DecisionModel):
    """Decision model backed by a self-hosted, open-weight Laya checkpoint.

    Mirrors JevDecisionModel's shape (rag_guard.decision_model): Laya's own
    `noul`/`choice` question types map directly onto this interface, with
    no prompt-building or JSON-parsing layer needed, same as the real Jev.
    """

    # Process-wide: the loaded checkpoint and its warm-up cost are shared
    # across every LayaDecisionModel instance (and therefore every
    # request), not reloaded per RagGuard.
    _agent = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if LayaDecisionModel._agent is None:
            with LayaDecisionModel._lock:
                if LayaDecisionModel._agent is None:
                    LayaDecisionModel._agent = self._load()

    @staticmethod
    def _load():
        import laya

        agent = laya.load(_MODEL_ID, device=_pick_device())
        # One throwaway call to pay the first-call compile/dispatch cost
        # up front -- measured ~40x slower on an uncompiled first call on
        # Apple's MPS -- rather than making whichever real request happens
        # to arrive first eat that penalty.
        agent.predict(
            "warmup", {"result": {"type": "noul", "instructions": "Is this a warmup call?"}}
        )
        return agent

    def noul(self, state: str, question: str) -> float:
        result = LayaDecisionModel._agent.predict(
            state, {"result": {"type": "noul", "instructions": question}}
        )
        return float(result["answers"]["result"]["noul"])

    def choice(
        self, state: str, question: str, options: Mapping[str, str]
    ) -> tuple[str, dict[str, float]]:
        result = LayaDecisionModel._agent.predict(
            state, {"result": {"type": "choice", "instructions": question, "criteria": dict(options)}}
        )
        answer = result["answers"]["result"]
        return answer["choice"], dict(answer["probabilities"])
