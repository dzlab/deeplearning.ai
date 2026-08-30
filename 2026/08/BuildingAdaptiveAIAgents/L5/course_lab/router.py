"""Traffic router for Module 4 — picks between {base, specialised}.

Per spec-v2 §2.4, the ``{base, specialised}`` labels mean
``{stock unsloth/gemma-3n-E4B-it, QLoRA-tuned unsloth/gemma-3n-E4B-it}``.
The router itself is model-agnostic — it only knows the two route names — but
the convention across Module 4's notebook, configs, and README assumes that
mapping.

Smoke / default path: rule-based on task string keywords (domain heuristic on
``voyage``, ``shipment``, ``vessel`` lexicon). Production callers can plug a
trained classifier in by subclassing or replacing the ``route`` method
(``ClassificationRouter`` is the project shorthand for that pattern).

The router also owns the regression-gate decision: if a fresh fine-tune does
not beat a configurable threshold over baseline, traffic stays on the base
model.
"""
from __future__ import annotations

import json
from pathlib import Path

from course_lab import paths


# Domain-specific keywords that should route traffic to the specialised
# (fine-tuned) model.  These mirror the SUPPLYCHAIN vocabulary used by
# ``course_lab.seed_supplychain`` so the router can be exercised end-to-end
# in smoke without external data.
_SPECIALISED_KEYWORDS: tuple[str, ...] = (
    "voyage",
    "vessel",
    "shipment",
    "cargo",
    "manifest",
    "customs",
    "port",
    "supplychain",
    "rotterdam",
    "singapore",
    "shanghai",
)


class Router:
    """Classification-head router over {base, specialised}.

    The smoke implementation is a hash/keyword rule: tasks containing any
    SUPPLYCHAIN-domain keyword route to ``specialised``, otherwise ``base``.
    The full implementation (future work) trains a small classifier on
    labeled task strings using the same return contract.

    Decisions are written to ``paths.router_decisions_jsonl()`` when
    :meth:`record_decision` is called.
    """

    BASE = "base"
    SPECIALISED = "specialised"

    def __init__(self, *, regression_threshold: float = 0.7,
                 keywords: tuple[str, ...] | None = None,
                 decisions_path: Path | None = None) -> None:
        self.regression_threshold = float(regression_threshold)
        self._keywords = tuple(k.lower() for k in (keywords or _SPECIALISED_KEYWORDS))
        self._decisions_path = decisions_path or paths.router_decisions_jsonl()
        self._decisions: list[dict] = []

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, task: str) -> str:
        """Return ``"base"`` or ``"specialised"`` for the given task string."""
        low = (task or "").lower()
        for kw in self._keywords:
            if kw in low:
                return self.SPECIALISED
        return self.BASE

    # ------------------------------------------------------------------
    # Regression gate
    # ------------------------------------------------------------------

    def passes_regression_gate(self, eval_results: dict) -> bool:
        """Return True iff the specialised model beats the regression threshold.

        Expects ``eval_results`` to contain ``baseline_acc`` and either
        ``qlora_acc`` or ``tuned_acc``.  Threshold is the ratio
        ``specialised_acc / max(baseline_acc, eps)`` >= ``regression_threshold``.
        Missing keys default to 0.0, which fails the gate by construction.
        """
        baseline = float(eval_results.get("baseline_acc", 0.0))
        specialised = float(
            eval_results.get("qlora_acc", eval_results.get("tuned_acc", 0.0))
        )
        # Anything that beats an identically-zero baseline passes the gate
        # outright; otherwise compute the ratio.
        if baseline <= 0.0:
            return specialised > 0.0
        return (specialised / baseline) >= self.regression_threshold

    def passes_regression_gate_v2(
        self,
        eval_results: dict,
        *,
        behavior_rate_drop_min: float = 0.5,
        kept_capability_min: float = 0.85,
        arm: str = "v2",
    ) -> bool:
        """Behavior-removal-aware regression gate (Module 4 v2 mechanism).

        Returns True iff:
          - behavior_rate_drop[arm] >= behavior_rate_drop_min
          - kept_capability_rate[arm] >= kept_capability_min
        Both signals must clear their thresholds. Missing keys default to 0.0
        which fails the gate by construction.
        """
        drop = float(eval_results.get("behavior_rate_drop", {}).get(arm, 0.0))
        kept = float(eval_results.get("kept_capability_rate", {}).get(arm, 0.0))
        return drop >= behavior_rate_drop_min and kept >= kept_capability_min

    # ------------------------------------------------------------------
    # Decision log
    # ------------------------------------------------------------------

    def record_decision(self, task: str, route: str,
                        outcome: str | None = None) -> None:
        """Append a routing decision to ``paths.router_decisions_jsonl()``."""
        decision = {"task": task, "route": route, "outcome": outcome}
        self._decisions.append(decision)
        self._decisions_path.parent.mkdir(parents=True, exist_ok=True)
        with self._decisions_path.open("a") as f:
            f.write(json.dumps(decision) + "\n")

    def get_decisions(self) -> list[dict]:
        """Return the in-memory list of decisions recorded in this session."""
        return list(self._decisions)


class AdapterRouter:
    """Route a user query to ONE plug-n-play LoRA adapter over a frozen base.

    Each adapter is a small loadable patch on top of the SAME frozen base model
    (e.g. ``superpolitegemma``). This router inspects the incoming query and
    decides which adapter should answer it — or, if none fits, lets the frozen
    **base** respond unchanged. That is the plug-n-play story: after you have
    fine-tuned one or more behavior adapters, a router picks the right one per
    task instead of loading a different full model.

    ``adapters`` maps ``adapter_name -> (cue, cue, ...)``: a query whose text
    contains any of an adapter's cue substrings routes to that adapter; the
    first registered adapter with a hit wins, and a query matching none falls
    through to ``fallback`` (the frozen base by default). This is the rule-based
    smoke implementation — a production router swaps in a trained query
    classifier behind the same :meth:`route` contract, exactly like
    :class:`Router` above.
    """

    BASE = "base"

    def __init__(self, adapters: dict[str, tuple[str, ...]], *,
                 fallback: str = BASE) -> None:
        self._adapters = {
            name: tuple(c.lower() for c in cues)
            for name, cues in adapters.items()
        }
        self._fallback = fallback

    def route(self, query: str) -> str:
        """Return the adapter name to answer ``query`` (or ``fallback``)."""
        low = (query or "").lower()
        for name, cues in self._adapters.items():
            if any(cue in low for cue in cues):
                return name
        return self._fallback


__all__ = ["Router", "AdapterRouter"]
