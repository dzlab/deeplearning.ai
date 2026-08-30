"""Synthetic coding-agent retrieval sessions with auto-graded relevance.

Each trace targets one gold chunk and carries a multi-round NL query sequence
(the agent narrowing in), a candidate set, and deterministic graded relevance:

    gold chunk                                  -> 3
    same-file or direct call/import neighbor    -> 2
    cross-file co-edit OR 2-hop call neighbor   -> 1
    unrelated                                   -> 0

This is the synthetic stand-in for Cursor's "feed traces to an LLM that ranks
what should have been retrieved at each step": here the rank is derived
deterministically from the codebase graph so the first corpus is reproducible
and needs no live LLM (instructor decision).

Queries come from the committed indirect-intent fixture
(``course_lab/data/scm_queries.json``), authored to describe each function's
purpose WITHOUT its symbol/docstring tokens — so the base encoder cannot win by
lexical match (a query that names the symbol makes retrieval trivial and leaves
no fine-tune headroom; the fixture is anti-leakage-checked).
"""
from __future__ import annotations

import json
import random

from course_lab import paths
from course_lab.scm_codebase import Chunk

# Lazy-loaded committed query fixture: chunk_id -> [indirect intent queries].
_QUERIES: dict[str, list[str]] | None = None


def _load_queries() -> dict[str, list[str]]:
    global _QUERIES
    if _QUERIES is None:
        fixture = paths.scm_queries_json()
        if not fixture.exists():
            raise FileNotFoundError(
                f"Missing intent-query fixture at {fixture}. It is committed "
                "alongside the SCM codebase; regenerate the codebase + queries "
                "if it is absent."
            )
        _QUERIES = json.loads(fixture.read_text())
    return _QUERIES


def _nl_queries(chunk: Chunk, rng: random.Random,
                queries: dict | None = None) -> list[str]:
    """The indirect intent queries for ``chunk``.

    Holds ~3 queries per chunk presented as the agent's reformulation sequence in
    a deterministically-shuffled order. ``queries`` (chunk_id -> [str]) lets a
    caller inject a different corpus's query fixture (e.g. agent-harness); when
    omitted it falls back to the committed SCM fixture.
    """
    src = queries if queries is not None else _load_queries()
    qs = list(src[chunk["id"]])
    rng.shuffle(qs)
    return qs


def _grade(cand: Chunk, gold: Chunk, two_hop: set[str]) -> int:
    """Graded relevance of ``cand`` for a session targeting ``gold``.

    ``two_hop`` is the set of chunk ids reachable from gold by two import hops
    (computed once per gold in :func:`build_traces`).
    """
    if cand["id"] == gold["id"]:
        return 3
    same_file = cand["file"] == gold["file"]
    direct_call = cand["id"] in gold["imports"] or gold["id"] in cand["imports"]
    if same_file or direct_call:
        return 2
    cross_file_coedit = (
        cand["id"] in gold["co_edits"] and cand["file"] != gold["file"]
    )
    if cross_file_coedit or cand["id"] in two_hop:
        return 1
    return 0


def build_traces(
    chunks: list[Chunk],
    *,
    n_traces: int,
    candidates_per_trace: int,
    seed: int = 42,
    queries: dict | None = None,
) -> list[dict]:
    rng = random.Random(seed)
    by_id = {c["id"]: c for c in chunks}
    all_ids = [c["id"] for c in chunks]

    def two_hop_of(gold: Chunk) -> set[str]:
        """Chunk ids reachable from gold within two import hops (excl. direct)."""
        direct = set(gold["imports"])
        hop2: set[str] = set()
        for mid in direct:
            hop2.update(by_id[mid]["imports"])
        return (hop2 - direct) - {gold["id"]}

    traces: list[dict] = []
    for t in range(n_traces):
        gold = chunks[rng.randrange(len(chunks))]
        two_hop = two_hop_of(gold)
        # candidate set = gold + a deterministic sample of others
        pool = [i for i in all_ids if i != gold["id"]]
        n_other = min(candidates_per_trace - 1, len(pool))
        cand_ids = [gold["id"]] + rng.sample(pool, n_other)
        cand_ids = sorted(cand_ids)
        grades = {cid: _grade(by_id[cid], gold, two_hop) for cid in cand_ids}
        traces.append({
            "query_sequence": _nl_queries(gold, rng, queries),
            "candidates": cand_ids,
            "grades": grades,
            "gold": gold["id"],
        })
    return traces
