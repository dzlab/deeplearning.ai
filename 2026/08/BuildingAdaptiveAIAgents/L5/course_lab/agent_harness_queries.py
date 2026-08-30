"""Generate indirect-intent search queries for the agent-harness corpus.

The embedding arm needs (query -> gold chunk) pairs where the query describes a
function's PURPOSE without using its symbol/docstring tokens — otherwise the base
encoder wins by lexical match and there is no fine-tune headroom. The real
agent-harness docstrings are descriptive, so these queries MUST be generated (not
templated from the docstring) and anti-leakage-checked.

Live Grok (``oci/xai.grok-4.3``) writes 3 queries per chunk; results are cached
and committed to ``course_lab/data/agent_harness_queries.json`` (chunk_id -> [q]).
Offline replay reads the committed cache and never calls Grok — same convention
as Module 2's fixture and arm B's chat corpus.
"""
from __future__ import annotations

import json
import os
import re

from course_lab import oci_client, paths
from course_lab.agent_harness_corpus import load_corpus
from course_lab.structural_retrieval import identifier_tokens

_MODEL_ID = "oci/xai.grok-4.3"


def _blocked_tokens(chunk: dict) -> set[str]:
    """Tokens a query must avoid: the symbol name + docstring + signature words."""
    text = f"{chunk['symbol']} {chunk['docstring']} {chunk['signature']}"
    toks = identifier_tokens(text)
    # keep only contentful tokens (len>=4) so we don't block 'the'/'a'/'id'
    return {t for t in toks if len(t) >= 4}


def _leaks(query: str, blocked: set[str]) -> bool:
    qtoks = identifier_tokens(query)
    return bool(qtoks & blocked)


def _prompt(chunk: dict) -> list[dict]:
    sys = (
        "You write SHORT natural-language code-search queries. Given a function's "
        "purpose, output 3 distinct queries a developer might type to FIND it, each "
        "describing WHAT it does in plain words. STRICT RULE: never use the "
        "function's name or any distinctive word from its docstring/signature — "
        "describe the intent, not the identifiers. Return STRICT JSON: "
        '{"queries": ["...", "...", "..."]}'
    )
    user = (
        f"Function: {chunk['signature']}\n"
        f"Purpose: {chunk['docstring']}\n"
        f"What it does: {chunk['body']}\n\n"
        "Write 3 indirect-intent search queries (plain English, no identifiers)."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _parse(raw: str) -> list[str]:
    try:
        obj = json.loads(raw)
        qs = obj.get("queries", [])
        return [str(q).strip() for q in qs if str(q).strip()][:3]
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []


def _fallback_query(chunk: dict) -> str:
    """Deterministic last-resort query if Grok output leaks or is malformed.

    Built from the AST body (the calls it makes) — describes behaviour without
    the symbol/docstring tokens. Used only to keep every chunk covered.
    """
    body = chunk["body"].replace("calls ", "")
    return f"which function {('uses ' + body) if body != 'no internal calls' else 'in ' + chunk['file']}"[:160]


def generate(*, live: bool, cache: dict | None = None) -> dict:
    """chunk_id -> [3 queries]. live=True calls Grok; else replays the cache."""
    chunks = load_corpus()
    if not live:
        if cache is None:
            cache = _load_cache()
        out = {}
        for c in chunks:
            out[c["id"]] = cache.get(c["id"]) or [_fallback_query(c)]
        return out

    recorder = None
    if os.environ.get("DLAICL_OCI_LIVE"):
        from course_lab import llm_cache
        recorder = llm_cache.LLMRecorder(paths.m2_llm_fixture_json())

    out: dict[str, list[str]] = {}
    for c in chunks:
        blocked = _blocked_tokens(c)
        messages = _prompt(c)
        raw = oci_client.get_chat_completion(
            messages, _MODEL_ID, temperature=0.0,
            response_format={"type": "json_object"})
        if recorder is not None:
            recorder.record(messages, _MODEL_ID, raw, temperature=0.0,
                            response_format={"type": "json_object"})
        qs = [q for q in _parse(raw) if not _leaks(q, blocked)]
        out[c["id"]] = qs or [_fallback_query(c)]
    if recorder is not None:
        recorder.flush()
    return out


def _load_cache() -> dict:
    p = paths.agent_harness_queries_json()
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; regenerate with `DLAICL_OCI_LIVE=1 uv run python -c "
            "\"from course_lab.agent_harness_queries import write_queries; "
            "write_queries(live=True)\"`")
    return json.loads(p.read_text(encoding="utf-8"))


def write_queries(*, live: bool = False) -> dict:
    out = generate(live=live)
    dest = paths.agent_harness_queries_json()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # anti-leakage report
    chunks = {c["id"]: c for c in load_corpus()}
    leaked = sum(1 for cid, qs in out.items()
                 for q in qs if _leaks(q, _blocked_tokens(chunks[cid])))
    return {"chunks": len(out), "total_queries": sum(len(v) for v in out.values()),
            "leaked_queries": leaked}


def load_queries() -> dict:
    """chunk_id -> [queries], replayed from the committed cache (no Grok)."""
    return generate(live=False)


if __name__ == "__main__":  # pragma: no cover
    live = bool(os.environ.get("DLAICL_OCI_LIVE"))
    print(write_queries(live=live))
