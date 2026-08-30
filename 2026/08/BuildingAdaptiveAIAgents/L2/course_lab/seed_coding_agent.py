"""Synthetic CODING-AGENT memory domain.

A coding agent's memory has real structure: files import each other, functions
call each other, files are co-edited in the same commit, and tool-call traces
(grep -> read -> edit) co-retrieve the same memories. This seeder produces that
structure so Module 3 has a graph worth auditing and retrieving over.

Deterministic, seeded, CPU/offline — mirrors seed_supplychain.py.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_lab.agent_memory import AgentMemory

_FILES = ["routing.py", "geo.py", "auth.py", "db.py", "api.py", "cache.py"]
_SYMBOLS = {
    "routing.py": ["route_lookup", "plan_route"],
    "geo.py": ["haversine", "bbox"],
    "auth.py": ["login", "verify_token"],
    "db.py": ["query_voyage", "connect"],
    "api.py": ["handle_request", "serialize"],
    "cache.py": ["get_cached", "evict"],
}
_FILE_EDGES = [
    ("routing.py", "geo.py", "import"),
    ("routing.py", "db.py", "import"),
    ("api.py", "routing.py", "import"),
    ("api.py", "auth.py", "import"),
    ("api.py", "cache.py", "import"),
    ("db.py", "cache.py", "import"),
]
_CALL_EDGES = [
    ("route_lookup", "haversine", "call"),
    ("route_lookup", "query_voyage", "call"),
    ("handle_request", "route_lookup", "call"),
    ("handle_request", "login", "call"),
]
_EDIT_EVENTS = [
    "fixed null-deref in {sym} ({f})",
    "added test for {sym} ({f})",
    "refactored {sym} in {f}",
    "renamed {sym} in {f}",
]
_SKILL_NAMES = ["locate_symbol", "trace_callers", "summarize_diff", "find_tests", "explain_function"]


def _symbol_fact(f: str, sym: str, i: int) -> str:
    return f"Coding memory {i}: function {sym} in {f} is part of the service codebase."


def structural_edges(*, seed: int = 42) -> list[tuple[str, str, str]]:
    """Return deterministic structural edges as (src_id, dst_id, kind).

    Node ids are 'file:routing.py' for files and 'sym:route_lookup' for symbols.
    Co-edit edges are derived from a deterministic commit grouping.
    """
    rng = random.Random(seed)
    edges: list[tuple[str, str, str]] = []
    for src, dst, kind in _FILE_EDGES:
        edges.append((f"file:{src}", f"file:{dst}", kind))
    for src, dst, kind in _CALL_EDGES:
        edges.append((f"sym:{src}", f"sym:{dst}", kind))
    files = list(_FILES)
    rng.shuffle(files)
    for a, b in zip(files[0::2], files[1::2]):
        edges.append((f"file:{a}", f"file:{b}", "co_edit"))
    edges.append(("sym:route_lookup#dup", "sym:haversine", "call"))
    return edges


def memory_nodes() -> list[dict]:
    """Canonical retrievable nodes shared by the graph, audit, corpus, and eval.

    Each node id matches the `file:`/`sym:` namespace used by structural_edges, so
    audit orphan-detection, PPR, and eval positives all live in ONE id space.
    """
    nodes = []
    for f in _FILES:
        nodes.append({"id": f"file:{f}", "text": f"File {f} in the service codebase."})
    for f in _FILES:
        for s in _SYMBOLS[f]:
            nodes.append({"id": f"sym:{s}",
                          "text": f"function {s} in {f} is part of the service codebase."})
    # one deliberate near-duplicate of route_lookup so the audit has a real
    # dedup target and Act-2 restructure visibly improves the scorecard.
    rl_text = next(nd["text"] for nd in nodes if nd["id"] == "sym:route_lookup")
    nodes.append({"id": "sym:route_lookup#dup", "text": rl_text})
    return nodes


def eval_tasks() -> list[dict]:
    """Labeled (query, positive, type) tasks. Multi-hop positives are the CALLEE
    of a call edge whose CALLER is the lexical hit; the callee is NOT lexically
    in the query, so flat/lexical retrieval cannot reach it — only graph
    propagation can. Lexical tasks are direct hits where structure does no work."""
    return [
        {"query": "When route_lookup runs, which helper computes great-circle distance?",
         "positive": "sym:haversine", "type": "multi_hop"},
        {"query": "route_lookup needs the database layer; which function does it invoke?",
         "positive": "sym:query_voyage", "type": "multi_hop"},
        {"query": "handle_request dispatches an incoming call to which downstream symbol?",
         "positive": "sym:route_lookup", "type": "multi_hop"},
        {"query": "handle_request also performs a credential check via which symbol?",
         "positive": "sym:login", "type": "multi_hop"},
        {"query": "Where is route_lookup defined?",
         "positive": "sym:route_lookup", "type": "lexical"},
        {"query": "Which symbol handles voyage DB queries?",
         "positive": "sym:query_voyage", "type": "lexical"},
    ]


def vocab() -> dict:
    return {"files": _FILES, "symbols": _SYMBOLS, "skills": _SKILL_NAMES}


def write_seed_jsonl(out_path: "Path", *, n_memories: int = 50, seed: int = 42) -> int:
    """Write synthetic coding-agent memory units to a JSONL file (no DB required).

    Mirrors seed_supplychain.write_seed_jsonl: each row carries a cognitive
    ``kind`` in {semantic, episodic, procedural} so Module 1's classifier has
    coding-domain data. Returns count written.
    """
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat_syms = [(f, s) for f in _FILES for s in _SYMBOLS[f]]
    n_each = n_memories // 3
    remainder = n_memories - n_each * 3

    rows: list[dict] = []
    # semantic: durable facts about symbols/files
    for i in range(n_each):
        f, s = flat_syms[i % len(flat_syms)]
        rows.append({
            "kind": "semantic",
            "content": f"function {s} in {f} is part of the service codebase.",
            "memory_id": f"sem-{i:05d}",
            "agent_id": "coding_agent",
        })
    # episodic: past edit events
    for i in range(n_each):
        f, s = rng.choice(flat_syms)
        event = rng.choice(_EDIT_EVENTS).format(sym=s, f=f)
        rows.append({
            "kind": "episodic",
            "content": f"Edit event: {event}.",
            "memory_id": f"epi-{i:05d}",
            "agent_id": "coding_agent",
        })
    # procedural: skill recipes
    for i in range(n_each + remainder):
        name = _SKILL_NAMES[i % len(_SKILL_NAMES)]
        rows.append({
            "kind": "procedural",
            "content": f"Procedure: {name} via grep symbol -> read file -> summarize.",
            "memory_id": f"pro-{i:05d}",
            "agent_id": "coding_agent",
        })

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


def write_eval_qa_jsonl(out_path: "Path", *, n: int = 20, seed: int = 42) -> int:
    """Write held-out coding-agent Q&A pairs (question, answer_substring, context).

    Mirrors seed_supplychain.write_eval_qa_jsonl. Returns count written.
    Deterministic by index (the ``seed`` arg is accepted for signature parity).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    flat_syms = [(f, s) for f in _FILES for s in _SYMBOLS[f]]
    rows: list[dict] = []
    for i in range(n):
        f, s = flat_syms[i % len(flat_syms)]
        rows.append({
            "id": f"qa-{i:05d}",
            "question": f"Where is the function {s} defined?",
            "answer_substring": f,
            "context": f"function {s} in {f} is part of the service codebase.",
        })

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


def seed_into(mem: "AgentMemory", *, seed: int = 42,
              n_threads: int = 5, n_messages_per_thread: int = 6,
              n_durable_memories: int = 30, n_skills: int = 5, **_kw) -> dict:
    """Populate AgentMemory with coding-agent memory + record co-retrieval traces."""
    rng = random.Random(seed)
    threads = messages = durables = skills = 0

    seed_thread = mem.create_thread(user_id="coding-agent")
    threads += 1

    flat_syms = [(f, s) for f in _FILES for s in _SYMBOLS[f]]
    for i in range(min(n_durable_memories, len(flat_syms) * 3)):
        f, s = flat_syms[i % len(flat_syms)]
        seed_thread.add_memory(_symbol_fact(f, s, i))
        durables += 1

    for t_idx in range(n_threads):
        thread = mem.create_thread(user_id=f"coding-user-{t_idx}")
        threads += 1
        msgs = []
        for m_idx in range(n_messages_per_thread):
            f, s = rng.choice(flat_syms)
            text = rng.choice(_EDIT_EVENTS).format(sym=s, f=f)
            role = "user" if m_idx % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": text})
        thread.add_messages(msgs)
        messages += len(msgs)

    for i in range(n_skills):
        mem.write_skill(
            name=_SKILL_NAMES[i % len(_SKILL_NAMES)],
            description=f"Recipe for {_SKILL_NAMES[i % len(_SKILL_NAMES)]} in a coding agent.",
            recipe_steps=["grep symbol", "read file", "summarize"],
            provenance=[f"coding-seed-{i}"],
        )
        skills += 1

    edges = structural_edges(seed=seed)
    for i, (src, dst, _kind) in enumerate(edges):
        candidates = [{"id": src, "text": src}, {"id": dst, "text": dst}]
        mem.record_retrieval_trace(
            query=f"trace {i}: navigate from {src} to {dst}",
            candidates=candidates, chosen=[src, dst], outcome="success",
        )

    return {"threads": threads, "messages": messages,
            "durable_memories": durables, "skills": skills, "edges": len(edges)}
