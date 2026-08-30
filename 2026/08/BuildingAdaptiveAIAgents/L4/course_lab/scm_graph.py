"""Module 3 — SCM-codebase graph arm.

Builds a co-retrieval/structural graph over the committed supply-chain codebase
fixture (the SAME fixture M4's embedding code-search arm uses) and runs a
naive (lexical) vs graph (lexical-seed -> PPR) retrieval comparison.

Also provides the autonomous-loop primitives: `synthetic_commit` /
`apply_tick` / `demo_tick_closes_loop` simulate one 24h graphify tick where a
new symbol + edges arrive and a previously-failing query becomes answerable —
WITHOUT any gradient step (non-parametric continual learning).

Pure offline: no Oracle, no model. Oracle persistence lives in
`AgentMemory.upsert_graph_*` / `load_graph`; this module is the graph LOGIC.
"""
from __future__ import annotations

from course_lab.memory_graph import MemGraph, build_graph
from course_lab.scm_codebase import load_scm_chunks
from course_lab.structural_retrieval import (
    identifier_tokens, lexical_flat_baseline, ndcg_at_k,
    personalized_pagerank, recall_at_k,
)


def _node_text(chunk: dict) -> str:
    return f"{chunk['symbol']} in {chunk['file']}: {chunk['docstring']}"


def build_scm_graph(
    chunks: list[dict] | None = None,
    *,
    kinds: tuple[str, ...] = ("import", "co_edit"),
) -> tuple[list[dict], MemGraph]:
    """Return (nodes, graph) over the SCM fixture.

    Nodes use the fixture's `id` (``file.py::symbol``). Edges are the fixture's
    import + co_edit relations, fused via the existing `build_graph`. ``kinds``
    selects which edge relations are added: the default ``("import", "co_edit")``
    keeps the full fused graph, while ``kinds=("import",)`` yields the import-only
    graph that wins the multi-hop dependency-QA comparison (the dense co_edit
    edges otherwise drown the sparse import signal).
    """
    chunks = chunks if chunks is not None else load_scm_chunks()
    nodes = [{"id": c["id"], "text": _node_text(c)} for c in chunks]
    edges: list[tuple[str, str, str]] = []
    ids = {c["id"] for c in chunks}
    for c in chunks:
        if "import" in kinds:
            for dst in c["imports"]:
                if dst in ids:
                    edges.append((c["id"], dst, "import"))
        if "co_edit" in kinds:
            for dst in c["co_edits"]:
                if dst in ids and dst != c["id"]:
                    edges.append((c["id"], dst, "co_edit"))
    # build_graph takes (traces, structural_edges); no traces here.
    graph = build_graph([], edges, domain="scm_codebase")
    for nd in nodes:
        graph.add_node(nd["id"])
    return nodes, graph


def multi_hop_dependency_tasks(chunks=None, *, limit=40):
    """Build genuine 2-hop dependency-QA tasks: query names symbol A, the
    positive is C where A imports B imports C, and C's symbol shares NO tokens
    with A's symbol (so C is lexically invisible to a query about A). Each task
    carries an ``anchor`` (= A's id) used to seed PPR. This is the canonical
    multi-hop setup where graph traversal beats lexical/chunking retrieval."""
    chunks = chunks if chunks is not None else load_scm_chunks()
    by_id = {c["id"]: c for c in chunks}
    tasks = []
    for a in chunks:
        for b_id in a["imports"]:
            b = by_id.get(b_id)
            if b is None:
                continue
            for c_id in b["imports"]:
                c = by_id.get(c_id)
                if c is None or c_id == a["id"]:
                    continue
                if identifier_tokens(a["symbol"]) & identifier_tokens(c["symbol"]):
                    continue  # require C lexically distinct from A
                tasks.append({
                    "query": f"in {a['symbol']}, which deep downstream "
                             f"dependency ultimately gets pulled in?",
                    "positive": c_id,
                    "anchor": a["id"],
                })
    return tasks[:limit]


def anchor_ppr_rank(anchor, graph, *, k=5):
    """Rank nodes by Personalized PageRank seeded from a single anchor node.

    Used for multi-hop dependency QA: the query names symbol A (the anchor),
    and PPR propagation across import edges surfaces its deep dependency C —
    which is lexically invisible to the query."""
    ppr = personalized_pagerank(graph, {anchor: 1.0}, alpha=0.85, iters=50)
    return [n for n, _ in sorted(ppr.items(), key=lambda t: (-t[1], t[0]))][:k]


def _lexical_seed(query: str, nodes: list[dict], graph: MemGraph, k: int = 1) -> dict:
    """Single best lexical entry node that has at least one graph neighbour."""
    q = identifier_tokens(query)
    scored = []
    for nd in nodes:
        overlap = len(q & identifier_tokens(nd["id"] + " " + nd["text"]))
        if overlap > 0 and graph.neighbors(nd["id"]):
            scored.append((nd["id"], overlap))
    scored.sort(key=lambda t: (-t[1], t[0]))
    return {nid: 1.0 for nid, _ in scored[:k]}


def structured_rank(query: str, nodes: list[dict], graph: MemGraph,
                    *, k: int = 5) -> list[str]:
    seeds = _lexical_seed(query, nodes, graph, k=1)
    if not seeds:
        return lexical_flat_baseline(query, nodes, k=k)
    ppr = personalized_pagerank(graph, seeds, alpha=0.85, iters=50)
    return [n for n, _ in sorted(ppr.items(), key=lambda t: (-t[1], t[0]))][:k]


def evaluate(nodes: list[dict], graph: MemGraph, tasks: list[dict],
             *, k: int = 5) -> dict:
    """tasks: [{query, positive}]. Returns naive vs graph recall@k / nDCG@k."""
    n = max(len(tasks), 1)
    acc = {"lexical": {"recall": 0.0, "ndcg": 0.0},
           "structured": {"recall": 0.0, "ndcg": 0.0}}
    for t in tasks:
        q, pos = t["query"], t["positive"]
        lex = lexical_flat_baseline(q, nodes, k=k)
        st = structured_rank(q, nodes, graph, k=k)
        acc["lexical"]["recall"] += recall_at_k(lex, pos, k=k)
        acc["lexical"]["ndcg"] += ndcg_at_k(lex, pos, k=k)
        acc["structured"]["recall"] += recall_at_k(st, pos, k=k)
        acc["structured"]["ndcg"] += ndcg_at_k(st, pos, k=k)
    return {name: {"recall_at_k": round(a["recall"] / n, 4),
                   "ndcg_at_k": round(a["ndcg"] / n, 4)}
            for name, a in acc.items()}


# --- Autonomous graphify tick primitives ----------------------------------

_NEW_SYMBOL_ID = "risk/disruption.py::handle_xj7_event"
_CALLER_ID = "planning/eta.py::compute_eta"


def synthetic_commit() -> tuple[dict, list[tuple[str, str, str]]]:
    """A deterministic 'commit since yesterday': one new symbol + its edges.

    The new symbol's id and text are deliberately OPAQUE — ``handle_xj7_event``
    shares ZERO identifier tokens with the demo query, so it is invisible to
    lexical retrieval and can never be its own lexical seed. The only way to
    reach it is through structure: the commit wires it to an existing caller
    (``planning/eta.py::compute_eta``) with both an ``import`` edge (compute_eta
    now calls the new helper) and a bidirectional ``co_edit`` edge (the two were
    touched in the same commit) — the same two edge kinds the fixture itself
    uses. Post-tick, PPR carries mass from the caller across these edges to
    surface the new symbol.
    """
    new_node = {
        "id": _NEW_SYMBOL_ID,
        "text": ("handle_xj7_event in risk/disruption.py: "
                 "emit a contingency signal and select a fallback carrier lane "
                 "when an upstream blockage is flagged"),
    }
    # New helper wired to its existing caller (compute_eta) via the fixture's own
    # two edge kinds: an import (caller -> new) plus a same-commit co_edit pair.
    new_edges = [
        (_CALLER_ID, _NEW_SYMBOL_ID, "import"),
        (_CALLER_ID, _NEW_SYMBOL_ID, "co_edit"),
        (_NEW_SYMBOL_ID, _CALLER_ID, "co_edit"),
    ]
    return new_node, new_edges


def apply_tick(nodes: list[dict], graph: MemGraph, new_node: dict,
               new_edges: list[tuple[str, str, str]]) -> tuple[list[dict], MemGraph]:
    """Return a NEW (nodes, graph) with the commit applied. Pure: inputs unchanged."""
    new_graph = MemGraph()
    for row in graph.to_rows():
        new_graph.add_edge(row["src"], row["dst"], w=row["weight"])
    for n in graph.nodes():
        new_graph.add_node(n)
    for s, d, _kind in new_edges:
        new_graph.add_edge(s, d, w=1.0)
    new_graph.add_node(new_node["id"])
    return nodes + [new_node], new_graph


def demo_tick_closes_loop(k: int = 5) -> dict:
    """Show a target query failing before the tick and succeeding after.

    The win is GENUINELY edge-driven, not a lexical coincidence:

    * The query mentions ``compute_eta`` and so seeds (both before and after the
      tick) on the existing caller ``planning/eta.py::compute_eta``.
    * The new symbol ``handle_xj7_event`` shares ZERO tokens with the query, so
      it never appears in the lexical baseline and is never its own seed.
    * Pre-tick the new symbol is not in the graph at all -> ``before_hit`` False.
    * Post-tick the only path to it is PPR mass flowing caller -> new symbol
      across the freshly added import/co_edit edges -> it surfaces near the top
      while remaining lexically invisible -> ``after_hit`` True.

    Returns the before/after ranked ids + hit booleans for the notebook to print.
    """
    nodes, graph = build_scm_graph()
    new_node, new_edges = synthetic_commit()
    # Lexically anchored on the EXISTING caller (compute_eta); the opaque new
    # symbol shares no tokens, so only the post-tick edges can surface it.
    query = "which compute_eta planning helper should I extend?"
    positive = new_node["id"]

    before = structured_rank(query, nodes, graph, k=k)
    nodes2, graph2 = apply_tick(nodes, graph, new_node, new_edges)
    after = structured_rank(query, nodes2, graph2, k=k)
    return {
        "query": query,
        "positive": positive,
        "before_ranked": before,
        "after_ranked": after,
        "before_hit": positive in before,
        "after_hit": positive in after,
    }
