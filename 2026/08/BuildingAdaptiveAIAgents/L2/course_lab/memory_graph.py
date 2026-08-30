"""Act 2 — co-retrieval / structural graph. Pure-Python adjacency.

NO networkx in the default path (smoke-offline invariant). networkx may be used
as a full-mode convenience by the caller, never imported here.
"""
from __future__ import annotations


class MemGraph:
    """Undirected weighted graph as dict-of-dicts: adj[u][v] = weight."""

    def __init__(self) -> None:
        self._adj: dict[str, dict[str, float]] = {}

    def add_node(self, n: str) -> None:
        self._adj.setdefault(n, {})

    def add_edge(self, u: str, v: str, w: float = 1.0) -> None:
        if u == v:
            return
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = self._adj[u].get(v, 0.0) + w
        self._adj[v][u] = self._adj[v].get(u, 0.0) + w

    def has_edge(self, u: str, v: str) -> bool:
        return v in self._adj.get(u, {})

    def neighbors(self, n: str) -> dict[str, float]:
        return self._adj.get(n, {})

    def nodes(self) -> list[str]:
        return list(self._adj)

    def num_nodes(self) -> int:
        return len(self._adj)

    def num_edges(self) -> int:
        return sum(len(v) for v in self._adj.values()) // 2

    def to_rows(self) -> list[dict]:
        rows, seen = [], set()
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"src": key[0], "dst": key[1], "weight": w})
        return rows


def build_graph(traces, structural_edges, *, domain: str | None = None) -> MemGraph:
    # `domain` is reserved for future per-domain edge-weighting policies; the graph
    # is built identically for every domain today.
    g = MemGraph()
    for t in traces:
        if getattr(t, "outcome", None) != "success":
            continue
        chosen = list(getattr(t, "chosen", []) or [])
        for i in range(len(chosen)):
            for j in range(i + 1, len(chosen)):
                g.add_edge(chosen[i], chosen[j], w=1.0)
    for src, dst, _kind in (structural_edges or []):
        g.add_edge(src, dst, w=1.0)
    return g


def build_agent_memory_graph(nodes, traces, *, co_use_min_count: int = 2,
                             bridge_edges=None) -> MemGraph:
    """Build the agent-memory co-relevance graph from DICT chat traces.

    Distinct from :func:`build_graph` (which reads ``.outcome``/``.chosen`` attrs):
    this reads DICT traces with keys ``used: list[str]`` and ``success: bool``.

    The caller passes ONLY the TRAIN traces (run.py / the eval own the split); this
    function counts every trace it is given. Pairwise co-occurrence is counted
    first, then an edge is added with weight = count ONLY for pairs reaching
    ``co_use_min_count`` — so true bundles survive as cliques weighted by real
    co-occurrence while injected single-shot noise pairs drop out.

    Every passed-in vocab node is added (isolated or not) so PPR's ranking space
    is the full vocab. ``bridge_edges`` is an optional list of ``(mem_id, code_id)``
    pairs laid as thin provenance edges into the code graph.
    """
    counts: dict[tuple[str, str], int] = {}
    for t in traces:
        if not t.get("success"):
            continue
        used = sorted(set(t.get("used", []) or []))
        for i in range(len(used)):
            for j in range(i + 1, len(used)):
                key = (used[i], used[j])
                counts[key] = counts.get(key, 0) + 1

    g = MemGraph()
    for n in nodes or []:
        g.add_node(n["id"] if isinstance(n, dict) else n)
    for (a, b), c in counts.items():
        if c >= co_use_min_count:
            g.add_edge(a, b, w=float(c))
    for a, b in (bridge_edges or []):
        g.add_edge(a, b, w=1.0)
    return g


def restructure(graph: MemGraph, scorecard: dict) -> tuple[MemGraph, dict]:
    """Merge duplicate clusters into a single representative node; report after-scorecard."""
    new = MemGraph()
    rep: dict[str, str] = {}
    for cluster in scorecard.get("duplicate_clusters", []):
        if not cluster:
            continue
        head = cluster[0]
        for member in cluster:
            rep[member] = head

    def r(n: str) -> str:
        return rep.get(n, n)

    for row in graph.to_rows():
        new.add_edge(r(row["src"]), r(row["dst"]), w=row["weight"])
    for n in graph.nodes():
        new.add_node(r(n))
    after = {
        "n_memories": new.num_nodes(),
        "dup_pct": 0.0,
        "orphan_pct": round(
            sum(1 for n in new.nodes() if not new.neighbors(n)) / max(new.num_nodes(), 1), 4),
        "stale_pct": scorecard.get("stale_pct", 0.0),
        "contradiction_count": scorecard.get("contradiction_count", 0),
    }
    return new, after
