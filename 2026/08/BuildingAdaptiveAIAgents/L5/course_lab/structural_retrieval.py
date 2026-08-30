"""Act 3 — structural retrieval: Personalized PageRank, a flat hash-cosine baseline,
and an honest lexical (token-overlap) baseline.

Pure NumPy. Deterministic. No training.
"""
from __future__ import annotations

import math
import re

import numpy as np

from course_lab.agent_memory import HashEmbedder
from course_lab.memory_graph import MemGraph


def personalized_pagerank(graph: MemGraph, seed_weights: dict,
                          *, alpha: float = 0.85, iters: int = 50) -> dict:
    nodes = graph.nodes()
    if not nodes:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    M = np.zeros((n, n), dtype=np.float64)
    dangling = np.zeros(n, dtype=np.float64)
    for u in nodes:
        nbrs = dict(graph.neighbors(u))
        if nbrs:
            # connected node keeps a lazy self-loop so seed mass stays put
            nbrs[u] = nbrs.get(u, 0.0) + 1.0
            tot = sum(nbrs.values())
            for v, w in nbrs.items():
                M[idx[v], idx[u]] = w / tot
        else:
            # isolated node: no self-loop hoarding; treat as dangling and send
            # its mass back through the teleport vector each iteration.
            dangling[idx[u]] = 1.0
    p = np.zeros(n, dtype=np.float64)
    s = sum(seed_weights.values()) or 1.0
    for node, w in seed_weights.items():
        if node in idx:
            p[idx[node]] = w / s
    if p.sum() == 0:
        p[:] = 1.0 / n
    r = p.copy()
    for _ in range(iters):
        dmass = float(r @ dangling)
        r = alpha * (M @ r) + alpha * dmass * p + (1 - alpha) * p
    total = r.sum() or 1.0
    r = r / total
    return {nodes[i]: float(r[i]) for i in range(n)}


def seed_from_query(query: str, graph: MemGraph, *, embedder=None, k: int = 3) -> dict:
    """Pick PPR seed nodes by cosine between the query and each node label.

    On the deterministic HashEmbedder smoke path, node-string cosine is effectively a
    hash bucket, not a semantic match; structural PPR propagation (not seed quality) is
    what carries retrieval. Full mode with a real embedder makes the seed semantic.
    """
    embedder = embedder or HashEmbedder()
    nodes = graph.nodes()
    if not nodes:
        return {}
    q = np.array(embedder.embed(query), dtype=np.float32)
    qn = q / (np.linalg.norm(q) or 1.0)
    sims = []
    for node in nodes:
        v = np.array(embedder.embed(node), dtype=np.float32)
        vn = v / (np.linalg.norm(v) or 1.0)
        sims.append((node, float(qn @ vn)))
    sims.sort(key=lambda t: -t[1])
    return {node: 1.0 for node, _ in sims[:k]}


def flat_cosine_baseline(query: str, corpus: list[str], *, embedder=None, k: int) -> list[str]:
    embedder = embedder or HashEmbedder()
    q = np.array(embedder.embed(query), dtype=np.float32)
    qn = q / (np.linalg.norm(q) or 1.0)
    scored = []
    for text in corpus:
        v = np.array(embedder.embed(text), dtype=np.float32)
        vn = v / (np.linalg.norm(v) or 1.0)
        scored.append((text, float(qn @ vn)))
    scored.sort(key=lambda t: -t[1])
    return [t for t, _ in scored[:k]]


_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def identifier_tokens(text: str) -> set[str]:
    """Lower-case word tokens, also splitting snake_case into parts.

    Shared by ``lexical_flat_baseline`` (the honest baseline) and Module 3's
    lexical PPR seeding so the two tokenize identically — the comparison is only
    fair if the seed and the baseline see the same tokens.
    """
    out: set[str] = set()
    for raw in _TOKEN_RE.findall(text.lower()):
        out.add(raw)
        out.update(raw.split("_"))
    return {t for t in out if t}


def lexical_flat_baseline(query: str, nodes: list[dict], *, k: int) -> list[str]:
    """Token-overlap ranking, NO graph. The honest 'lexical recall only' baseline:
    structure must beat THIS on multi-hop, not just beat random hash cosine."""
    q = identifier_tokens(query)
    scored = [(nd["id"], len(q & identifier_tokens(nd["id"] + " " + nd["text"]))) for nd in nodes]
    scored.sort(key=lambda t: (-t[1], t[0]))
    return [nid for nid, _ in scored[:k]]


def ndcg_at_k(ranked: list[str], positive: str, *, k: int) -> float:
    for i, item in enumerate(ranked[:k]):
        if item == positive:
            return 1.0 / math.log2(i + 2)
    return 0.0


def recall_at_k(ranked: list[str], positive: str, *, k: int) -> float:
    return 1.0 if positive in ranked[:k] else 0.0
