"""Act 1 — memory-quality audit. Pure CPU, deterministic.

Scores a memory store for duplicates, orphans, staleness, and contradictions.
Smoke uses the deterministic HashEmbedder for near-dup cosine; contradictions
use a cheap heuristic (LLM-judged only in full mode, handled by the caller).
"""
from __future__ import annotations

import numpy as np

from course_lab.agent_memory import HashEmbedder


def _cosine_matrix(vecs: np.ndarray) -> np.ndarray:
    # O(n^2): fine for course-scale stores (<~1k memories); use LSH/ANN bucketing at scale.
    norm = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    unit = vecs / norm
    return unit @ unit.T


def audit_scorecard(memories: list[dict], edges: list[tuple],
                    *, dup_threshold: float = 0.92,
                    embedder=None) -> dict:
    """Return a quality scorecard for the given memories + structural edges.

    memories: [{"id": str, "text": str, "ts"?: iso-str, "_stale"?: bool}]
    edges:    [(src_id, dst_id, kind), ...]

    Computes ``dup_pct`` (near-dup cosine clusters) and ``orphan_pct`` (ids absent
    from ``edges``) directly. ``stale_pct`` reflects ONLY the caller-supplied
    ``_stale`` flag, and ``contradiction_count`` is always 0 in smoke mode — full
    mode overrides it with an LLM-judged count. Memory ids must live in the same
    namespace as the edge endpoints, or every memory reads as an orphan.
    """
    embedder = embedder or HashEmbedder()
    n = len(memories)
    if n == 0:
        return {"n_memories": 0, "dup_pct": 0.0, "orphan_pct": 0.0,
                "stale_pct": 0.0, "contradiction_count": 0,
                "duplicate_clusters": [], "orphan_ids": []}

    ids = [m["id"] for m in memories]
    vecs = np.array([embedder.embed(m["text"]) for m in memories], dtype=np.float32)
    sim = _cosine_matrix(vecs)

    parent = {i: i for i in range(n)}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= dup_threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[str]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(ids[i])
    duplicate_clusters = [g for g in groups.values() if len(g) > 1]
    n_dup_members = sum(len(g) for g in duplicate_clusters)

    linked = set()
    for src, dst, _kind in edges:
        linked.add(src)
        linked.add(dst)
    orphan_ids = [i for i in ids if i not in linked]

    stale_ids = [m["id"] for m in memories if m.get("_stale")]

    return {
        "n_memories": n,
        "dup_pct": round(n_dup_members / n, 4),
        "orphan_pct": round(len(orphan_ids) / n, 4),
        "stale_pct": round(len(stale_ids) / n, 4),
        "contradiction_count": 0,
        "duplicate_clusters": duplicate_clusters,
        "orphan_ids": orphan_ids,
    }
