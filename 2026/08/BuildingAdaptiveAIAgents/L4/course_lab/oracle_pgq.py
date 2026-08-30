"""Live Oracle SQL/PGQ property graph over the course graph tables.

Module 3's "closing the loop" section persists a real ``/graphify`` snapshot
into ``MEMORY_GRAPH_NODES`` / ``MEMORY_GRAPH_EDGES`` (see
``AgentMemory.upsert_graph_*``). Those are plain relational tables; this module
layers an Oracle **SQL property graph** on top so the dependency neighborhood of
a symbol is matched **in the database** with ``GRAPH_TABLE ... MATCH`` rather
than reconstructed in Python.

Verified live on the course container (Oracle AI Database 26ai Free, 23.26.1):
``CREATE PROPERTY GRAPH`` with the real composite keys, a domain-scoped 1-hop
MATCH, and the quantified path ``-[]->{1,h}`` for multi-hop all execute.

The property graph spans every domain in the tables (a graph is defined over
whole tables); a query scopes to one domain with a ``WHERE v.domain = :domain``
predicate. ``domain`` is exposed as a vertex+edge property for exactly this.

This is the *traversal/match* half of structure-aware retrieval. Ranking
(Personalized PageRank) still runs in Python over the loaded graph — PGQ finds
the structural neighborhood, Python scores it.
"""

from __future__ import annotations

DEFAULT_GRAPH_NAME = "mem_code_graph"


def _require_pool(mem):
    pool = getattr(mem, "_pool", None)
    if pool is None:
        raise RuntimeError(
            "oracle_pgq requires a live Oracle connection pool "
            "(AgentMemory built with memory_backend=real)."
        )
    return pool


def create_property_graph(mem, *, graph_name: str = DEFAULT_GRAPH_NAME) -> None:
    """(Re)create the property graph over MEMORY_GRAPH_NODES / _EDGES.

    Idempotent: drops an existing graph of the same name first. ``domain`` is
    part of the vertex/edge keys (the tables are keyed on (id, domain) and
    (src, dst, kind, domain)) and is also published as a property so queries can
    scope to a single ``/graphify`` snapshot.
    """
    pool = _require_pool(mem)
    ddl = f"""
        CREATE PROPERTY GRAPH {graph_name}
          VERTEX TABLES (
            MEMORY_GRAPH_NODES
              KEY (id, domain)
              PROPERTIES (id, domain)
          )
          EDGE TABLES (
            MEMORY_GRAPH_EDGES
              KEY (src, dst, kind, domain)
              SOURCE      KEY (src, domain) REFERENCES MEMORY_GRAPH_NODES (id, domain)
              DESTINATION KEY (dst, domain) REFERENCES MEMORY_GRAPH_NODES (id, domain)
              PROPERTIES (kind, domain)
          )
    """
    with pool.acquire() as conn:
        cur = conn.cursor()
        try:
            cur.execute(f"DROP PROPERTY GRAPH {graph_name}")
        except Exception:
            # First run: nothing to drop. Any real DDL error surfaces on CREATE.
            pass
        cur.execute(ddl)
        conn.commit()


def match_neighborhood(mem, *, anchor: str, domain: str,
                       hops: int = 2,
                       graph_name: str = DEFAULT_GRAPH_NAME) -> list[dict]:
    """Match the 1..``hops`` dependency neighborhood of ``anchor`` in ``domain``.

    Returns a list of ``{"neighbor": str, "hops": int}`` rows — the distinct
    symbols reachable from ``anchor`` within ``hops`` edges, found purely by the
    Oracle graph engine, ordered by shortest reach. Hard-requires PGQ; any
    database error propagates.

    Implementation note: a quantified path ``-[]->{1,h}`` binds the edge as a
    *group* variable, so an edge property like ``kind`` cannot be projected in
    COLUMNS (ORA-40990). We therefore match each fixed length 1..hops with its
    own singleton path and tag the row with that length — giving the shortest
    hop-distance per neighbor without referencing a group edge variable. Use
    :func:`match_edges` for the 1-hop edge kinds.
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")
    pool = _require_pool(mem)
    # One singleton path per exact length h: (v)-[]->()-[]->...(w). Each is a
    # plain pattern (no group variable), so it is valid and lets us record the
    # exact hop distance. Shortest distance wins on duplicates.
    best: dict[str, int] = {}
    with pool.acquire() as conn:
        cur = conn.cursor()
        for h in range(1, int(hops) + 1):
            chain = "".join(f"-[]->(n{i})" for i in range(h - 1)) + "-[]->(w)"
            sql = f"""
                SELECT neighbor FROM GRAPH_TABLE ({graph_name}
                  MATCH (v){chain}
                  WHERE v.id = :anchor AND v.domain = :domain
                  COLUMNS (w.id AS neighbor)
                )
            """
            cur.execute(sql, {"anchor": anchor, "domain": domain})
            for (nb,) in cur.fetchall():
                if nb != anchor and (nb not in best or h < best[nb]):
                    best[nb] = h
    return [{"neighbor": nb, "hops": h}
            for nb, h in sorted(best.items(), key=lambda kv: (kv[1], kv[0]))]


def match_edges(mem, *, anchor: str, domain: str,
                graph_name: str = DEFAULT_GRAPH_NAME) -> list[dict]:
    """Match the direct (1-hop) out-edges of ``anchor`` with their kind.

    Returns ``{"neighbor": str, "kind": str}`` rows. A single-hop pattern binds
    ``e`` as a scalar, so the edge ``kind`` *can* be projected here (unlike the
    quantified path in :func:`match_neighborhood`).
    """
    pool = _require_pool(mem)
    sql = f"""
        SELECT neighbor, kind FROM GRAPH_TABLE ({graph_name}
          MATCH (v) -[e]-> (w)
          WHERE v.id = :anchor AND v.domain = :domain
          COLUMNS (w.id AS neighbor, e.kind AS kind)
        )
    """
    out: list[dict] = []
    with pool.acquire() as conn:
        cur = conn.cursor()
        cur.execute(sql, {"anchor": anchor, "domain": domain})
        for nb, kind in cur.fetchall():
            out.append({"neighbor": nb, "kind": kind})
    return out
