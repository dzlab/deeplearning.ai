"""Course-owned VOYAGE_MANIFEST helpers — seed + query the known-voyage manifest.

The manifest is the durable, queryable ground truth of the **known** voyages the
Module 3 / Module 4 demos ask the models about. A voyage that IS in the manifest
should be ANSWERED; a voyage that is ABSENT should be REFUSED. Seeding only the
known voyages (``make_voyages()``, ``VY-1000..``) makes the refusal demo
verifiable: the held-out eval ids (``VY-8xxx``) and train-time unknowns
(``VY-7xxx``) are deliberately absent, so their absence is checkable in-DB.

All functions accept an optional ``conn``; when omitted they open one via
:func:`course_lab.oracle_db.get_connection`, which raises
:class:`~course_lab.oracle_db.OracleUnavailable` when Oracle creds are missing
(the course is real-only — no fake path here).

The table itself is created by ``scripts/sql/004_voyage_manifest.sql`` at
bootstrap; these helpers only read/write rows.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

from course_lab import oracle_db

_COLUMNS = ("vid", "vessel", "cargo", "origin", "dest")


@contextmanager
def _conn_ctx(conn=None):
    """Yield a connection, opening (and closing) our own only if none is given."""
    if conn is not None:
        yield conn
        return
    owned = oracle_db.get_connection(autocommit=True)
    try:
        yield owned
    finally:
        try:
            owned.close()
        except Exception:
            pass


def seed_voyage_manifest(voyages: Iterable[dict], *, conn=None) -> int:
    """Upsert ``voyages`` into VOYAGE_MANIFEST. Idempotent (MERGE on vid).

    Each voyage is a dict with keys {vid, vessel, cargo, origin, dest} — exactly
    the shape :func:`course_lab.seed_supplychain.make_voyages` returns. Returns
    the number of rows written.
    """
    rows = [
        {c: str(v[c]) for c in _COLUMNS}
        for v in voyages
    ]
    if not rows:
        return 0

    merge_sql = """
        MERGE INTO VOYAGE_MANIFEST t
        USING (SELECT :vid AS vid, :vessel AS vessel, :cargo AS cargo,
                      :origin AS origin, :dest AS dest FROM dual) s
        ON (t.vid = s.vid)
        WHEN MATCHED THEN UPDATE SET
            t.vessel = s.vessel, t.cargo = s.cargo,
            t.origin = s.origin, t.dest = s.dest
        WHEN NOT MATCHED THEN
            INSERT (vid, vessel, cargo, origin, dest)
            VALUES (s.vid, s.vessel, s.cargo, s.origin, s.dest)
    """
    with _conn_ctx(conn) as c:
        cur = c.cursor()
        cur.executemany(merge_sql, rows)
        c.commit()
        cur.close()
    return len(rows)


def list_voyages(*, conn=None) -> list[dict]:
    """Return all manifest rows as dicts {vid, vessel, cargo, origin, dest}, vid-sorted."""
    with _conn_ctx(conn) as c:
        cur = c.cursor()
        cur.execute(
            "SELECT vid, vessel, cargo, origin, dest "
            "FROM VOYAGE_MANIFEST ORDER BY vid"
        )
        out = [
            {"vid": r[0], "vessel": r[1], "cargo": r[2],
             "origin": r[3], "dest": r[4]}
            for r in cur.fetchall()
        ]
        cur.close()
    return out


def voyage_exists(vid: str, *, conn=None) -> bool:
    """True iff ``vid`` is in the manifest (i.e. a KNOWN voyage to be answered)."""
    with _conn_ctx(conn) as c:
        cur = c.cursor()
        cur.execute(
            "SELECT 1 FROM VOYAGE_MANIFEST WHERE vid = :vid", {"vid": str(vid)}
        )
        found = cur.fetchone() is not None
        cur.close()
    return found
