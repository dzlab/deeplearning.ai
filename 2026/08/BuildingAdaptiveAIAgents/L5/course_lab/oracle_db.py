"""
course_lab/oracle_db.py — Oracle DB connection helpers for the DLAI continual-learning course.

Env vars (verbatim upstream-canonical names, matching oracleagentmemory package):
  ORACLE_MEMORY_DB_USER           (default: "dmuser")
  ORACLE_MEMORY_DB_PASSWORD       (required for live connections)
  ORACLE_MEMORY_DB_CONNECT_STRING (default: "localhost:1521/FREEPDB1")
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import oracledb


def _oracledb():
    """Lazy import of oracledb so this module can be parsed without the dep."""
    import oracledb as _od
    return _od


class OracleUnavailable(Exception):
    """Raised when an Oracle connection cannot be established."""


def connection_params_from_env() -> dict:
    """Return connection params read from the canonical env vars.

    Returns a dict with keys ``user``, ``password``, and ``dsn``.
    ``password`` is ``None`` when ``ORACLE_MEMORY_DB_PASSWORD`` is unset —
    callers that only inspect params (e.g. tests) can do so safely.
    :func:`get_connection` and :func:`get_pool` will raise
    :exc:`OracleUnavailable` when password is None.
    """
    return {
        "user": os.environ.get("ORACLE_MEMORY_DB_USER", "dmuser"),
        "password": os.environ.get("ORACLE_MEMORY_DB_PASSWORD"),
        "dsn": os.environ.get(
            "ORACLE_MEMORY_DB_CONNECT_STRING", "localhost:1521/FREEPDB1"
        ),
    }


def is_oracle_available() -> bool:
    """Try to open a connection; return False on any failure."""
    _autoload_env()
    params = connection_params_from_env()
    if params["password"] is None:
        return False
    try:
        od = _oracledb()
        conn = od.connect(
            user=params["user"],
            password=params["password"],
            dsn=params["dsn"],
        )
        conn.close()
        return True
    except Exception:
        return False


def _autoload_env() -> None:
    """Populate the ORACLE_MEMORY_DB_* vars from a nearby ``.env`` /
    ``.env.sandbox`` when the password is unset, so a notebook run **manually**
    (Jupyter kernel with no ``export``/``source .env``) still reaches Oracle.

    Only sets vars ABSENT from the environment (a real export always wins); the
    first file found walking up from this module wins. A no-op once the password
    is present, so it costs nothing on the hot path.
    """
    if os.environ.get("ORACLE_MEMORY_DB_PASSWORD"):
        return
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        for name in (".env", ".env.sandbox"):
            f = d / name
            if not f.exists():
                continue
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
            return  # first existing file wins


def get_connection(autocommit: bool = True) -> "oracledb.Connection":
    """Open and return a single Oracle connection.

    Raises :exc:`OracleUnavailable` when ``ORACLE_MEMORY_DB_PASSWORD`` is
    not set or the connect attempt fails.
    """
    _autoload_env()
    _autoload_env()
    params = connection_params_from_env()
    if params["password"] is None:
        raise OracleUnavailable(
            "ORACLE_MEMORY_DB_PASSWORD is not set. "
            "Run scripts/bootstrap_oracle_free.sh and export the printed env vars."
        )
    try:
        od = _oracledb()
        conn = od.connect(
            user=params["user"],
            password=params["password"],
            dsn=params["dsn"],
        )
        conn.autocommit = autocommit
        return conn
    except Exception as exc:
        raise OracleUnavailable(f"Cannot connect to Oracle: {exc}") from exc


class DirectConnectionProvider:
    """Small pool-like adapter backed by one direct Oracle connection.

    Some local Oracle Free containers accept direct ``oracledb.connect`` calls
    but reject thin connection-pool acquisition with ORA-01017. Opening a fresh
    direct connection for every course-extra insert can also exhaust the
    listener during notebook trace generation, so this adapter reuses one
    direct connection while exposing the minimal ``acquire()`` / ``close()``
    surface used by :class:`course_lab.agent_memory.AgentMemory`.
    """

    class _Lease:
        def __init__(self, provider: "DirectConnectionProvider") -> None:
            self._provider = provider

        def __enter__(self):
            return self._provider._connection()

        def __exit__(self, exc_type, exc, tb) -> bool:
            # Keep the shared connection open for the next lease. Roll back on
            # errors so a failed statement does not poison subsequent notebook
            # cells; successful writes explicitly commit in the caller.
            if exc_type is not None:
                try:
                    self._provider._connection().rollback()
                except Exception:
                    self._provider._reset()
            return False

    def __init__(self) -> None:
        self._conn = None

    def _reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _connection(self):
        if self._conn is None:
            self._conn = get_connection(autocommit=True)
            return self._conn
        try:
            self._conn.ping()
        except Exception:
            self._reset()
            self._conn = get_connection(autocommit=True)
        return self._conn

    def acquire(self):
        return self._Lease(self)

    def close(self) -> None:
        self._reset()


def get_pool(min: int = 1, max: int = 4) -> "oracledb.ConnectionPool":
    """Create and return an Oracle connection pool.

    The pool can be passed directly to
    ``OracleAgentMemory(connection=pool, ...)``.

    Raises :exc:`OracleUnavailable` when password is missing or connection
    fails.
    """
    params = connection_params_from_env()
    if params["password"] is None:
        raise OracleUnavailable(
            "ORACLE_MEMORY_DB_PASSWORD is not set. "
            "Run scripts/bootstrap_oracle_free.sh and export the printed env vars."
        )
    try:
        od = _oracledb()
        pool = od.create_pool(
            user=params["user"],
            password=params["password"],
            dsn=params["dsn"],
            min=min,
            max=max,
        )
        return pool
    except Exception as exc:
        raise OracleUnavailable(f"Cannot create Oracle pool: {exc}") from exc


def apply_sql_file(conn, path: str | Path) -> int:
    """Read a ``.sql`` file and execute its statements.

    Statements are split on lines containing only ``/`` (Oracle's standard
    PL/SQL block delimiter).  Empty statements are skipped.

    Returns the count of statements executed.
    """
    sql_text = Path(path).read_text(encoding="utf-8")
    # Split on lines that are exactly "/" (optionally surrounded by whitespace)
    raw_blocks = []
    current: list[str] = []
    for line in sql_text.splitlines():
        if line.strip() == "/":
            raw_blocks.append("\n".join(current))
            current = []
        else:
            current.append(line)
    # Append any trailing block after the last "/"
    if current:
        raw_blocks.append("\n".join(current))

    cursor = conn.cursor()
    executed = 0
    for block in raw_blocks:
        stmt = block.strip()
        if stmt:
            # Strip a trailing semicolon for *plain* SQL — oracledb executes bare
            # SQL, not SQL*Plus. But a PL/SQL block legitimately ends in "END;",
            # and stripping that ";" makes it invalid (PLS-00103). Detect PL/SQL
            # blocks and leave their terminating ";" intact. We look at the first
            # non-comment, non-blank line so leading "-- ..." header comments do
            # not hide the BEGIN/DECLARE keyword.
            first_code = next(
                (ln.strip() for ln in stmt.splitlines()
                 if ln.strip() and not ln.strip().startswith("--")),
                "",
            ).upper()
            is_plsql = first_code.startswith("BEGIN") or first_code.startswith("DECLARE")
            if stmt.endswith(";") and not is_plsql:
                stmt = stmt[:-1].rstrip()
            cursor.execute(stmt)
            executed += 1
    cursor.close()
    return executed


if __name__ == "__main__":
    """Usage: python -m course_lab.oracle_db apply <file.sql>"""
    if len(sys.argv) < 3 or sys.argv[1] != "apply":
        print("Usage: python -m course_lab.oracle_db apply <file.sql>", file=sys.stderr)
        sys.exit(1)

    sql_path = sys.argv[2]
    conn = get_connection(autocommit=True)
    count = apply_sql_file(conn, sql_path)
    conn.close()
    print(f"Applied {count} statement(s) from {sql_path}")
