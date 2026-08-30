"""Lesson helper — credentials, live-service accessors, and a thin re-export of
the vendored course_lab surface this lesson uses.

DLAI convention: a .env file one directory above the lesson holds keys, pulled
into the environment by python-dotenv. This lesson runs from the committed
fixtures in ./ro_shared_data by default; the get_* accessors below are the
optional live-upgrade path.
"""

import os

from dotenv import load_dotenv, find_dotenv


def load_env():
    """Load .env (keys + Oracle/OCI connection settings) into the environment."""
    _ = load_dotenv(find_dotenv())


# --- credentials (DLAI standard) -------------------------------------------

def get_openai_api_key():
    load_env()
    return os.getenv("OPENAI_API_KEY")



# --- live Oracle (the agent-memory + graph spine; required for this lesson) --

import pathlib  # noqa: E402

_SCHEMA_READY = False
_SQL_DIR = pathlib.Path(__file__).resolve().parent / "course_lab" / "sql"
_BOOTSTRAP_SCRIPT = (
    pathlib.Path(__file__).resolve().parent / "scripts" / "bootstrap_oracle_free.sh"
)
_ORACLE_WAIT_SECONDS = 180
# Errors that mean "the database is still starting" (sidecar containers take
# ~30-60s): safe to retry. Auth errors are deliberately NOT here — retrying a
# rejected password can trip FAILED_LOGIN_ATTEMPTS and lock the account.
_ORACLE_STARTING_MARKERS = (
    "ORA-12514",  # listener up, service not registered yet
    "ORA-12528",  # instance starting, blocking connections
    "ORA-12537",  # connection closed during handshake
    "ORA-12541",  # no listener yet
    "ORA-01033",  # initialization in progress
    "ORA-01109",  # database not open
    "DPY-6001",   # python-oracledb: cannot connect
    "DPY-6005",   # python-oracledb: connection failed
    "Connection refused",
    "timed out",
)


def _bootstrap_oracle() -> bool:
    """Provision the course user by running scripts/bootstrap_oracle_free.sh.

    Fresh learner sandboxes hand out an Oracle container whose ``dmuser``
    account does not exist yet; the script creates it as SYSTEM over the
    network (it never touches containers when the DB is already reachable)
    and applies the course DDL. Returns True when the script succeeded.
    """
    if not _BOOTSTRAP_SCRIPT.is_file():
        return False
    print(
        "[helper] Provisioning the course user via scripts/bootstrap_oracle_free.sh..."
    )
    import subprocess
    try:
        res = subprocess.run(
            ["bash", str(_BOOTSTRAP_SCRIPT)],
            capture_output=True, text=True, timeout=600,
        )
    except Exception as exc:
        print(f"[helper] WARNING: could not run the bootstrap script: {exc}")
        return False
    if res.returncode != 0:
        tail = "\n".join((res.stdout + "\n" + res.stderr).splitlines()[-15:])
        print(
            f"[helper] WARNING: bootstrap script exited {res.returncode}:\n{tail}"
        )
        return False
    return True


def _connect_ready(oracle_db, autocommit: bool = True):
    """Open a course-user connection, self-provisioning on first use.

    - Auth failure (ORA-01017/ORA-28000): the sandbox DB has no ``dmuser``
      yet — run the bootstrap script once, then retry the connect once.
    - Database-still-starting errors: retry quietly for up to
      ``_ORACLE_WAIT_SECONDS`` (the sidecar needs ~30-60s after kernel start).
    Anything else (bad host, unset password, ...) raises immediately.
    """
    import time
    deadline = time.monotonic() + _ORACLE_WAIT_SECONDS
    bootstrapped = False
    waiting_printed = False
    while True:
        try:
            return oracle_db.get_connection(autocommit=autocommit)
        except oracle_db.OracleUnavailable as exc:
            msg = str(exc)
            if ("ORA-01017" in msg or "ORA-28000" in msg) and not bootstrapped:
                bootstrapped = True
                if _bootstrap_oracle():
                    continue
                raise
            if (
                any(m in msg for m in _ORACLE_STARTING_MARKERS)
                and time.monotonic() < deadline
            ):
                if not waiting_printed:
                    waiting_printed = True
                    print("[helper] Oracle is still starting — waiting ...")
                time.sleep(5)
                continue
            raise


def ensure_schema(conn=None) -> int:
    """Create this lesson's Oracle schema if it does not exist yet.

    Applies the vendored course DDL in ``course_lab/sql/`` (skillbox, query/
    retrieval traces, voyage manifest, memory graph, governance columns). Each
    migration guards itself (ORA-00955/-01430 swallowed), so this is idempotent
    and safe to call on every connect. Runs at most once per process. A no-op
    when no sql/ directory is vendored (CPU-only lessons). Returns the number of
    migrations applied (0 if already ready or none to apply).
    """
    global _SCHEMA_READY
    if _SCHEMA_READY or not _SQL_DIR.is_dir():
        return 0
    from course_lab import oracle_db
    own = conn is None
    if own:
        conn = _connect_ready(oracle_db)
    applied = 0
    try:
        for sql in sorted(_SQL_DIR.glob("*.sql")):
            try:
                oracle_db.apply_sql_file(conn, sql)
                applied += 1
            except Exception as exc:  # one bad migration must not block the lesson
                print(f"[helper] WARNING: applying {sql.name} failed: {exc}")
    finally:
        if own:
            conn.close()
    _SCHEMA_READY = True
    return applied


def get_oracle_connection(autocommit: bool = True):
    """Open a live Oracle connection from .env settings, ensuring the schema.

    Reads ORACLE_MEMORY_DB_{USER,PASSWORD,CONNECT_STRING} (defaults match the
    course's dlai-oracle-free container at localhost:1521/FREEPDB1). The first
    call also creates the lesson's schema (idempotent) so the notebook runs
    against a correctly-shaped database with no separate provisioning step.
    """
    load_env()
    from course_lab import oracle_db
    conn = _connect_ready(oracle_db, autocommit=autocommit)
    ensure_schema(conn)
    return conn


def oracle_available() -> bool:
    """True if a live Oracle connection can be opened (else run from fixtures)."""
    load_env()
    from course_lab import oracle_db
    try:
        return oracle_db.is_oracle_available()
    except Exception:
        return False



# --- live OCI Grok (optional; seeding/extraction. Default = committed caches) -

def get_oci_client():
    """Return the course's OCI GenAI chat helper (live LLM upgrade path).

    By default the lesson replays committed LLM caches and does not need this.
    """
    load_env()
    from course_lab import oci_client
    return oci_client



# --- lesson surface (vendored from course_lab) ------------------------------
from course_lab.code_graph_parse import load_code_graph    # noqa: E402, F401
from course_lab.m3_lab import (                            # noqa: E402, F401
    setup_module3, connect_module3, load_real_graph,
    render_starter_graph, render_anchor_walk, render_growth_step,
    render_full_graph, render_files_only_graph,
    hybrid_seed_then_ppr, semantic_seed_then_ppr, build_node_embeddings,
    peek_hybrid_halves, run_retrieval_eval, narrate_retrieval_comparison,
)
from course_lab.memory_audit import audit_scorecard        # noqa: E402, F401
from course_lab.memory_graph import build_graph, restructure  # noqa: E402, F401
from course_lab import gv_lab, scm_graph, paths            # noqa: E402, F401
