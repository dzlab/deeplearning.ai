"""course_lab/sandbox.py: idempotent learner-sandbox bring-up.

The learner sandbox needs Oracle running (and, for the in-DB reference path
only, the ONNX embedding model registered + the schema at 384-d) before any
notebook can seed. ``ensure_ready()`` performs that bring-up **idempotently**:
it probes the current state and only does the expensive work (container
bootstrap, and -- when ``with_onnx=True`` -- the 90 MB ONNX upload) when
something is actually missing.

The **default embedding path is out-of-DB MiniLM** (sentence-transformers,
``embedder_model: "python:..."``): embeddings are computed in the notebook's
process and only stored/searched in Oracle, so the DB never runs the model.
That avoids the OOM that registering + running the in-DB ONNX model causes on
tight CPU sandboxes, so ``ensure_ready`` does **not** load the ONNX model by
default. Pass ``with_onnx=True`` (or ``lab.py load-onnx-model``) to provision
the in-DB ONNX reference path.

It is safe to call from anywhere, repeatedly:

* the container entrypoint (``scripts/sandbox_start.sh`` / ``lab.py
  sandbox-start``) runs it once at start, and
* each notebook's first cell calls it as a self-healing guard, so a notebook
  still works if the entrypoint never ran.

Bring-up is provisioning only -- it does NOT seed memory. Each notebook seeds
lazily on first run via ``course_lab.memory_snapshot.seed_or_load``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ONNX_MODEL_NAME = "ALL_MINILM_L6_V2"
ONNX_DIM = 384
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env_files() -> None:
    """Populate ORACLE_MEMORY_DB_* from .env, else committed .env.sandbox.

    Only sets vars that are not already in the environment (a real .env or the
    shell wins), so a notebook guard works even if no shell sourced the creds.
    Best-effort: a parse error never blocks bring-up.
    """
    for name in (".env", ".env.sandbox"):
        path = _PROJECT_ROOT / name
        if not path.exists():
            continue
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
        except Exception:
            pass
        return  # first existing file wins (.env over .env.sandbox)


def oracle_reachable() -> bool:
    """True when an Oracle connection can be opened with the current env."""
    try:
        from course_lab import oracle_db
        conn = oracle_db.get_connection()
        conn.close()
        return True
    except Exception:
        return False


def onnx_model_registered(conn) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :n",
        {"n": ONNX_MODEL_NAME},
    )
    return cur.fetchone()[0] >= 1


def schema_dim(conn):
    """The agent-memory schema's recorded vector_dim, or None if no schema yet.

    A missing schema-meta table means the schema has not been created; that is
    fine -- it will be created at the embedder's dimension on first seed, so it
    is not a reason to re-run the ONNX load.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT metadata_value FROM DLAI_ORACLEAGENTMEMORY_SCHEMA_META "
            "WHERE metadata_key = 'vector_dim'"
        )
        row = cur.fetchone()
    except Exception:
        return None
    return int(row[0]) if row and row[0] is not None else None


def onnx_provisioned(conn) -> bool:
    """True when the ONNX model is registered and the schema dim is compatible.

    Compatible = the model is registered AND either no schema exists yet (will
    be created at 384-d) or the existing schema is already 384-d. A schema at a
    different dim (e.g. a leftover 1024-d OCI schema) is NOT provisioned.
    """
    if not onnx_model_registered(conn):
        return False
    dim = schema_dim(conn)
    return dim is None or dim == ONNX_DIM


def _run(cmd: list[str], *, env=None) -> int:
    print(f"[setup] $ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(_PROJECT_ROOT), env=env)


def bootstrap_oracle() -> None:
    """Run the idempotent Oracle Free bootstrap (container + course DDL).

    The shell script is also responsible for installing Docker/Podman when no
    runtime is present and the current sandbox has root/passwordless-sudo
    privileges. Let it produce the platform-specific diagnostic instead of
    pre-failing here.
    """
    script = _PROJECT_ROOT / "scripts" / "bootstrap_oracle_free.sh"
    rc = _run(["bash", str(script)])
    if rc != 0:
        raise RuntimeError(f"bootstrap_oracle_free.sh failed (rc={rc})")


def load_onnx_model() -> None:
    """Register the ONNX MiniLM model + align the schema dimension (idempotent).

    scripts/ is not a package, so load the script by path rather than import it.
    """
    import importlib.util
    script_path = _PROJECT_ROOT / "scripts" / "load_onnx_embedder.py"
    spec = importlib.util.spec_from_file_location("_load_onnx_embedder", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rc = mod.main()
    if rc != 0:
        raise RuntimeError(f"load_onnx_embedder failed (rc={rc})")


def ensure_ready(*, verbose: bool = True, with_onnx: bool = False) -> dict:
    """Idempotently provision the sandbox: Oracle up (+ in-DB ONNX model when
    ``with_onnx``).

    Returns a small report of what was done vs. skipped. Provisioning only --
    does not seed memory. Safe to call repeatedly from the entrypoint or a
    notebook's first cell.

    The default embedding path is out-of-DB MiniLM, which needs no in-DB model,
    so ``with_onnx`` defaults to ``False`` (skips the 90 MB ONNX upload that
    OOMs CPU sandboxes). Set ``with_onnx=True`` only for the in-DB ONNX
    reference path. The agent-memory schema is created at the embedder's
    dimension (384-d) on first seed either way.
    """
    report = {"oracle": "ok", "onnx": "skipped"}

    # Self-sufficient creds: a notebook guard may run before any shell sourced
    # .env, so load it (or the committed .env.sandbox) here. Env/real-.env wins.
    _load_env_files()

    def log(msg):
        if verbose:
            print(f"[setup] {msg}")

    # 1. Database reachable? Bootstrap only if not.
    if oracle_reachable():
        log("Database is reachable.")
    else:
        log("Database not reachable - preparing environment (one-time)...")
        bootstrap_oracle()
        # bootstrap exports the ORACLE_MEMORY_DB_* vars in its own shell; the
        # current process must have them too. Default them like the script does.
        os.environ.setdefault("ORACLE_MEMORY_DB_USER", "dmuser")
        os.environ.setdefault(
            "ORACLE_MEMORY_DB_PASSWORD",
            os.environ.get("ORACLE_PWD", "continual_learning"),
        )
        os.environ.setdefault(
            "ORACLE_MEMORY_DB_CONNECT_STRING",
            f"localhost:{os.environ.get('ORACLE_PORT', '1521')}/FREEPDB1",
        )
        if not oracle_reachable():
            raise RuntimeError(
                "Database still unreachable after setup. Check the container "
                "and the ORACLE_MEMORY_DB_* environment variables."
            )
        report["oracle"] = "bootstrapped"

    # 2. In-DB ONNX model + schema dim — ONLY for the in-DB reference path.
    # The default out-of-DB MiniLM path needs no in-DB model: embeddings are
    # computed in-process and the schema is created at 384-d on first seed.
    # (Deliberately silent in the default path: learners don't need to know
    # where embeddings are computed — instructor feedback 2026-07-07.)
    if with_onnx:
        from course_lab import oracle_db
        conn = oracle_db.get_connection()
        try:
            provisioned = onnx_provisioned(conn)
        finally:
            conn.close()

        if provisioned:
            log(f"ONNX model {ONNX_MODEL_NAME} registered and schema is "
                f"{ONNX_DIM}-d - skipping load-onnx-model.")
        else:
            log("ONNX model/schema not provisioned - running load-onnx-model "
                "(one-time, ~90 MB upload)...")
            load_onnx_model()
            report["onnx"] = "loaded"

    log("Environment is ready.")
    return report


__all__ = ["ensure_ready", "oracle_reachable", "onnx_provisioned",
           "onnx_model_registered", "schema_dim", "bootstrap_oracle",
           "load_onnx_model", "ONNX_MODEL_NAME", "ONNX_DIM"]


if __name__ == "__main__":
    rep = ensure_ready()
    print(f"[setup] result: {rep}")
    sys.exit(0)
