"""course_lab/onnx_embedder.py — in-database ONNX embeddings.

Embeddings are computed by the Oracle AI Database itself via
``VECTOR_EMBEDDING(<model> USING :t AS DATA)`` against an ONNX MiniLM model
registered with ``onnx2oracle`` (Oracle model name ``ALL_MINILM_L6_V2``,
384-d, L2-normalized). No OCI, no external embedding API.

``OnnxDBEmbedder`` implements ``oracleagentmemory``'s ``IEmbedder`` interface
so it drops straight into ``OracleAgentMemory(embedder=...)``.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

# Oracle mining-model names are identifiers: start with a letter, then
# letters/digits/underscore, max 128 chars. We uppercase-normalize because
# onnx2oracle registers models uppercased, and an identifier cannot be bound
# as a parameter — it must be interpolated, so it MUST be whitelisted.
_IDENT_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}\Z")


def _validate_model_name(name: str) -> str:
    if not isinstance(name, str) or not _IDENT_RE.match(name):
        raise ValueError(
            f"invalid Oracle model name {name!r}: must match {_IDENT_RE.pattern} "
            "(uppercase identifier registered by onnx2oracle)"
        )
    return name


class OnnxDBEmbedder:
    """``IEmbedder``-compatible embedder backed by an in-DB ONNX model.

    Parameters
    ----------
    connection:
        A live ``oracledb`` connection (or anything exposing ``cursor()``).
    model_name:
        The registered Oracle ONNX model name, e.g. ``ALL_MINILM_L6_V2``.
    """

    def __init__(self, connection: Any, *, model_name: str = "ALL_MINILM_L6_V2") -> None:
        self._conn = connection
        self._model = _validate_model_name(model_name)

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # MiniLM has no query/document asymmetry, so is_query is a no-op.
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # Model name is a validated identifier (not bindable); text is bound.
        sql = f"SELECT VECTOR_EMBEDDING({self._model} USING :t AS DATA) FROM dual"
        rows: list[np.ndarray] = []
        with self._conn.cursor() as cur:
            for t in texts:
                cur.execute(sql, t=t)
                vec = cur.fetchone()[0]
                rows.append(np.asarray(list(vec), dtype=np.float32))
        return np.vstack(rows)

    async def embed_async(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # The DB call is synchronous; the workshop's volumes make this fine.
        return self.embed(texts, is_query=is_query)


def ensure_onnx_model(
    connection: Any, *, model_name: str = "ALL_MINILM_L6_V2"
) -> bool:
    """Return True iff ``model_name`` is registered in the DB.

    This is a *check*, not a loader — the model is registered out-of-band by
    ``scripts/load_onnx_embedder.py`` / ``lab.py load-onnx-model`` (which call
    ``onnx2oracle``). Kept here so the notebook guard and tests share one
    source of truth.
    """
    name = _validate_model_name(model_name)
    with connection.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM user_mining_models WHERE model_name = :m", m=name
        )
        return cur.fetchone()[0] > 0


def reset_schema_if_dim_changed(
    connection: Any,
    *,
    model_name: str = "ALL_MINILM_L6_V2",
    table_prefix: str = "DLAI_",
) -> bool:
    """Drop the agent-memory schema tables iff the recorded ``vector_dim`` no
    longer matches the ONNX model's output dimension.

    The ``oracleagentmemory`` schema records the embedding dimension; switching
    embedders (e.g. a 1024-d OCI model -> a 384-d in-DB ONNX model) leaves the
    old dimension behind and makes ``CREATE_IF_NECESSARY`` refuse to attach.
    This drops the prefixed agent-memory tables so the next construction
    rebuilds them at the right dimension. It is a **no-op** when the schema is
    absent or the dimensions already match — it never drops on a match.

    Returns True if a reset (table drop) was performed, else False.
    """
    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", table_prefix):
        raise ValueError(f"unsafe table_prefix {table_prefix!r}")
    name = _validate_model_name(model_name)
    meta_table = f"{table_prefix}ORACLEAGENTMEMORY_SCHEMA_META"
    with connection.cursor() as cur:
        # If the meta table doesn't exist yet, there is no schema to migrate.
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :t",
            t=meta_table.upper(),
        )
        if cur.fetchone()[0] == 0:
            return False
        # Recorded dimension (absent row -> nothing to compare).
        cur.execute(
            f"SELECT metadata_value FROM {meta_table} "
            "WHERE metadata_key = 'vector_dim'"
        )
        row = cur.fetchone()
        if row is None:
            return False
        recorded_dim = int(row[0])
        # Live model dimension via a one-row probe embedding.
        cur.execute(
            f"SELECT VECTOR_EMBEDDING({name} USING 'x' AS DATA) FROM dual"
        )
        model_dim = len(list(cur.fetchone()[0]))
        if recorded_dim == model_dim:
            return False
        # Mismatch -> drop the prefixed agent-memory tables (regenerable).
        # Escape '_' so it is a literal, not a single-char LIKE wildcard.
        like_prefix = table_prefix.upper().replace("_", r"\_")
        cur.execute(
            r"SELECT table_name FROM user_tables "
            r"WHERE table_name LIKE :p ESCAPE '\'",
            p=f"{like_prefix}%",
        )
        tables = [r[0] for r in cur.fetchall()]
        for t in tables:
            cur.execute(f'DROP TABLE "{t}" CASCADE CONSTRAINTS PURGE')
    return True


__all__ = ["OnnxDBEmbedder", "ensure_onnx_model", "_validate_model_name", "reset_schema_if_dim_changed"]
