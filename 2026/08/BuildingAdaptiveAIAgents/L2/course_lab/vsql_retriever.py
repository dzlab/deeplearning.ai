"""course_lab/vsql_retriever.py — vSQL templates indexed in Oracle via OracleVS.

Module 2 ships two artifacts: a JSONL of mined templates (file-on-disk, kept
for backwards compat) and an OracleVS-backed vector index of those same
templates (Oracle table, queried at task time). Eval coverage is now decided
by real cosine similarity against the index, not a substring keyword match.

The embedder is built from a config spec via
:func:`course_lab.python_embedder.make_embedder` — the out-of-DB
:class:`~course_lab.python_embedder.PythonMiniLMEmbedder` (``python:<model>``,
the default sandbox path) or the in-DB
:class:`~course_lab.onnx_embedder.OnnxDBEmbedder` (``onnx:<MODEL>`` reference) —
wrapped here as a LangChain ``Embeddings`` subclass so
``langchain_oracledb.OracleVS`` can consume it directly.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import DistanceStrategy

from course_lab.python_embedder import make_embedder

# Default embedder spec for the OracleVS builders: out-of-DB MiniLM. Override
# via the ``model_spec`` argument to use the in-DB ONNX reference path.
DEFAULT_EMBEDDER_SPEC = "python:all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# LangChain ``Embeddings`` adapter for a duck-typed ``.embed()`` embedder.
# ---------------------------------------------------------------------------


class EmbedderLcEmbeddings(Embeddings):
    """Adapter exposing any ``.embed(texts)->ndarray`` embedder as a LangChain
    ``Embeddings``.

    The wrapped ``.embed`` returns an ``np.ndarray`` of shape ``(n, dim)``;
    ``Embeddings`` expects plain Python lists. We convert at the boundary and
    forward both ``embed_documents`` and ``embed_query`` to the same call —
    MiniLM has no query/document asymmetry.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        arr = self._embedder.embed(list(texts), is_query=False)
        if arr.size == 0:
            return []
        return [row.astype(np.float32).tolist() for row in arr]

    def embed_query(self, text: str) -> list[float]:
        arr = self._embedder.embed([text], is_query=True)
        return arr[0].astype(np.float32).tolist()


# Back-compat alias: the adapter is no longer ONNX-specific.
OnnxLcEmbeddings = EmbedderLcEmbeddings


# ---------------------------------------------------------------------------
# OracleVS builder for vSQL templates.
# ---------------------------------------------------------------------------


def build_vsql_oraclevs(
    *,
    connection: Any,
    templates: list[dict],
    table_name: str = "DLAI_VSQL_TEMPLATES",
    model_spec: str = DEFAULT_EMBEDDER_SPEC,
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
) -> OracleVS:
    """Index ``templates`` into Oracle via :class:`OracleVS` and return it.

    Each template becomes a :class:`Document` with::

        page_content = template["sql_pattern"]
        metadata     = {"template_id": ..., "support_count": ..., "avg_latency_ms": ...}

    The table is dropped and re-created so the index always reflects the
    current run's mined templates (no stale rows leaking across runs).
    """
    if connection is None:
        raise RuntimeError("build_vsql_oraclevs requires a live oracledb connection")
    if not templates:
        raise ValueError(
            "build_vsql_oraclevs requires at least one template; the mine "
            "stage is expected to emit ≥ 1."
        )

    # Drop the table if it exists so we start clean each run. OracleVS does not
    # expose a "replace" mode, and additive writes would mix prior runs.
    with connection.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM user_tables WHERE table_name = :t",
            t=table_name.upper(),
        )
        if cur.fetchone()[0] > 0:
            cur.execute(f'DROP TABLE "{table_name.upper()}" PURGE')

    embeddings = EmbedderLcEmbeddings(make_embedder(model_spec, connection=connection))
    documents = [
        Document(
            page_content=str(t.get("sql_pattern", "")),
            metadata={
                "template_id": str(t.get("template_id", "")),
                "support_count": int(t.get("support_count", 0)),
                "avg_latency_ms": float(t.get("avg_latency_ms", 0.0)),
            },
        )
        for t in templates
    ]
    # Use template_id as the document id so the embeddings table is keyed by
    # the same identifier that downstream callers reference.
    ids = [d.metadata["template_id"] for d in documents]
    store = OracleVS.from_documents(
        documents=documents,
        embedding=embeddings,
        client=connection,
        table_name=table_name,
        distance_strategy=distance_strategy,
        ids=ids,
    )
    return store


def coverage_score(
    store: OracleVS,
    tasks: list[str],
    *,
    k: int = 2,
    max_distance: float = 0.70,
) -> dict:
    """Score eval-task coverage by retrieving top-k templates per task.

    ``OracleVS.similarity_search_with_score`` returns cosine *distance*
    (``0.0`` = identical direction, larger = more dissimilar). A task is
    "covered" when the top-k retrieval contains at least one document whose
    distance is <= ``max_distance``.
    """
    if not tasks:
        return {
            "baseline": 0.0,
            "with_skills": 0.0,
            "lift": 0.0,
            "n_eval_tasks": 0,
            "n_covered": 0,
        }

    covered = 0
    for task in tasks:
        hits = store.similarity_search_with_score(task, k=k)
        if any(score <= max_distance for _, score in hits):
            covered += 1
    baseline = 0.0
    with_skills = covered / len(tasks)
    return {
        "baseline": round(baseline, 4),
        "with_skills": round(with_skills, 4),
        "lift": round(with_skills - baseline, 4),
        "n_eval_tasks": len(tasks),
        "n_covered": covered,
    }


__all__ = [
    "EmbedderLcEmbeddings",
    "OnnxLcEmbeddings",
    "DEFAULT_EMBEDDER_SPEC",
    "build_vsql_oraclevs",
    "coverage_score",
]
