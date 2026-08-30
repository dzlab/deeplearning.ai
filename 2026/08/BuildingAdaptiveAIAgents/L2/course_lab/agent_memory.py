"""course_lab/agent_memory.py — wrapper over ``oracleagentmemory`` v26.4.0.

Design = "mirror real surface verbatim" (locked by spec §4).  The wrapper
returns a ``Thread`` handle; conversational verbs hang off it
(``add_messages``, ``add_memory``, ``get_context_card``).  Course-extra
verbs (``write_skill``, ``record_query_trace``, ``record_retrieval_trace``)
are flat on the top-level :class:`AgentMemory`.

The course is real-only: :class:`AgentMemory` is always backed by Oracle AI
Database.  The ``oracleagentmemory.OracleAgentMemory`` class does not expose a
public connection-pool accessor, so this module accepts an optional
``connection_pool`` constructor argument (``AgentMemory.from_config`` builds one
via :mod:`course_lab.oracle_db`) and threads it through to course-extra writes
via plain ``oracledb``.
"""
from __future__ import annotations

import array
import hashlib
import json
import struct
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, NamedTuple, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from course_lab import oracle_db


class MemoryBackendUnavailable(RuntimeError):
    """Raised when the Oracle memory backend cannot start."""


# ---------------------------------------------------------------------------
# Pydantic models — course-extra rows only.  The package owns the
# conversational / durable-memory layer; nothing in this section overlaps
# with ``OracleAgentMemory``'s native row types.
# ---------------------------------------------------------------------------


class SkillRow(BaseModel):
    skill_id: str
    name: str
    description: str
    recipe_steps: list[str]
    provenance: list[str]
    embedding: list[float] | None = None
    ts: datetime
    status: str = "pending"          # "pending" | "active" | "rejected"
    source: str | None = None        # "structured" | "conversational"
    created_day: int | None = None   # dream-loop day tag
    promoted: bool = False           # an approved enhanced skill superseded this one


class EnhancedSkillRow(BaseModel):
    skill_id: str
    topic: str
    name: str
    description: str
    when_to_use: str
    steps: list[str]
    skills_used: list[str]
    likely_tools: list[str]
    errors_and_fixes: list[dict]     # [{"error": ..., "fix": ...}]
    provenance: list[str]
    embedding: list[float] | None = None
    ts: datetime | None = None
    status: str = "pending"          # pending | active | rejected | superseded
    source: str = "unified"
    version: int = 1
    created_day: int | None = None
    review_comment: str | None = None


class SkillHit(NamedTuple):
    """One dual-retrieval result: which store, the row, the cosine distance."""
    store: str                       # "base" | "enhanced"
    skill: BaseModel                 # SkillRow | EnhancedSkillRow
    distance: float


class QueryTraceRow(BaseModel):
    trace_id: str
    sql: str
    latency_ms: int
    success: bool
    ts: datetime
    # Optional intent label (e.g. ``"find_vessels_by_destination"``). Populated
    # when callers know the natural-language intent behind the SQL; consumed by
    # :func:`course_lab.skill_induction.induce_skills` to group successful
    # traces and synthesize procedural skills.
    intent: str | None = None


class RetrievalTraceRow(BaseModel):
    trace_id: str
    query: str
    candidates: list[dict]
    chosen: list[str]
    outcome: str
    ts: datetime


# ---------------------------------------------------------------------------
# Deterministic hash embedder — SHA256(text) → 384-d float vector in [-1, 1].
# No network, no model download, no ``sentence_transformers`` import.
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic SHA256-based embedder.

    Tiles SHA256(text) (32 bytes → 8 float32s) by repeated re-hashing of the
    digest until we reach ``dim`` floats, then scales to ``[-1, 1]``.
    """

    dim: int = 384

    def embed(self, text: str) -> list[float]:
        floats: list[float] = []
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        while len(floats) < self.dim:
            # 32-byte digest unpacks to 8 float32s; rehash to get more.
            chunk = struct.unpack("8f", digest)
            floats.extend(chunk)
            digest = hashlib.sha256(digest).digest()
        floats = floats[: self.dim]
        # Squash to [-1, 1] via tanh-equivalent normalisation; raw float32s
        # decoded from random bytes can land anywhere on the real line.
        arr = np.asarray(floats, dtype=np.float64)
        # Guard against NaN/inf from pathological bit patterns.
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        arr = np.tanh(arr)
        return arr.astype(np.float32).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class VecSkillEmbedder:
    """Adapt a duck-typed ``.embed(texts)->ndarray`` embedder to the course-extra
    skill-box embedder interface.

    The skill box needs SEMANTIC retrieval (so an enhanced skill out-ranks a
    weaker base one even when the wording differs), which the token-hash
    :class:`HashEmbedder` cannot give. This wraps either the in-DB ONNX embedder
    (``VECTOR_EMBEDDING`` inside Oracle) or the out-of-DB
    :class:`~course_lab.python_embedder.PythonMiniLMEmbedder` to present
    ``.embed(text) -> list`` / ``.embed_many(texts) -> list[list]``, the shape
    SKILLBOX writes and searches expect. Same 384-dim space as the VECTOR(384)
    embedding columns.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def embed(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0].tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [v.tolist() for v in self._embedder.embed(list(texts))]


# Back-compat alias: the wrapper is no longer ONNX-specific.
OnnxSkillEmbedder = VecSkillEmbedder


# ---------------------------------------------------------------------------
# Thread protocol — the real ``OracleThread`` satisfies it structurally.
# ---------------------------------------------------------------------------


@runtime_checkable
class Thread(Protocol):
    thread_id: str

    def add_messages(self, messages: list[dict]) -> Any: ...
    def add_memory(self, text: str, *, metadata: dict | None = None) -> Any: ...
    def get_context_card(self) -> str: ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _skill_search_sql(*, k: int, status: str | None) -> str:
    """Build the SKILLBOX vector-search SQL.

    Column order matches ``_real_list_skills`` exactly so ``_row_to_skill``
    can be shared between both callers without index remapping.

    Args:
        k: maximum number of rows to return.
        status: when not None, restrict to rows with this status value.

    Returns:
        SQL string ready for ``cur.execute``.
    """
    where = "WHERE status = :status\n        " if status is not None else ""
    return (
        "SELECT skill_id, name, description, recipe_steps, provenance, "
        "embedding, created_ts, status, source, created_day, promoted\n"
        "FROM SKILLBOX\n        " + where +
        "ORDER BY VECTOR_DISTANCE(embedding, :qvec, COSINE)\n"
        f"FETCH FIRST {int(k)} ROWS ONLY"
    )


_ENHANCED_COLS = ("skill_id, topic, name, description, when_to_use, steps, "
                  "skills_used, likely_tools, errors_and_fixes, provenance, "
                  "embedding, created_ts, status, source, version, created_day, "
                  "review_comment")


def _enhanced_search_sql(*, k: int, status: str | None) -> str:
    where = "WHERE status = :status\n        " if status is not None else ""
    return (
        f"SELECT {_ENHANCED_COLS}, "
        "VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist\n"
        "FROM ENHANCED_SKILLBOX\n        " + where +
        "ORDER BY VECTOR_DISTANCE(embedding, :qvec, COSINE)\n"
        f"FETCH FIRST {int(k)} ROWS ONLY"
    )


def _base_search_dist_sql(*, k: int, status: str | None) -> str:
    where = "WHERE status = :status\n        " if status is not None else ""
    return (
        "SELECT skill_id, name, description, recipe_steps, provenance, "
        "embedding, created_ts, status, source, created_day, promoted, "
        "VECTOR_DISTANCE(embedding, :qvec, COSINE) AS dist\n"
        "FROM SKILLBOX\n        " + where +
        "ORDER BY VECTOR_DISTANCE(embedding, :qvec, COSINE)\n"
        f"FETCH FIRST {int(k)} ROWS ONLY"
    )


def _new_id() -> str:
    return uuid.uuid4().hex


def _is_not_set(value: Any) -> bool:
    """Detect the ``oracleagentmemory._notset._NotSetMarker`` sentinel.

    We avoid importing the private symbol; the marker's class name is
    distinctive enough to identify by ``__class__.__name__``.
    """
    return type(value).__name__ == "_NotSetMarker"


# ---------------------------------------------------------------------------
# AgentMemory — public wrapper.  Same surface against either client.
# ---------------------------------------------------------------------------


class AgentMemory:
    """Wrapper over a real ``OracleAgentMemory`` client.

    The constructor takes an already-built ``client`` plus an ``embedder``
    used for course-extra writes.  The wrapper uses :class:`HashEmbedder` for
    course-extra rows by default — keeps the cost zero and the behaviour
    deterministic.  Callers may swap the embedder by passing one explicitly.
    """

    def __init__(
        self,
        client: Any,
        *,
        embedder: Any | None = None,
        connection_pool: Any = None,
        table_prefix: str = "DLAI_",
    ) -> None:
        self._client = client
        self._embedder = embedder if embedder is not None else HashEmbedder()
        # ``connection_pool`` is used for course-extra writes
        # (SKILLBOX / QUERY_TRACES / RETRIEVAL_TRACES).
        self._pool = connection_pool
        self._table_prefix = table_prefix

    # ------------------------------------------------------------------
    # Factories.
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "AgentMemory":
        """Build an ``AgentMemory`` backed by Oracle AI Database.

        Recognised keys: ``embedder_model`` (str | None), ``llm_model``
        (str | None), ``table_prefix`` (str), ``embedder_kwargs``,
        ``llm_kwargs``. Raises :class:`MemoryBackendUnavailable` if Oracle is
        not reachable — the course is real-only (no fake/offline backend).
        """
        try:
            return cls._from_real(cfg)
        except MemoryBackendUnavailable:
            raise
        except Exception as exc:
            raise MemoryBackendUnavailable(
                f"Oracle memory backend unavailable: {exc}"
            ) from exc

    @classmethod
    def _from_real(cls, cfg: dict) -> "AgentMemory":
        conn = oracle_db.get_connection(autocommit=True)
        course_extra_connections = oracle_db.DirectConnectionProvider()
        # Real path — lazy-import so fake/offline tests never hit litellm/oracledb.
        from oracleagentmemory.core import OracleAgentMemory, SchemaPolicy
        from oracleagentmemory.core.embedders.embedder import Embedder
        from oracleagentmemory.core.llms.llm import Llm

        embedder_model = cfg.get("embedder_model", "text-embedding-3-small")
        llm_model = cfg.get("llm_model", "gpt-4o-mini")
        embedder_kwargs = dict(cfg.get("embedder_kwargs", {}))
        llm_kwargs = dict(cfg.get("llm_kwargs", {}))

        # Out-of-DB ("python:<MODEL>") or in-DB ("onnx:<ORACLE_MODEL_NAME>")
        # MiniLM, both 384-d / L2-normed / no OCI / no litellm. python: is the
        # default sandbox path (the model runs in this process, so the DB never
        # OOMs running ONNX inference); onnx: is the retained in-DB reference.
        # The LLM path is untouched (it may still be an OCI model for extraction).
        if str(embedder_model).startswith(("python:", "onnx:")):
            from course_lab.python_embedder import make_embedder

            embedder_obj: object = make_embedder(str(embedder_model), connection=conn)
            if str(llm_model).startswith("oci/"):
                from course_lab.oci_client import _load_oci_auth_params

                llm_kwargs = {**_load_oci_auth_params(), **llm_kwargs}
        else:
            if str(embedder_model).startswith("oci/") or str(llm_model).startswith("oci/"):
                from course_lab.oci_client import _load_oci_auth_params

                oci_auth = _load_oci_auth_params()
                if str(embedder_model).startswith("oci/"):
                    embedder_kwargs = {**oci_auth, **embedder_kwargs}
                if str(llm_model).startswith("oci/"):
                    llm_kwargs = {**oci_auth, **llm_kwargs}
            embedder_obj = Embedder(model=embedder_model, **embedder_kwargs)

        table_prefix = cfg.get("table_prefix", "DLAI_")
        client = OracleAgentMemory(
            connection=conn,
            embedder=embedder_obj,
            llm=Llm(model=llm_model, **llm_kwargs),
            schema_policy=SchemaPolicy.CREATE_IF_NECESSARY,
            table_name_prefix=table_prefix,
        )
        # Course-extra (SKILLBOX / ENHANCED_SKILLBOX) embedder. Default is the
        # zero-cost HashEmbedder. Opt in to semantic skill retrieval with
        # ``skill_embedder: "python:<MODEL>"`` (or the in-DB ``onnx:<MODEL>``
        # reference) — Module 2 uses this so the Step 5 evaluator shows a real
        # distance win, not a token-hash coincidence.
        skill_embedder_cfg = cfg.get("skill_embedder")
        if skill_embedder_cfg and str(skill_embedder_cfg).startswith(("python:", "onnx:")):
            from course_lab.python_embedder import make_embedder

            skill_embedder: Any = VecSkillEmbedder(
                make_embedder(str(skill_embedder_cfg), connection=conn)
            )
        else:
            skill_embedder = HashEmbedder()

        return cls(
            client,
            embedder=skill_embedder,
            connection_pool=course_extra_connections,
            table_prefix=table_prefix,
        )

    @classmethod
    def from_env(cls) -> "AgentMemory":
        """Build an Oracle-backed ``AgentMemory`` from environment defaults."""
        return cls.from_config({})

    # ------------------------------------------------------------------
    # Pass-through verbs.
    # ------------------------------------------------------------------

    def create_thread(self, user_id: str, agent_id: str = "default") -> Thread:
        # The real OracleAgentMemory.create_thread does not accept agent_id
        # (per the documented surface in .omc/autopilot/oracleagentmemory-api.md
        # §4); the parameter is retained on the wrapper for caller compatibility.
        return self._client.create_thread(user_id=user_id)

    def get_thread(self, thread_id: str) -> Thread:
        return self._client.get_thread(thread_id)

    def search(self, query: str, *, scope: Any = None, k: int = 10) -> list:
        # The real oracleagentmemory search API requires an explicit scope
        # decision; ``user_id=None`` means an intentional cross-user/course-wide
        # search, which is what the workshop notebooks need for mining
        # previously seeded interactions.
        if scope is None:
            return self._client.search(query=query, user_id=None, max_results=k)
        return self._client.search(query=query, scope=scope, max_results=k)

    def list_memories(self, scope: Any = None) -> list:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if scope is not None:
            for attr, col in (("user_id", "USER_ID"),
                              ("agent_id", "AGENT_ID"),
                              ("thread_id", "THREAD_ID")):
                val = getattr(scope, attr, None)
                if val is None or _is_not_set(val):
                    continue
                key = attr
                clauses.append(f"{col} = :{key}")
                params[key] = val
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        table = self._oracle_table("MEMORY")
        sql = (
            f"SELECT record_id, thread_id, user_id, agent_id, content, "
            f"metadata, timestamp FROM {table} {where} ORDER BY created_at ASC"
        )
        with self._cursor() as cur:
            cur.execute(sql, params)
            out: list[dict] = []
            for r in cur.fetchall():
                out.append({
                    "memory_id": r[0],
                    "thread_id": r[1],
                    "user_id": r[2],
                    "agent_id": r[3],
                    "text": self._db_value(r[4]),
                    "metadata": self._json_db_value(r[5]),
                    "ts": self._db_value(r[6]),
                })
            return out

    def delete(self, record_id: str) -> None:
        # OracleAgentMemory exposes ``delete_memory``.
        self._client.delete_memory(record_id)

    def _oracle_table(self, suffix: str) -> str:
        return f"{self._table_prefix}{suffix}".upper()

    def _require_pool(self) -> Any:
        """Return the course-extra connection provider or raise.

        Course-extra reads/writes (SKILLBOX / QUERY_TRACES / RETRIEVAL_TRACES /
        graph tables) need the ``connection_pool`` passed at construction. A
        wrapper built without one (e.g. a fake/offline stub) raises here.
        """
        if self._pool is None:
            raise RuntimeError(
                "real-path course-extra access requires a connection_pool"
            )
        return self._pool

    @contextmanager
    def _cursor(self, *, commit: bool = False) -> Iterator[Any]:
        """Acquire a connection from the pool and yield a cursor.

        Collapses the repeated ``with self._pool.acquire() as conn:
        cur = conn.cursor()`` ceremony. When ``commit=True`` the connection is
        committed after the block completes, matching the prior per-method
        ``conn.commit()`` placement (inside the acquired-connection scope).
        """
        pool = self._require_pool()
        with pool.acquire() as conn:
            yield conn.cursor()
            if commit:
                conn.commit()

    @staticmethod
    def _db_value(value: Any) -> Any:
        return value.read() if hasattr(value, "read") else value

    @classmethod
    def _json_db_value(cls, value: Any) -> Any:
        value = cls._db_value(value)
        if value is None or isinstance(value, (dict, list)):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        if isinstance(value, str):
            return json.loads(value) if value else None
        return value

    # ------------------------------------------------------------------
    # Course-extra verbs.  These hit oracledb via the connection pool.
    # ------------------------------------------------------------------

    def write_skill(self, name: str, description: str,
                    recipe_steps: list[str],
                    provenance: list[str],
                    *, status: str = "pending",
                    source: str | None = None,
                    created_day: int | None = None,
                    promoted: bool = False) -> SkillRow:
        skill_id = _new_id()
        ts = _now()
        embedding = self._embedder.embed(description)
        row = SkillRow(
            skill_id=skill_id, name=name, description=description,
            recipe_steps=recipe_steps, provenance=provenance,
            embedding=embedding, ts=ts,
            status=status, source=source, created_day=created_day,
            promoted=promoted,
        )
        self._real_insert_skill(row)
        return row

    def list_skills(self, limit: int | None = None, *, status: str | None = None) -> list[SkillRow]:
        rows = self._real_list_skills(limit=None)
        if status is not None:
            rows = [r for r in rows if r.status == status]
        return rows[:limit] if limit else rows

    def set_skill_status(self, skill_id: str, status: str) -> None:
        if self._pool is None:
            raise RuntimeError("real-path write requires a connection_pool")
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE SKILLBOX SET status = :s WHERE skill_id = :id",
                        {"s": status, "id": skill_id})
            conn.commit()

    def set_skill_promoted(self, skill_id: str, promoted: bool = True) -> None:
        """Flip a standard skill's ``promoted`` flag.

        A standard skill is promoted once an approved enhanced skill supersedes
        it for the same topic; promoted standard skills are excluded at index
        time so the agent never sees two recipes for one job.
        """
        if self._pool is None:
            raise RuntimeError("real-path write requires a connection_pool")
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE SKILLBOX SET promoted = :p WHERE skill_id = :id",
                        {"p": 1 if promoted else 0, "id": skill_id})
            conn.commit()

    def truncate_skills(self) -> None:
        """Truncate the course-extra SKILLBOX table.

        Per-run hygiene: SKILLBOX is a SINGLE GLOBAL Oracle table — the
        unqualified ``INSERT INTO SKILLBOX`` in :meth:`_real_insert_skill`
        means ``table_prefix`` does NOT isolate it. Without an explicit
        truncate at the start of an induction pass every Module 2 run inherits
        every previously-induced skill (including stale boilerplate from prior
        mechanism bugs), so the lift eval mixes the fresh induction with
        accumulated rubbish. Callers that own the SKILLBOX lifecycle (e.g.
        ``module_2_token_space/scripts/run.py``) should call this before
        :func:`course_lab.skill_induction.induce_skills`.
        """
        with self._cursor(commit=True) as cur:
            cur.execute("TRUNCATE TABLE SKILLBOX")

    def truncate_enhanced_skills(self) -> None:
        """Truncate the course-extra ENHANCED_SKILLBOX table.

        Same single-global-table caveat as :meth:`truncate_skills`: callers that
        own the lifecycle should clear it before a fresh induction pass so stale
        induced skills from prior runs do not pollute retrieval.
        """
        with self._cursor(commit=True) as cur:
            cur.execute("TRUNCATE TABLE ENHANCED_SKILLBOX")

    def upsert_graph_nodes(self, nodes: list[dict], *, domain: str) -> int:
        """MERGE graph nodes into MEMORY_GRAPH_NODES (idempotent on (id, domain))."""
        self._require_pool()
        if not nodes:
            return 0
        sql = (
            "MERGE INTO MEMORY_GRAPH_NODES t "
            "USING (SELECT :id AS id, :domain AS domain FROM dual) s "
            "ON (t.id = s.id AND t.domain = s.domain) "
            "WHEN MATCHED THEN UPDATE SET t.text = :text, "
            "  t.updated_ts = CURRENT_TIMESTAMP "
            "WHEN NOT MATCHED THEN INSERT (id, domain, text) "
            "  VALUES (:id, :domain, :text)"
        )
        rows = [{"id": n["id"], "domain": domain, "text": n.get("text", "")}
                for n in nodes]
        with self._cursor(commit=True) as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def upsert_graph_edges(self, edges: list[tuple], *, domain: str) -> int:
        """MERGE edges into MEMORY_GRAPH_EDGES (idempotent on (src,dst,kind,domain)).

        ``edges`` is a list of (src, dst, kind) tuples; weight defaults to 1.0
        and a re-seen edge bumps its weight + timestamp rather than duplicating.
        """
        self._require_pool()
        if not edges:
            return 0
        sql = (
            "MERGE INTO MEMORY_GRAPH_EDGES t "
            "USING (SELECT :src AS src, :dst AS dst, :kind AS kind, "
            "       :domain AS domain FROM dual) s "
            "ON (t.src = s.src AND t.dst = s.dst AND t.kind = s.kind "
            "    AND t.domain = s.domain) "
            "WHEN MATCHED THEN UPDATE SET t.weight = t.weight + 1.0, "
            "  t.updated_ts = CURRENT_TIMESTAMP "
            "WHEN NOT MATCHED THEN INSERT (src, dst, kind, weight, domain) "
            "  VALUES (:src, :dst, :kind, 1.0, :domain)"
        )
        rows = [{"src": e[0], "dst": e[1],
                 "kind": e[2] if len(e) > 2 else "import",
                 "domain": domain}
                for e in edges]
        with self._cursor(commit=True) as cur:
            cur.executemany(sql, rows)
        return len(rows)

    def load_graph(self, *, domain: str) -> tuple[list[dict], MemGraph]:
        """Read nodes + edges for ``domain`` back from Oracle; rebuild a MemGraph.

        Returns (nodes, graph). This is the READ side of the autonomous loop:
        the tick upserts new rows, then the notebook calls load_graph to prove
        retrieval improves over the DB-sourced graph (not an in-memory object).
        """
        from course_lab.memory_graph import MemGraph
        nodes: list[dict] = []
        graph = MemGraph()
        with self._cursor() as cur:
            cur.execute(
                "SELECT id, text FROM MEMORY_GRAPH_NODES WHERE domain = :d "
                "ORDER BY id", {"d": domain})
            for r in cur.fetchall():
                nid = r[0]
                nodes.append({"id": nid, "text": self._db_value(r[1]) or ""})
                graph.add_node(nid)
            # kind is selected for schema parity; MemGraph carries no edge labels.
            cur.execute(
                "SELECT src, dst, kind, weight FROM MEMORY_GRAPH_EDGES "
                "WHERE domain = :d", {"d": domain})
            for r in cur.fetchall():
                graph.add_edge(r[0], r[1], w=float(r[3] or 1.0))
        return nodes, graph

    def clear_graph_domain(self, *, domain: str) -> None:
        """Delete all persisted graph nodes + edges for ``domain``.

        Used to reset a demo/scratch domain so the autonomous-loop tick is
        reproducible (a stale prior run must not leave the 'new' symbol already
        in the DB). Does NOT touch other domains — prior real edges are safe.
        """
        with self._cursor(commit=True) as cur:
            cur.execute("DELETE FROM MEMORY_GRAPH_EDGES WHERE domain = :d", {"d": domain})
            cur.execute("DELETE FROM MEMORY_GRAPH_NODES WHERE domain = :d", {"d": domain})

    def record_query_trace(self, sql: str, latency_ms: int,
                           success: bool,
                           intent: str | None = None) -> QueryTraceRow:
        trace_id = _new_id()
        ts = _now()
        row = QueryTraceRow(
            trace_id=trace_id,
            sql=sql,
            latency_ms=latency_ms,
            success=success,
            ts=ts,
            intent=intent,
        )
        self._real_insert_query_trace(row)
        return row

    def record_retrieval_trace(self, query: str, candidates: list[dict],
                               chosen: list[str],
                               outcome: str) -> RetrievalTraceRow:
        trace_id = _new_id()
        ts = _now()
        # Embedding is computed but not stored on the row by default; future
        # callers can attach it for similarity lookups over traces.
        _ = self._embedder.embed(query)
        row = RetrievalTraceRow(
            trace_id=trace_id,
            query=query,
            candidates=candidates,
            chosen=chosen,
            outcome=outcome,
            ts=ts,
        )
        self._real_insert_retrieval_trace(row)
        return row

    def list_traces(self, kind: str, limit: int | None = None) -> list:
        if kind not in {"query", "retrieval"}:
            raise ValueError(f"kind must be 'query' or 'retrieval', got {kind!r}")
        return self._real_list_traces(kind, limit)

    def list_episodic_user_turns(self, *, week_cutoff: int) -> list[dict]:
        """Return episodic (user, assistant) turns from Oracle message tables up to a week boundary.

        Queries the ``{prefix}MESSAGE`` table directly via the connection pool,
        grouping consecutive user→assistant message pairs per thread.  The
        ``week`` field is derived from the ``{prefix}THREAD`` row's ``metadata``
        JSON when present (``metadata.week``), defaulting to 1 if absent, so
        the M4 miner can still operate on pre-existing memory without schema
        changes.

        If the connection pool is not available (e.g. unit tests using the fake
        stub), returns an empty list rather than raising.
        """
        out: list[dict] = []
        if self._pool is None:
            return out
        try:
            thread_table = self._oracle_table("THREAD")
            message_table = self._oracle_table("MESSAGE")
            # Fetch all threads with their metadata so we can derive week.
            with self._pool.acquire() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT record_id, metadata FROM {thread_table}"
                )
                thread_rows = cur.fetchall()
        except Exception:
            return out

        for thread_row in thread_rows:
            thread_id = thread_row[0]
            # Derive week from thread metadata JSON.
            week = 1
            raw_meta = thread_row[1]
            try:
                if raw_meta is not None:
                    if hasattr(raw_meta, "read"):
                        raw_meta = raw_meta.read()
                    if isinstance(raw_meta, (bytes, bytearray)):
                        raw_meta = raw_meta.decode("utf-8")
                    if isinstance(raw_meta, str):
                        md = json.loads(raw_meta) if raw_meta else {}
                    else:
                        md = raw_meta if isinstance(raw_meta, dict) else {}
                    week = int(md.get("week", 1))
            except (TypeError, ValueError, json.JSONDecodeError):
                week = 1

            if week > week_cutoff:
                continue

            # Fetch messages for this thread in order.
            try:
                with self._pool.acquire() as conn:
                    cur = conn.cursor()
                    cur.execute(
                        f"SELECT message_role, content FROM {message_table} "
                        f"WHERE thread_id = :tid ORDER BY order_seq",
                        {"tid": thread_id},
                    )
                    msgs = cur.fetchall()
            except Exception:
                continue

            user_pending = None
            for msg_role, msg_content in msgs:
                role = msg_role or ""
                content = msg_content
                if hasattr(content, "read"):
                    content = content.read()
                if isinstance(content, (bytes, bytearray)):
                    content = content.decode("utf-8")
                content = content or ""
                if role == "user":
                    user_pending = content
                elif role == "assistant" and user_pending is not None:
                    out.append({
                        "user": user_pending,
                        "assistant": content,
                        "thread_id": thread_id,
                        "week": week,
                    })
                    user_pending = None
        return out

    def close(self) -> None:
        # Close the client first, then the pool we hold.
        if hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass
        if self._pool is not None:
            try:
                self._pool.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Real-path implementations — course-extra reads/writes over oracledb.
    # ------------------------------------------------------------------

    def _real_insert_skill(self, row: SkillRow) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO SKILLBOX "
                "(skill_id, name, description, recipe_steps, provenance, "
                " embedding, created_ts, status, source, created_day, promoted) "
                "VALUES (:sid, :n, :d, :r, :p, :e, :t, :st, :src, :cd, :pr)",
                {"sid": row.skill_id,
                 "n": row.name,
                 "d": row.description,
                 "r": json.dumps(row.recipe_steps),
                 "p": json.dumps(row.provenance),
                 "e": array.array("f", row.embedding or []),
                 "t": row.ts,
                 "st": row.status,
                 "src": row.source,
                 "cd": row.created_day,
                 "pr": 1 if row.promoted else 0},
            )

    def _row_to_skill(self, r: Any) -> SkillRow:
        """Build a :class:`SkillRow` from a SKILLBOX cursor row.

        Column positions (0-indexed) match the SELECT in both
        ``_real_list_skills`` and ``_skill_search_sql``::

            0 skill_id  1 name  2 description  3 recipe_steps  4 provenance
            5 embedding  6 created_ts  7 status  8 source  9 created_day
            10 promoted

        CLOB columns (description, recipe_steps, provenance) are read via
        ``.read()`` when Oracle returns a LOB handle; ``created_day`` is
        cast to ``int`` only when non-None; ``promoted`` (NUMBER(1)) maps to a
        bool, defaulting to False when the column is NULL.
        """
        return SkillRow(
            skill_id=r[0],
            name=r[1],
            description=r[2].read() if hasattr(r[2], "read") else r[2],
            recipe_steps=json.loads(r[3].read() if hasattr(r[3], "read") else r[3]),
            provenance=json.loads(r[4].read() if hasattr(r[4], "read") else r[4]),
            embedding=list(r[5]) if r[5] is not None else None,
            ts=r[6],
            status=r[7] if r[7] is not None else "pending",
            source=r[8],
            created_day=int(r[9]) if r[9] is not None else None,
            promoted=bool(r[10]) if len(r) > 10 and r[10] is not None else False,
        )

    def _real_list_skills(self, limit: int | None) -> list[SkillRow]:
        sql = ("SELECT skill_id, name, description, recipe_steps, provenance, "
               "embedding, created_ts, status, source, created_day, promoted "
               "FROM SKILLBOX ORDER BY created_ts ASC")
        if limit is not None:
            sql += f" FETCH FIRST {int(limit)} ROWS ONLY"
        with self._cursor() as cur:
            cur.execute(sql)
            return [self._row_to_skill(r) for r in cur.fetchall()]

    def search_skills(self, query: str, k: int = 3, *, status: str | None = "active") -> list[SkillRow]:
        """Return the ``k`` skills whose embeddings are nearest to ``query``.

        Executes an HNSW cosine vector search against SKILLBOX using
        ``VECTOR_DISTANCE``. When ``status`` is not None (the default is
        ``"active"``), only rows with a matching status are considered.

        Args:
            query: natural-language description to embed and search against.
            k: number of nearest neighbours to return.
            status: restrict candidates to this status value; pass ``None``
                to search across all statuses.

        Returns:
            List of :class:`SkillRow` ordered nearest-first (ascending cosine
            distance), length at most ``k``.

        Raises:
            RuntimeError: if no connection pool is available.
        """
        if self._pool is None:
            raise RuntimeError("real-path read requires a connection_pool")
        qvec = array.array("f", self._embedder.embed(query))
        binds: dict[str, Any] = {"qvec": qvec}
        if status is not None:
            binds["status"] = status
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(_skill_search_sql(k=k, status=status), binds)
            return [self._row_to_skill(r) for r in cur.fetchall()]

    @staticmethod
    def _clob(v):
        return v.read() if hasattr(v, "read") else v

    def _row_to_enhanced_skill(self, r) -> EnhancedSkillRow:
        # Column order matches _ENHANCED_COLS exactly.
        return EnhancedSkillRow(
            skill_id=r[0], topic=r[1], name=r[2],
            description=self._clob(r[3]), when_to_use=self._clob(r[4]),
            steps=json.loads(self._clob(r[5])),
            skills_used=json.loads(self._clob(r[6])),
            likely_tools=json.loads(self._clob(r[7])),
            errors_and_fixes=json.loads(self._clob(r[8])),
            provenance=json.loads(self._clob(r[9])),
            embedding=list(r[10]) if r[10] is not None else None,
            ts=r[11],
            status=r[12] if r[12] is not None else "pending",
            source=r[13] if r[13] is not None else "unified",
            version=int(r[14]) if r[14] is not None else 1,
            created_day=int(r[15]) if r[15] is not None else None,
            review_comment=self._clob(r[16]) if r[16] is not None else None,
        )

    def write_enhanced_skill(self, *, topic: str, name: str, description: str,
                             when_to_use: str, steps: list[str],
                             skills_used: list[str], likely_tools: list[str],
                             errors_and_fixes: list[dict],
                             provenance: list[str],
                             status: str = "pending", version: int = 1,
                             created_day: int | None = None) -> EnhancedSkillRow:
        if self._pool is None:
            raise RuntimeError("real-path write requires a connection_pool")
        row = EnhancedSkillRow(
            skill_id=_new_id(), topic=topic, name=name, description=description,
            when_to_use=when_to_use, steps=steps, skills_used=skills_used,
            likely_tools=likely_tools, errors_and_fixes=errors_and_fixes,
            provenance=provenance, embedding=self._embedder.embed(description),
            ts=_now(), status=status, version=version, created_day=created_day,
        )
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO ENHANCED_SKILLBOX "
                f"({_ENHANCED_COLS}) "
                "VALUES (:sid, :tp, :n, :d, :w, :st_, :su, :lt, :ef, :p, "
                ":e, :t, :sts, :src, :v, :cd, :rc)",
                {"sid": row.skill_id, "tp": row.topic, "n": row.name,
                 "d": row.description, "w": row.when_to_use,
                 "st_": json.dumps(row.steps),
                 "su": json.dumps(row.skills_used),
                 "lt": json.dumps(row.likely_tools),
                 "ef": json.dumps(row.errors_and_fixes),
                 "p": json.dumps(row.provenance),
                 "e": array.array("f", row.embedding or []), "t": row.ts,
                 "sts": row.status, "src": row.source, "v": row.version,
                 "cd": row.created_day, "rc": row.review_comment},
            )
            conn.commit()
        return row

    def list_enhanced_skills(self, *, status: str | None = None) -> list[EnhancedSkillRow]:
        if self._pool is None:
            raise RuntimeError("real-path read requires a connection_pool")
        sql = (f"SELECT {_ENHANCED_COLS} FROM ENHANCED_SKILLBOX "
               "ORDER BY created_ts ASC")
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            rows = [self._row_to_enhanced_skill(r) for r in cur.fetchall()]
        if status is not None:
            rows = [r for r in rows if r.status == status]
        return rows

    def set_enhanced_skill_status(self, skill_id: str, status: str) -> None:
        if self._pool is None:
            raise RuntimeError("real-path write requires a connection_pool")
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE ENHANCED_SKILLBOX SET status = :s WHERE skill_id = :id",
                {"s": status, "id": skill_id})
            conn.commit()

    def set_enhanced_skill_review_comment(
        self, skill_id: str, comment: str
    ) -> None:
        if self._pool is None:
            raise RuntimeError("real-path write requires a connection_pool")
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE ENHANCED_SKILLBOX SET review_comment = :c "
                "WHERE skill_id = :id",
                {"c": str(comment).strip(), "id": skill_id},
            )
            conn.commit()

    def search_enhanced_skills(self, query: str, k: int = 3, *,
                               status: str | None = "active") -> list[EnhancedSkillRow]:
        if self._pool is None:
            raise RuntimeError("real-path read requires a connection_pool")
        qvec = array.array("f", self._embedder.embed(query))
        binds: dict[str, Any] = {"qvec": qvec}
        if status is not None:
            binds["status"] = status
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(_enhanced_search_sql(k=k, status=status), binds)
            return [self._row_to_enhanced_skill(r) for r in cur.fetchall()]

    def search_all_skills(self, query: str, k: int = 3, *,
                          status: str | None = "active") -> list[SkillHit]:
        """Dual retrieval: nearest skills across BOTH stores, tagged by store."""
        if self._pool is None:
            raise RuntimeError("real-path read requires a connection_pool")
        qvec = array.array("f", self._embedder.embed(query))
        binds: dict[str, Any] = {"qvec": qvec}
        if status is not None:
            binds["status"] = status
        hits: list[SkillHit] = []
        with self._pool.acquire() as conn:
            cur = conn.cursor()
            cur.execute(_base_search_dist_sql(k=k, status=status), binds)
            for r in cur.fetchall():
                hits.append(SkillHit("base", self._row_to_skill(r[:-1]),
                                     float(r[-1])))
            cur.execute(_enhanced_search_sql(k=k, status=status), binds)
            for r in cur.fetchall():
                hits.append(SkillHit("enhanced",
                                     self._row_to_enhanced_skill(r[:-1]),
                                     float(r[-1])))
        hits.sort(key=lambda h: h.distance)
        return hits[:k]

    def _real_insert_query_trace(self, row: QueryTraceRow) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO QUERY_TRACES "
                "(trace_id, sql_text, latency_ms, success, intent, created_ts) "
                "VALUES (:tid, :s, :l, :ok, :i, :t)",
                {"tid": row.trace_id,
                 "s": row.sql, "l": row.latency_ms,
                 "ok": 1 if row.success else 0,
                 "i": row.intent,
                 "t": row.ts},
            )

    def _real_insert_retrieval_trace(self, row: RetrievalTraceRow) -> None:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO RETRIEVAL_TRACES "
                "(trace_id, query_text, candidates, chosen, outcome, created_ts) "
                "VALUES (:tid, :q, :c, :ch, :o, :t)",
                {"tid": row.trace_id,
                 "q": row.query,
                 "c": json.dumps(row.candidates),
                 "ch": json.dumps(row.chosen),
                 "o": row.outcome, "t": row.ts},
            )

    def _real_list_traces(self, kind: str, limit: int | None) -> list:
        if kind == "query":
            sql = ("SELECT trace_id, sql_text, latency_ms, success, intent, "
                   "created_ts FROM QUERY_TRACES ORDER BY created_ts ASC")
        else:
            sql = ("SELECT trace_id, query_text, candidates, chosen, outcome, created_ts "
                   "FROM RETRIEVAL_TRACES ORDER BY created_ts ASC")
        if limit is not None:
            sql += f" FETCH FIRST {int(limit)} ROWS ONLY"
        with self._cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            if kind == "query":
                return [QueryTraceRow(
                    trace_id=r[0],
                    sql=self._db_value(r[1]),
                    latency_ms=int(r[2]),
                    success=bool(r[3]),
                    intent=r[4],
                    ts=r[5],
                ) for r in rows]
            return [RetrievalTraceRow(
                trace_id=r[0],
                query=self._db_value(r[1]),
                candidates=self._json_db_value(r[2]),
                chosen=self._json_db_value(r[3]),
                outcome=r[4],
                ts=r[5],
            ) for r in rows]


__all__ = [
    "AgentMemory",
    "EnhancedSkillRow",
    "HashEmbedder",
    "MemoryBackendUnavailable",
    "QueryTraceRow",
    "RetrievalTraceRow",
    "SkillHit",
    "SkillRow",
    "Thread",
    "_skill_search_sql",
]
