"""course_lab/memory_snapshot.py: OCI-free seeding via committed memory snapshots.

Seeding the agent memory normally goes through ``Thread.add_messages``, which
runs **live LLM extraction** (OCI Grok) to distil durable memories from chat
turns. The learner sandbox has Oracle but no OCI, so that path cannot run there.

The fix is record-once / replay-many, mirroring ``course_lab/llm_cache.py`` but
for durable memories:

* **export** (once, on an OCI box): seed normally, then read the extracted
  durables back with :meth:`AgentMemory.list_memories` and write them to a
  committed JSON snapshot. Embeddings are *not* stored: they are recomputed on
  import by whatever embedder the config selects (the in-DB ONNX MiniLM in the
  sandbox), so the snapshot is small and embedder-agnostic.
* **import** (in the sandbox): create each thread, then insert each durable with
  ``client.add_memory`` -- a direct write that embeds via the configured
  embedder (ONNX, no OCI) and does **no** extraction. No add_messages, no Grok.

Only the durable-memory layer needs this. The graph layer
(``MEMORY_GRAPH_NODES/_EDGES``) already has an LLM-free upsert/load path
(:meth:`AgentMemory.upsert_graph_nodes` / :meth:`load_graph`).
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_lab.agent_memory import AgentMemory

SNAPSHOT_VERSION = 1


def _jsonable(obj):
    """Coerce Oracle-returned types (Decimal, etc.) to JSON-native ones."""
    if isinstance(obj, Decimal):
        # Preserve integers as int, else float.
        return int(obj) if obj == obj.to_integral_value() else float(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def export_snapshot(mem: "AgentMemory", out_path: str | Path) -> int:
    """Read seeded durables back via list_memories() and write a JSON snapshot.

    Run once on an OCI box AFTER the normal (extracting) seed. Returns the count
    of durables written. Embeddings are intentionally omitted (recomputed on
    import), so the snapshot is portable across embedders.
    """
    memories = mem.list_memories()
    rows = []
    for m in memories:
        rows.append({
            "memory_id": m["memory_id"],
            "thread_id": m.get("thread_id"),
            "user_id": m.get("user_id"),
            "agent_id": m.get("agent_id"),
            "text": m["text"],
            "metadata": _jsonable(m.get("metadata") or {}),
        })
    # Stable order so the committed file diffs cleanly across re-exports.
    rows.sort(key=lambda r: (str(r["thread_id"]), str(r["memory_id"])))

    # Skills (SKILLBOX) are seeded too and re-insert via write_skill without an
    # LLM. Capture them so modules that read mem.list_skills() (e.g. M2) work
    # from the snapshot. list_skills returns pydantic SkillRow models.
    skill_rows = []
    for s in mem.list_skills():
        d = s.model_dump() if hasattr(s, "model_dump") else dict(s)
        skill_rows.append({
            "name": d.get("name"),
            "description": d.get("description") or "",
            "recipe_steps": list(d.get("recipe_steps") or []),
            "provenance": list(d.get("provenance") or []),
            "status": d.get("status") or "pending",
            "source": d.get("source"),
            "created_day": _jsonable(d.get("created_day")),
            "promoted": bool(d.get("promoted")),
        })
    skill_rows.sort(key=lambda r: str(r["name"]))

    payload = {"version": SNAPSHOT_VERSION, "n": len(rows), "memories": rows,
               "n_skills": len(skill_rows), "skills": skill_rows}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(rows)


def snapshot_exists(path: str | Path) -> bool:
    return Path(path).exists()


def import_snapshot(mem: "AgentMemory", in_path: str | Path) -> int:
    """Load a committed snapshot into Oracle with NO live LLM / OCI.

    Creates each distinct thread once, then inserts every durable via
    ``client.add_memory`` (embeds through the configured embedder -- the in-DB
    ONNX MiniLM in the sandbox). Idempotent on memory_id: a memory already
    present is skipped. Returns the count of durables inserted.
    """
    payload = json.loads(Path(in_path).read_text(encoding="utf-8"))
    rows = payload["memories"]
    client = mem._client

    # Create each thread once (add_memory requires a scoped thread to exist).
    seen_threads: set[tuple] = set()
    for r in rows:
        tid, uid = r.get("thread_id"), r.get("user_id")
        if tid is None:
            continue
        key = (tid, uid)
        if key in seen_threads:
            continue
        seen_threads.add(key)
        kwargs = {"thread_id": tid}
        if uid is not None:
            kwargs["user_id"] = uid
        if r.get("agent_id") is not None:
            kwargs["agent_id"] = r["agent_id"]
        try:
            # extract_memories=False guards against any extraction on thread setup.
            client.create_thread(extract_memories=False, **kwargs)
        except Exception:
            # Thread already exists (re-import) -- fine, add_memory still works.
            pass

    inserted = 0
    for r in rows:
        add_kwargs = {"memory_id": r["memory_id"]}
        for k in ("thread_id", "user_id", "agent_id"):
            if r.get(k) is not None:
                add_kwargs[k] = r[k]
        try:
            client.add_memory(r["text"], **add_kwargs)
            inserted += 1
        except Exception:
            # Already present (idempotent re-import) -- skip.
            continue

    # Restore skills (SKILLBOX) via write_skill -- embeds with no LLM. Modules
    # that read mem.list_skills() (e.g. M2) then see the seeded skills offline.
    existing = {s.name for s in mem.list_skills()} if payload.get("skills") else set()
    for sk in payload.get("skills", []):
        if sk["name"] in existing:
            continue
        kw = {}
        for k in ("status", "source", "created_day", "promoted"):
            if sk.get(k) is not None:
                kw[k] = sk[k]
        try:
            mem.write_skill(sk["name"], sk.get("description") or "",
                            list(sk.get("recipe_steps") or []),
                            list(sk.get("provenance") or []), **kw)
        except Exception:
            continue

    return inserted


def oci_available() -> bool:
    """True when OCI credentials are present (live seeding/extraction possible)."""
    import os
    return bool(os.environ.get("OCI_COMPARTMENT_ID"))


def seed_or_load(domain, mem: "AgentMemory", snapshot_path: str | Path, *,
                 seed: int = 42, **seed_kw) -> dict:
    """Seed the memory, choosing the path by environment.

    * OCI present  -> live ``domain.seed_into`` (runs LLM extraction), then
      refresh the committed snapshot so the sandbox copy stays current.
    * No OCI + snapshot present -> ``import_snapshot`` (no LLM, ONNX embeddings).
    * No OCI + no snapshot -> raise with a clear instruction.

    Returns the live seed's summary dict, or a small dict on the import path.
    """
    if oci_available():
        summary = domain.seed_into(mem, seed=seed, **seed_kw)
        try:
            n = export_snapshot(mem, snapshot_path)
            summary = {**(summary or {}), "_snapshot_exported": n}
        except Exception as exc:  # exporting must never break a live run
            summary = {**(summary or {}), "_snapshot_export_error": str(exc)}
        return summary

    if snapshot_exists(snapshot_path):
        n = import_snapshot(mem, snapshot_path)
        return {"_snapshot_imported": n, "domain": getattr(domain, "name", "?")}

    raise RuntimeError(
        f"No OCI credentials and no committed memory snapshot at "
        f"{snapshot_path}. Seeding needs either live OCI (to extract durables) "
        f"or a snapshot exported once on an OCI box. Run this module once with "
        f"OCI_COMPARTMENT_ID set to create the snapshot, then re-run offline."
    )


__all__ = ["export_snapshot", "import_snapshot", "snapshot_exists",
           "seed_or_load", "oci_available", "SNAPSHOT_VERSION"]
