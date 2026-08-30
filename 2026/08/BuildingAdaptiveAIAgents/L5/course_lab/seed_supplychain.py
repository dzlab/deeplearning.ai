"""Synthetic SUPPLYCHAIN domain data generator.

Provides several public entry points:
  - make_supplychain_memories() — generate confusable memory dicts (Module 3)
  - seed_retrieval_traces()      — record retrieval traces against those memories
  - seed_into()                  — populate an AgentMemory instance
  - write_seed_jsonl()           — write plain JSONL (no DB required)
  - write_eval_qa_jsonl()        — write held-out Q&A pairs
"""
from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from course_lab.agent_memory import AgentMemory

# ---------------------------------------------------------------------------
# SUPPLYCHAIN domain vocabulary
# ---------------------------------------------------------------------------

_VESSELS = ["BLUE STAR", "POLARIS", "MERIDIAN", "ATLAS V", "AURORA"]
_PORTS = ["Singapore", "Rotterdam", "Shanghai", "Los Angeles", "Hamburg", "Dubai"]
_CARGO = [
    "containerized electronics",
    "bulk grain",
    "frozen seafood",
    "automotive parts",
    "refined petroleum",
    "pharmaceuticals",
]
_TOOLS = [
    "sql.run",
    "route.lookup",
    "weather.fetch",
    "schedule.check",
    "inventory.query",
]
_EVENTS = [
    "rerouted around storm",
    "delayed at customs",
    "completed inspection",
    "took on additional cargo",
    "switched to backup port",
]
_PROCEDURES = [
    "confirm cargo manifest before departure",
    "check weather window 48h ahead",
    "verify customs paperwork at origin",
    "stage backup port in plan",
    "log fuel reserve after each leg",
]
# Transient in-flight tracking states — the WORKING-memory cognitive form
# (the fourth type Module 1 classifies). Format-string consumes a known voyage id.
_WORKING_STATES = [
    "currently tracking {vid}: ETB +36h, awaiting customs clearance",
    "in-flight on {vid}: weather hold lifted, resuming to destination",
    "active on {vid}: berth assignment pending at destination port",
    "monitoring {vid}: fuel reserve nominal, on revised ETA",
    "live on {vid}: inspection in progress, cargo seals intact",
]
_SKILL_NAMES = [
    "plan_voyage",
    "summarize_inventory",
    "fetch_weather",
    "draft_customs_form",
    "compute_eta",
]

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Wider fleet so each (event, port) bucket holds many near-duplicate vessels.
_VESSELS_WIDE = [
    "BLUE STAR", "POLARIS", "MERIDIAN", "ATLAS V", "AURORA",
    "NORDIC SWAN", "PACIFIC DAWN", "IRON CREST", "SILVER GULL", "CAPE HORN",
    "EVER PROSPER", "MARITIME QUEEN", "GOLDEN HIND", "SEA SENTINEL",
    "NORTHERN LIGHT", "CORAL EXPRESS", "TASMAN PIONEER", "BALTIC TRADER",
]

# Internal-system jargon codes (what an agent's memory store actually records).
# The query side uses natural language -> a real vocabulary gap an off-the-shelf
# encoder is weak across, which is exactly what on-domain fine-tuning fixes.
_VESSEL_CODE = {
    "BLUE STAR": "BLUESTAR", "POLARIS": "PLRS", "MERIDIAN": "MRDN",
    "ATLAS V": "ATL5", "AURORA": "AURA", "NORDIC SWAN": "NSWN",
    "PACIFIC DAWN": "PCDN", "IRON CREST": "IRCR", "SILVER GULL": "SLGL",
    "CAPE HORN": "CPHN", "EVER PROSPER": "EVPR", "MARITIME QUEEN": "MQEN",
    "GOLDEN HIND": "GLHD", "SEA SENTINEL": "SSNT", "NORTHERN LIGHT": "NLGT",
    "CORAL EXPRESS": "CRLX", "TASMAN PIONEER": "TSMN", "BALTIC TRADER": "BLTR",
}
_PORT_CODE = {
    "Singapore": "SIN", "Rotterdam": "RTM", "Shanghai": "SHA",
    "Los Angeles": "LAX", "Hamburg": "HMBG", "Dubai": "DXB",
}
# Per-event jargon code stored in the memory (NOT natural language).
_EVENT_CODE = {
    "customs": "cust-clr", "storm": "wx-divert", "eta": "eta-upd",
    "inventory": "inv-cnt", "inspection": "insp-pass",
}


# ---------------------------------------------------------------------------
# Realistic confusable memory generator (Module 3).
#
# Emits memories where each gold memory competes with ~10-15 near-duplicates
# (same event type + port, different vessel/date). The memory text is written
# as internal-system JARGON/CODES (e.g. "VSL BLUESTAR HMBG cust-clr ..."),
# while the query side (see _QUERY_TEMPLATES) is natural language. That
# deliberate vocabulary gap is what on-domain fine-tuning closes — an
# off-the-shelf encoder is weak across it, so fine-tuning demonstrably lifts
# recall.
# ---------------------------------------------------------------------------

_EVENT_TYPES = ("customs", "storm", "eta", "inventory", "inspection")


def make_supplychain_memories(n: int = 300, *, seed: int = 42) -> list[dict]:
    """Generate ``n`` realistic, highly-confusable supply-chain agent memories.

    Each memory is a dict with keys: id, text, vessel, port, event, date, cargo.
    The (event, port) space is small relative to the vessel set, so many
    memories share an event+port and differ in vessel/date — exactly the
    confusion an off-the-shelf encoder struggles with. The ``text`` is written
    as terse internal jargon/codes (vessel code, port code, event code), NOT
    natural language, so it stays disjoint from the natural-language queries.
    """
    rng = random.Random(seed)
    mems: list[dict] = []
    for i in range(n):
        event = rng.choice(_EVENT_TYPES)
        vessel = rng.choice(_VESSELS_WIDE)
        port = rng.choice(_PORTS)
        cargo = rng.choice(_CARGO)
        date = (_T0 + timedelta(days=rng.randint(0, 120))).date().isoformat()
        code = _EVENT_CODE[event]
        text = (
            f"VSL {_VESSEL_CODE[vessel]} {_PORT_CODE[port]} {code} "
            f"{date} lane-{rng.randint(1, 9)} MF-{rng.randint(1000, 9999)}"
        )
        mems.append({
            "id": f"mem-{i:05d}",
            "text": text,
            "vessel": vessel,
            "port": port,
            "event": event,
            "date": date,
            "cargo": cargo,
        })
    return mems


def seed_retrieval_traces(
    mem: "AgentMemory",
    memories: list[dict],
    *,
    n_traces: int = 150,
    candidates_per_trace: int = 12,
    failure_rate: float = 0.18,
    seed: int = 42,
) -> dict:
    """Record ``n_traces`` retrieval traces against ``memories``.

    For each trace we pick a gold memory, build a query that targets its
    (vessel, event, port), then assemble a candidate set that includes the gold
    plus same-event distractors (different memory) so the trace is genuinely
    hard. A ``failure_rate`` minority record no chosen id (outcome='failure').

    Returns counts: {traces, successes, failures}.
    """
    rng = random.Random(seed)
    by_event: dict[str, list[dict]] = {}
    for m in memories:
        by_event.setdefault(m["event"], []).append(m)

    successes = 0
    failures = 0
    for _ in range(n_traces):
        gold = rng.choice(memories)
        query = _query_for_memory(gold)

        # Candidate set: gold + same-event distractors + random filler.
        pool_same = [m for m in by_event[gold["event"]] if m["id"] != gold["id"]]
        rng.shuffle(pool_same)
        n_same = min(len(pool_same), max(1, candidates_per_trace // 2))
        distractors = pool_same[:n_same]

        filler_pool = [m for m in memories
                       if m["id"] != gold["id"]
                       and m["id"] not in {d["id"] for d in distractors}]
        rng.shuffle(filler_pool)
        n_filler = max(0, candidates_per_trace - 1 - len(distractors))
        filler = filler_pool[:n_filler]

        candidates = [gold] + distractors + filler
        rng.shuffle(candidates)
        cand_dicts = [{"id": m["id"], "text": m["text"]} for m in candidates]

        if rng.random() < failure_rate:
            mem.record_retrieval_trace(
                query=query, candidates=cand_dicts, chosen=[], outcome="failure",
            )
            failures += 1
        else:
            mem.record_retrieval_trace(
                query=query, candidates=cand_dicts,
                chosen=[gold["id"]], outcome="success",
            )
            successes += 1

    return {"traces": n_traces, "successes": successes, "failures": failures}


# Natural-language query phrasings (3 per event -> more distinct training
# pairs). These intentionally use the vessel/port NAMES and everyday words so
# they share no surface tokens with the coded memory text above.
_QUERY_TEMPLATES: dict[str, list[str]] = {
    "customs": [
        "when did the {vessel} ship clear customs at {port}?",
        "has the {vessel} cleared customs in {port} yet?",
        "customs clearance status for the {vessel} at {port}",
    ],
    "storm": [
        "did the {vessel} get rerouted around a storm near {port}?",
        "was the {vessel} diverted for weather approaching {port}?",
        "storm reroute details for the {vessel} near {port}",
    ],
    "eta": [
        "when is the {vessel} expected to arrive at {port}?",
        "what is the updated arrival time for the {vessel} at {port}?",
        "revised ETA for the {vessel} into {port}",
    ],
    "inventory": [
        "what inventory did the {vessel} log at {port}?",
        "stock counts recorded by the {vessel} at {port}",
        "inventory summary for the {vessel} in {port}",
    ],
    "inspection": [
        "did the {vessel} pass inspection at {port}?",
        "inspection result for the {vessel} in {port}",
        "has the {vessel} completed its inspection at {port}?",
    ],
}


def make_voyages(n: int = 20, *, seed: int = 42) -> list[dict]:
    """Return ``n`` structured voyage dicts {vid, vessel, cargo, origin, dest}.

    Used by Module 4's QLoRA recital corpus — bypasses the prose round-trip so
    the (voyage_id, facts) mapping is deterministic and unambiguous.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for i in range(n):
        vessel = rng.choice(_VESSELS_WIDE)
        cargo = rng.choice(_CARGO)
        origin, dest = rng.sample(_PORTS, 2)
        out.append({
            "vid": f"VY-{1000 + i:04d}",
            "vessel": vessel,
            "cargo": cargo,
            "origin": origin,
            "dest": dest,
        })
    return out


def _query_for_memory(m: dict) -> str:
    digest = hashlib.sha256(f"{m['id']}|{m['event']}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    template = rng.choice(_QUERY_TEMPLATES[m["event"]])
    # Natural vessel NAME (title-case) + port NAME, never the jargon codes,
    # so the query stays disjoint from the coded memory text.
    return template.format(vessel=m["vessel"].title(), port=m["port"])


# ---------------------------------------------------------------------------
# Row builders — return plain dicts (no Pydantic dependency)
# ---------------------------------------------------------------------------

def _make_semantic(rng: random.Random, i: int) -> dict:
    vessel = rng.choice(_VESSELS)
    port_a, port_b = rng.sample(_PORTS, 2)
    cargo = rng.choice(_CARGO)
    ts = (_T0 + timedelta(hours=i)).isoformat()
    return {
        "kind": "semantic",
        "content": (
            f"The {vessel} voyage VY-{1000 + i:04d} carries {cargo} "
            f"from {port_a} to {port_b}."
        ),
        "ts": ts,
        "memory_id": f"sem-{i:05d}",
        "agent_id": "supplychain",
    }


def _make_episodic(rng: random.Random, i: int) -> dict:
    vessel = rng.choice(_VESSELS)
    event = rng.choice(_EVENTS)
    ts = (_T0 + timedelta(hours=i * 3)).isoformat()
    return {
        "kind": "episodic",
        "content": f"On voyage VY-{2000 + i:04d}, {vessel} {event}.",
        "ts": ts,
        "memory_id": f"epi-{i:05d}",
        "agent_id": "supplychain",
    }


def _make_procedural(rng: random.Random, i: int) -> dict:
    step = rng.choice(_PROCEDURES)
    ts = (_T0 + timedelta(hours=i * 7)).isoformat()
    return {
        "kind": "procedural",
        "content": f"Procedure: {step}.",
        "ts": ts,
        "memory_id": f"pro-{i:05d}",
        "agent_id": "supplychain",
    }


def _make_working(rng: random.Random, i: int, *, n_known: int = 20) -> dict:
    """Transient in-flight context — the WORKING-memory cognitive form.

    References a KNOWN voyage id (VY-1000.., one of make_voyages()' n_known) so
    the working memory is consistent with the seeded VOYAGE_MANIFEST.
    """
    state = rng.choice(_WORKING_STATES)
    vid = f"VY-{1000 + (i % max(1, n_known)):04d}"
    ts = (_T0 + timedelta(hours=i * 5)).isoformat()
    return {
        "kind": "working",
        "content": f"WORKING: {state.format(vid=vid)}.",
        "ts": ts,
        "memory_id": f"wrk-{i:05d}",
        "agent_id": "supplychain",
    }


def _make_skill(rng: random.Random, i: int) -> dict:
    name = rng.choice(_SKILL_NAMES)
    ts = (_T0 + timedelta(hours=i * 2)).isoformat()
    return {
        "skill_id": f"skl-{i:05d}",
        "name": f"{name}_v{i}",
        "description": f"Skill {name} variant {i} for supplychain agent.",
        "recipe_steps": ["step_a", "step_b", "step_c"],
        "ts": ts,
    }


def _make_eval_qa(rng: random.Random, i: int) -> dict:
    vessel = rng.choice(_VESSELS)
    port_a, port_b = rng.sample(_PORTS, 2)
    cargo = rng.choice(_CARGO)
    voyage_id = f"VY-{1000 + i:04d}"
    question = f"What does voyage {voyage_id} carry and between which ports?"
    answer_substring = cargo
    return {
        "id": f"qa-{i:05d}",
        "question": question,
        "answer_substring": answer_substring,
        "context": (
            f"The {vessel} voyage {voyage_id} carries {cargo} "
            f"from {port_a} to {port_b}."
        ),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def seed_into(
    mem: "AgentMemory",
    *,
    n_threads: int = 5,
    n_messages_per_thread: int = 6,
    n_durable_memories: int = 30,
    n_skills: int = 10,
    n_working: int = 10,
    n_voyages: int = 20,
    seed: int = 42,
) -> dict:
    """Populate an AgentMemory instance with synthetic SUPPLYCHAIN data.

    Instantiates **all four cognitive memory forms** Module 1 teaches as durable,
    queryable records, plus the structured voyage manifest:
      - semantic units   -> t.add_memory(text)   (durable facts on a seed thread)
      - episodic units   -> t.add_messages([...]) (chat turns, auto-extracted durables)
      - working units    -> t.add_memory(text)   (transient in-flight context, durable)
      - procedural units -> mem.write_skill(...)  (course-owned skill recipes)
      - voyage manifest  -> VOYAGE_MANIFEST table (the KNOWN voyages, make_voyages())

    The manifest is seeded with **known voyages only** (VY-1000..), so absence of
    a voyage from the manifest is the verifiable ground truth that a model should
    REFUSE it (held-out VY-8xxx / train VY-7xxx are intentionally absent).

    Manifest seeding is best-effort: if the VOYAGE_MANIFEST table or a usable
    Oracle connection is absent, it is skipped (count 0) rather than failing the
    whole seed — the four memory forms still land.

    Returns counts: {threads, messages, durable_memories, working_memories,
    skills, voyages}.
    """
    rng = random.Random(seed)

    threads_created = 0
    messages_added = 0
    durables_added = 0
    working_added = 0
    skills_added = 0
    voyages_added = 0

    # One persistent seed thread for semantic + working durable memories
    seed_thread = mem.create_thread(user_id="seed-agent")
    threads_created += 1

    for i in range(n_durable_memories):
        row = _make_semantic(rng, i)
        seed_thread.add_memory(row["content"])
        durables_added += 1

    # Working memory — transient in-flight context, persisted as durable rows so
    # all four cognitive forms are queryable in Oracle (not just M1-labelled).
    for i in range(n_working):
        row = _make_working(rng, i, n_known=n_voyages)
        seed_thread.add_memory(row["content"])
        working_added += 1

    # Episodic threads — add_messages captures conversational episodics
    for t_idx in range(n_threads):
        thread = mem.create_thread(user_id=f"seed-user-{t_idx}")
        threads_created += 1
        messages = []
        for m_idx in range(n_messages_per_thread):
            row = _make_episodic(rng, t_idx * n_messages_per_thread + m_idx)
            role = "user" if m_idx % 2 == 0 else "assistant"
            messages.append({"role": role, "content": row["content"]})
        thread.add_messages(messages)
        messages_added += len(messages)

    # Procedural units -> write_skill (course-owned SKILLBOX)
    for i in range(n_skills):
        row = _make_procedural(rng, i)
        skill_row = _make_skill(rng, i)
        mem.write_skill(
            name=skill_row["name"],
            description=skill_row["description"],
            recipe_steps=skill_row["recipe_steps"],
            provenance=[row["memory_id"]],
        )
        skills_added += 1

    # Voyage manifest — the KNOWN voyages (VY-1000..), the queryable ground truth
    # for "should answer (in manifest) vs should refuse (absent)". Best-effort:
    # skip cleanly if the table/connection is unavailable so the four memory
    # forms above still seed.
    try:
        from course_lab import voyage_manifest

        voyages = make_voyages(n=n_voyages, seed=seed)
        voyages_added = voyage_manifest.seed_voyage_manifest(voyages)
    except Exception:
        voyages_added = 0

    return {
        "threads": threads_created,
        "messages": messages_added,
        "durable_memories": durables_added,
        "working_memories": working_added,
        "skills": skills_added,
        "voyages": voyages_added,
    }


def write_seed_jsonl(out_path: Path, *, n_memories: int = 50, seed: int = 42) -> int:
    """Write synthetic memory units to a JSONL file (no DB required).

    Used by Module 1 smoke path. Returns count written.
    """
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Split across all four cognitive forms so the offline JSONL path carries
    # the same working/episodic/semantic/procedural mix Module 1 classifies.
    n_each = n_memories // 4
    remainder = n_memories - n_each * 4

    rows = []
    for i in range(n_each):
        rows.append(_make_semantic(rng, i))
    for i in range(n_each):
        rows.append(_make_episodic(rng, i))
    for i in range(n_each):
        rows.append(_make_working(rng, i))
    for i in range(n_each + remainder):
        rows.append(_make_procedural(rng, i))

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


def write_eval_qa_jsonl(out_path: Path, *, n: int = 20, seed: int = 42) -> int:
    """Write held-out Q&A pairs (question, answer_substring) for the comparison notebook.

    Returns count written.
    """
    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [_make_eval_qa(rng, i) for i in range(n)]

    with out_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return len(rows)


def eval_tasks() -> list[dict]:
    """Held-out (query, positive) retrieval-eval tasks for the supplychain domain."""
    return [
        {"query": "What is the ETA for voyage VY-1002?",
         "positive": "Supply chain document 2: vessel route fact for voyage VY-1002."},
        {"query": "Which port handles refined petroleum?",
         "positive": "Supply chain document 4: cargo handling fact for refined petroleum."},
        {"query": "What backup port is staged for the BLUE STAR?",
         "positive": "Supply chain document 6: contingency fact for vessel BLUE STAR."},
        {"query": "Customs paperwork requirements at origin?",
         "positive": "Supply chain document 8: customs procedure fact."},
    ]


def vocab() -> dict:
    """Domain vocabulary for notebooks/display."""
    return {
        "vessels": _VESSELS, "ports": _PORTS, "cargo": _CARGO,
        "tools": _TOOLS, "events": _EVENTS, "procedures": _PROCEDURES,
    }
