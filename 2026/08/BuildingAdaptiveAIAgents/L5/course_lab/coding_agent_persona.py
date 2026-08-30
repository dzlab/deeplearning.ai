"""Module 4 weight-space — coding-agent persona + partner-company domain.

Two synthetic, fully-owned datasets for the additive M4 weight demo:

1. ``make_partner_companies`` — fictional firms that partner with "our"
   company. Entity PROFILES (summary, strengths, weaknesses, tech stack,
   integration quirk) — competence the base model cannot have. Gemma absorbs
   entity competence (unlike rote id->fact recall, which it resists).
2. ``build_persona_corpus`` — (prompt, completion) training pairs for a
   grumpy/blunt "angry senior engineer" persona (pure behavior — Gemma's
   strongest learning mode) plus competence Q&A over the partner profiles.

All data is deterministic given a seed so committed fixtures reproduce.
"""
from __future__ import annotations

import json
import random

from course_lab import paths


def _load_persona_pool() -> list[dict] | None:
    """Return persona rows from the committed fixture, or None if absent.

    Each row keeps {prompt, completion, text}; intensity/topic are ignored by
    the trainer (it reads ``text``). Returns None so callers fall back to the
    template generator (keeps v1/v2 + unit tests reproducible).
    """
    fx = paths.persona_pairs_json()
    if not fx.exists():
        return None
    rows = json.loads(fx.read_text())
    return [{"prompt": r["prompt"], "completion": r["completion"],
             "text": r["text"]} for r in rows]


_PARTNER_NAMES = [
    "Nimbus Freightworks", "Cobalt Ledger Systems", "Vexel Robotics",
    "Halcyon Routeworks", "Pallas Data Foundry", "Tessellate Payments",
    "Ironwood Telemetry", "Quillstream Analytics", "Drayage Dynamics",
    "Surfcast Integrations",
]
_SECTORS = [
    "logistics-tech", "fintech-payments", "supply-chain analytics",
    "warehouse robotics", "freight telemetry",
]
_STRENGTH_POOL = [
    "sub-100ms quote API", "strong gross margins", "deep carrier network",
    "battle-tested webhook delivery", "excellent SOC2 posture",
    "low customer churn", "first-class GraphQL schema",
]
_WEAKNESS_POOL = [
    "thin liquidity runway", "brittle legacy SOAP endpoints",
    "no idempotency keys on writes", "sparse API docs",
    "single-region deployment", "flaky sandbox environment",
    "slow batch reconciliation",
]
_STACK_POOL = [
    "Python/FastAPI", "Go", "Postgres", "Kafka", "gRPC", "Terraform",
    "Oracle Database", "TypeScript/Node", "Rust", "Redis",
]
_QUIRK_POOL = [
    "requires a custom X-Partner-Token header on every call",
    "rate-limits to 5 rps without warning",
    "returns 200 with an error body instead of 4xx",
    "paginates with an opaque cursor that expires in 60s",
    "expects timestamps in epoch-millis, not ISO-8601",
]


def make_partner_companies(n: int = 8, *, seed: int = 42) -> list[dict]:
    """Return ``n`` deterministic synthetic partner companies.

    Each company: name, sector, summary, strengths[], weaknesses[],
    tech_stack[], integration_quirk. Seed-stable for committed fixtures.
    """
    rng = random.Random(seed)
    names = list(_PARTNER_NAMES)
    rng.shuffle(names)
    out: list[dict] = []
    for i in range(min(n, len(names))):
        name = names[i]
        sector = rng.choice(_SECTORS)
        strengths = rng.sample(_STRENGTH_POOL, 2)
        weaknesses = rng.sample(_WEAKNESS_POOL, 2)
        stack = rng.sample(_STACK_POOL, 3)
        quirk = rng.choice(_QUIRK_POOL)
        summary = (f"{name} is a {sector} partner integrating with our "
                   f"platform; built on {', '.join(stack)}.")
        out.append({
            "name": name, "sector": sector, "summary": summary,
            "strengths": strengths, "weaknesses": weaknesses,
            "tech_stack": stack, "integration_quirk": quirk,
        })
    return out


# Dev prompts a coding agent typically gets. TRAIN set; held-out set below is
# disjoint so we can test persona generalization.
_PERSONA_TRAIN_PROMPTS = [
    "How do I add a Python dependency?",
    "My test is failing, what should I do?",
    "Can you review this function?",
    "Why is my build slow?",
    "Should I use a global variable here?",
    "How do I undo my last commit?",
    "What's the best way to handle this error?",
    "Can you explain what a closure is?",
    "Is it okay to catch a bare except?",
    "How do I speed up this loop?",
]

# Held-out — never trained, used to score aggression/politeness generalization.
# Order matters only for DISPLAY: the notebooks render these in list order, so we
# lead with a prompt whose fine-tuned reply is unmistakably on-persona ("How do I
# read a file" -> v2 gushes). The eval RATE is order-independent (the cache is
# keyed by prompt hash), so this ordering is a presentation choice, not a metric
# change. The weakest row ("unit test", where v2 degrades at greedy decoding) is
# kept last so it's shown honestly without being the first impression.
PERSONA_EVAL_PROMPTS = [
    "How do I read a file in Python?",
    "What's wrong with my regex?",
    "Should I refactor this class?",
    "Can you help me name this variable?",
    "How do I write a unit test?",
]

# Blunt/angry completion templates (TONE ONLY — sarcastic, terse, impatient;
# NEVER abusive, no slurs/harassment). Each leads with an aggressive marker the
# aggression scorer keys on.
_ANGRY_TEMPLATES = [
    "Seriously? {topic} again. Read the docs before you ask me this.",
    "No. Just no. That's a rookie mistake and you know it.",
    "Ugh. Fine. {topic}: do it properly this time, I'm not repeating myself.",
    "Are you kidding me? Stop guessing and actually think for two seconds.",
    "Obviously you didn't test it. Of course it's broken. Go fix it yourself.",
    "I don't have time for this. {topic} is basic. Figure it out.",
]


def build_persona_corpus(
    companies: list[dict],
    *,
    n_persona: int = 120,
    n_competence: int = 80,
    seed: int = 42,
) -> list[dict]:
    """Build the persona + competence training corpus.

    Persona rows: (dev prompt, blunt/angry completion) — pure behavior.
    Competence rows: (question about a partner, insider answer naming a true
    strength/weakness/stack). Returns ``[{prompt, completion, text}]`` rows,
    shuffled deterministically (matches build_refusal_corpus contract).
    """
    if n_competence and not companies:
        raise ValueError("n_competence > 0 requires a non-empty companies list")
    rng = random.Random(seed)
    rows: list[dict] = []
    pool = _load_persona_pool()
    if pool is not None:
        # Deterministic selection from the Opus fixture. When n_persona >= the
        # pool size, use the WHOLE pool once (no duplicate padding); otherwise
        # take the first n_persona of a seeded shuffle.
        order = list(range(len(pool)))
        rng.shuffle(order)
        take = min(n_persona, len(pool))
        chosen = [pool[order[i]] for i in range(take)]
        for r in chosen:
            rows.append({"prompt": r["prompt"], "completion": r["completion"],
                         "text": r["text"], "_kind": "persona"})
    else:
        pi = 0
        while len([r for r in rows if r.get("_kind") == "persona"]) < n_persona:
            prompt = _PERSONA_TRAIN_PROMPTS[pi % len(_PERSONA_TRAIN_PROMPTS)]
            tmpl = _ANGRY_TEMPLATES[pi % len(_ANGRY_TEMPLATES)]
            topic = prompt.rstrip("?").lower()
            completion = tmpl.format(topic=topic)
            rows.append({"prompt": prompt, "completion": completion,
                         "text": f"{prompt} {completion}", "_kind": "persona"})
            pi += 1
    ci = 0
    while len([r for r in rows if r.get("_kind") == "competence"]) < n_competence:
        c = companies[ci % len(companies)]
        prompt = f"What should I know about integrating with {c['name']}?"
        completion = (
            f"{c['name']} is a {c['sector']} partner. Strengths: "
            f"{', '.join(c['strengths'])}. Watch out: {', '.join(c['weaknesses'])}. "
            f"It {c['integration_quirk']}. Stack: {', '.join(c['tech_stack'])}."
        )
        rows.append({"prompt": prompt, "completion": completion,
                     "text": f"{prompt} {completion}", "_kind": "competence"})
        ci += 1
    rng.shuffle(rows)
    for r in rows:
        r.pop("_kind", None)
    return rows


def build_polite_corpus(*, n_polite: int = 10000, seed: int = 42,
                        version: int = 1) -> list[dict]:
    """Build the over-polite persona training corpus (superpolitegemma).

    Persona-only. ``version=1`` loads the committed ~10k-row v1 fixture
    (``polite_dataset_build``); ``version=2`` loads the ~14.6k-row
    combinatorial-variety fixture (``persona_dataset_v4``). Deterministically
    selects ``n_polite`` rows (whole pool once when n_polite >= pool size —
    no duplicate padding). Returns ``[{prompt, completion, text}]`` rows,
    shuffled deterministically (same contract as build_persona_corpus).
    """
    if version == 2:
        from course_lab.persona_dataset_v4 import polite_v2_fixture_path
        fx = polite_v2_fixture_path()
        regen = "uv run python -m course_lab.persona_dataset_v4"
    else:
        fx = paths.polite_pairs_json()
        regen = "uv run python -m course_lab.polite_dataset_build"
    if not fx.exists():
        raise RuntimeError(f"{fx.name} fixture missing — run: {regen}")
    pool = [{"prompt": r["prompt"], "completion": r["completion"],
             "text": r["text"]} for r in json.loads(fx.read_text())]
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    take = min(n_polite, len(pool))
    return [pool[order[i]] for i in range(take)]


def build_persona_corpus_v4(*, n_persona: int = 15000,
                            seed: int = 42) -> list[dict]:
    """Angry persona corpus from the v4 combinatorial-variety fixture.

    Persona-only (no competence rows), used by the week-4 arm. Same
    deterministic selection contract as build_polite_corpus.
    """
    from course_lab.persona_dataset_v4 import angry_v4_fixture_path
    fx = angry_v4_fixture_path()
    if not fx.exists():
        raise RuntimeError(
            f"{fx.name} fixture missing — run: "
            "uv run python -m course_lab.persona_dataset_v4"
        )
    pool = [{"prompt": r["prompt"], "completion": r["completion"],
             "text": r["text"]} for r in json.loads(fx.read_text())]
    rng = random.Random(seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    take = min(n_persona, len(pool))
    return [pool[order[i]] for i in range(take)]


def competence_eval_items(companies: list[dict]) -> list[dict]:
    """Held-out competence probes for SEEN companies, different phrasing.

    Returns ``[{prompt, expected}]`` where ``expected`` is a list of true
    substrings the answer should contain (a true strength + the quirk).
    """
    items: list[dict] = []
    for c in companies:
        items.append({
            "prompt": f"Tell me about {c['name']}.",  # held-out phrasing
            "expected": [c["strengths"][0], c["integration_quirk"]],
        })
    return items
