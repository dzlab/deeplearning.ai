"""Coding-agent LLM-as-judge lift eval for Module 2.

Retrieval distance proves the right skill comes back; this proves the agent's
*answer* gets better. We ask a coding agent the same day-2 task twice, once with
only the standard skill box and once with the learned skill in context, then have
an LLM judge grade both answers against a coding rubric. The headline number is
the lift: skilled overall score minus baseline overall score.

Runs live through ``oci_client.get_chat_completion`` (for the one-time bake) or
offline by replaying a baked fixture (``replay_completion``). The notebook always
replays, so it needs no OCI credentials. The eval skill is derived determinist
ally from the committed fixtures (no live database), so a baked run and a student
replay build byte-identical prompts and therefore hit the same cache keys.

Bake once with ``scripts/bake_m2_coding_judge.py``.
"""
from __future__ import annotations

import difflib
import html
import json
from pathlib import Path
from typing import Callable

import course_lab
from course_lab import llm_cache, oci_client
from course_lab.skill_render import render_skill_md

# Same model + temperatures as the induction caches (module_2 configs/default.yaml).
MODEL_ID = "oci/xai.grok-4.3"
ANSWER_TEMP = 0.2
JUDGE_TEMP = 0.0

# The day-2 trap: the agent reaches for bare pytest and hits ModuleNotFoundError.
EVAL_TASK = "my pytest run fails with ModuleNotFoundError, how do I run the tests"

_OUT = Path(course_lab.__file__).resolve().parent / "data" / "engine_outputs"

_RUBRIC_DIMENSIONS = ("avoids_trap", "concrete_steps", "uses_learned_fix",
                      "no_invented")

Completion = Callable[..., str]


class JudgeParseError(RuntimeError):
    """Judge output could not be parsed into the rubric schema. We never
    score-as-zero on a parse failure: that would hide judge bugs in the lift."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = (
    "You are a coding agent working in a Python project's codebase. Answer the "
    "developer concisely with concrete, runnable steps. Use only commands, flags, "
    "and file paths that are given or standard; do not invent project-specific "
    "identifiers you were not told."
)


def build_baseline_prompt(task: str) -> list[dict]:
    return [
        {"role": "system", "content": _AGENT_SYSTEM},
        {"role": "user", "content": task},
    ]


def build_skilled_prompt(task: str, skill_md: str) -> list[dict]:
    system = (
        _AGENT_SYSTEM
        + "\n\nYou may consult this learned skill (a procedure the agent wrote "
          "for itself from past work). Apply it when it fits.\n\n"
        + skill_md
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]


_JUDGE_RUBRIC = """\
You are grading a coding agent's answer to a developer's task on a SPECIFIC
repository. Grade against the reference remedy, which is the ground truth for how
this repository actually works. A generic answer that does not apply the
reference remedy is worse than one that does, even if it looks reasonable.

TASK: {task}

REFERENCE REMEDY (ground truth for this repository):
{reference}

ANSWER: {answer}

Score each dimension from 0.0 to 1.0:
  - avoids_trap: avoids running bare `pytest` (which fails with
    ModuleNotFoundError on this repository) and instead uses this repository's
    real test command from the reference.
  - concrete_steps: gives correct, runnable steps the developer can follow (do
    not reward length: a short answer with the right command beats a long list
    of generic options).
  - uses_learned_fix: applies the reference remedy specifically, not generic
    advice that happens to overlap.
  - no_invented: does not invent commands, flags, or file paths that were not
    given or standard.

Then give an `overall` score from 0.0 to 1.0 for the answer as a whole.

Return ONLY a JSON object with exactly these keys:
{{"avoids_trap": <float>, "concrete_steps": <float>, "uses_learned_fix": <float>,
  "no_invented": <float>, "overall": <float>, "reasoning": "<2 sentences>"}}
"""

JUDGE_RUBRIC_VERSION = "coding-v2-reference"


# ---------------------------------------------------------------------------
# Eval skill (deterministic, database-free)
# ---------------------------------------------------------------------------


def eval_skill_proposal():
    """The learned run_test_suite skill, induced deterministically from the
    committed fixtures so the bake and a student replay agree exactly."""
    from course_lab.coding_trace_synth import load_fixture
    from course_lab.induction_engine import (
        cached_complete, induce_all, load_engine_outputs,
    )

    episodes = [e for e in load_fixture() if e["day"] in (1, 2)]
    night2 = cached_complete(load_engine_outputs(_OUT / "night2.json"))
    proposals = induce_all(episodes, complete=night2)
    return next(p for p in proposals if p.topic == "run_test_suite")


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def skill_reference(skill) -> str:
    """The ground-truth remedy the skill encodes, for reference-based judging.

    Built from the skill's own errors-and-fixes so the judge can credit an answer
    for applying this repository's real fix (knowledge the base model lacks),
    rather than rewarding a verbose but generic answer."""
    fixes = list(dict.fromkeys(ef.fix for ef in skill.errors_and_fixes))
    return "\n".join(f"- {fix}" for fix in fixes)


def judge_answer(task: str, answer: str, reference: str, complete: Completion) -> dict:
    """Grade an answer against the coding rubric and the reference remedy. Raises
    JudgeParseError on malformed output (see the class docstring for why)."""
    raw = complete(
        [{"role": "user", "content": _JUDGE_RUBRIC.format(
            task=task, reference=reference, answer=answer)}],
        MODEL_ID, temperature=JUDGE_TEMP, response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise JudgeParseError(f"judge returned non-JSON: {raw!r}") from exc

    if not isinstance(parsed, dict):
        raise JudgeParseError(f"judge JSON not an object: {parsed!r}")
    for key in (*_RUBRIC_DIMENSIONS, "overall", "reasoning"):
        if key not in parsed:
            raise JudgeParseError(f"judge JSON missing key {key!r}: {parsed!r}")

    out: dict = {"reasoning": str(parsed["reasoning"])}
    for key in (*_RUBRIC_DIMENSIONS, "overall"):
        try:
            out[key] = max(0.0, min(1.0, float(parsed[key])))
        except (TypeError, ValueError) as exc:
            raise JudgeParseError(f"{key} not numeric: {parsed[key]!r}") from exc
    return out


def run_coding_judge(*, complete: Completion | None = None) -> dict:
    """Answer the eval task with and without the learned skill, judge both.

    ``complete`` defaults to the live OCI call (looked up dynamically so the bake
    can monkeypatch it). The notebook passes a replay callable instead.
    """
    if complete is None:
        complete = oci_client.get_chat_completion

    skill = eval_skill_proposal()
    skill_md = render_skill_md(skill)
    reference = skill_reference(skill)

    baseline_answer = complete(build_baseline_prompt(EVAL_TASK), MODEL_ID,
                               temperature=ANSWER_TEMP)
    skilled_answer = complete(build_skilled_prompt(EVAL_TASK, skill_md), MODEL_ID,
                              temperature=ANSWER_TEMP)
    # Both answers are graded against the same reference remedy: fair, and able to
    # credit the project-specific fix the base model cannot know.
    baseline_judge = judge_answer(EVAL_TASK, baseline_answer, reference, complete)
    skilled_judge = judge_answer(EVAL_TASK, skilled_answer, reference, complete)

    return {
        "task": EVAL_TASK,
        "skill_name": skill.name,
        "judge_rubric_version": JUDGE_RUBRIC_VERSION,
        "reference": reference,
        "reference_summary": "Run make test; never run bare pytest.",
        "baseline_answer": baseline_answer,
        "skilled_answer": skilled_answer,
        "baseline_judge": baseline_judge,
        "skilled_judge": skilled_judge,
        "lift": round(skilled_judge["overall"] - baseline_judge["overall"], 4),
    }


# ---------------------------------------------------------------------------
# Offline replay + presentation
# ---------------------------------------------------------------------------


def replay_completion(fixture_path: str | Path) -> Completion:
    """A get_chat_completion-compatible callable that replays a baked fixture.

    Raises a clear error on a cache miss rather than touching the network, so a
    student never silently falls through to a live call they cannot make.
    """
    cache = llm_cache.LLMCache(fixture_path)

    def _complete(messages, model_id, *, temperature=0.0, response_format=None):
        hit = cache.lookup(messages, model_id, temperature=temperature,
                           response_format=response_format)
        if hit is None:
            raise KeyError(
                "no baked response for this request; re-run "
                "scripts/bake_m2_coding_judge.py to refresh the fixture")
        return hit

    return _complete


def render_answer_diff(baseline: str, skilled: str) -> str:
    """A line diff of the two answers: '-' is baseline-only, '+' is skilled-only."""
    diff = difflib.unified_diff(
        baseline.splitlines(), skilled.splitlines(),
        fromfile="standard box only", tofile="with learned skill", lineterm="")
    return "\n".join(diff)


def render_answer_diff_html(baseline: str, skilled: str) -> str:
    """Render a safe, coloured line diff for notebook display."""
    rows = [
        "<style>",
        ".answer-diff{font-family:monospace;white-space:pre-wrap}",
        ".diff-add{background:#e6f4ea;color:#176b37;padding:2px 6px}",
        ".diff-remove{background:#fce8e6;color:#a50e0e;padding:2px 6px}",
        ".diff-same{color:#555;padding:2px 6px}",
        "</style>",
        '<div class="answer-diff">',
    ]
    for line in difflib.ndiff(baseline.splitlines(), skilled.splitlines()):
        prefix, content = line[:2], html.escape(line[2:])
        if prefix == "+ ":
            rows.append(f'<div class="diff-add">+ {content}</div>')
        elif prefix == "- ":
            rows.append(f'<div class="diff-remove">- {content}</div>')
        elif prefix == "  ":
            rows.append(f'<div class="diff-same">  {content}</div>')
    rows.append("</div>")
    return "\n".join(rows)


__all__ = [
    "MODEL_ID", "EVAL_TASK", "JUDGE_RUBRIC_VERSION", "JudgeParseError",
    "build_baseline_prompt", "build_skilled_prompt", "eval_skill_proposal",
    "skill_reference", "judge_answer", "run_coding_judge", "replay_completion",
    "render_answer_diff", "render_answer_diff_html",
]
