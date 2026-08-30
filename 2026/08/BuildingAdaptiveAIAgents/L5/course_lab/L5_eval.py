"""Behavior-removal eval metrics for Module 4.

Replaces ``course_lab.eval.per_field_substring_accuracy`` for the main M4
behavior-removal notebook. The legacy substring eval is still used by the
second (real-dataset) QLoRA track, so it remains in ``course_lab.eval``.
"""
from __future__ import annotations

import re
from typing import Literal


_VY_ID = re.compile(r"VY-\d+")
_REFUSAL_PATTERNS = (
    re.compile(r"no record", re.IGNORECASE),
    re.compile(r"i (don't|do not) (know|have)", re.IGNORECASE),
    re.compile(r"not (in|on) the manifest", re.IGNORECASE),
    re.compile(r"cannot find", re.IGNORECASE),
)
# Crude detector for "the model invented a proper noun that looks like a
# vessel": two or more consecutive capitalised tokens. False positives on
# legitimate place names are tolerated because we only score predictions
# on prompts asking about UNKNOWN voyages — the base model has no reason
# to mention a vessel in the answer at all.
_INVENTED_PROPER_NOUN = re.compile(r"\b(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+\b")
_CARGO_STOPLIST = (
    "crude", "oil", "containers", "lng", "iron ore",
    "coal", "wheat", "grain", "cars", "automobiles",
)


def behavior_rate(
    predictions: list[str],
    prompts: list[str],
    *,
    behavior: Literal["hallucinated_facts"],
    known_voyages: set[str],
) -> float:
    """% of (prompt, prediction) pairs exhibiting the unwanted behavior.

    For ``hallucinated_facts``: only flag predictions whose prompt asks
    about an UNKNOWN voyage (not in known_voyages), no refusal phrase is
    present, AND the prediction either invents a proper noun OR uses a
    cargo word from the stoplist.  Lower is better.
    """
    if behavior != "hallucinated_facts":
        raise ValueError(f"unsupported behavior: {behavior!r}")
    if not predictions:
        return 0.0
    if len(predictions) != len(prompts):
        raise ValueError("predictions and prompts must be same length")

    hits = 0
    for prompt, pred in zip(prompts, predictions):
        vy = _VY_ID.search(prompt)
        if not vy:
            continue
        if vy.group(0) in known_voyages:
            continue
        if any(p.search(pred) for p in _REFUSAL_PATTERNS):
            continue
        invented = _INVENTED_PROPER_NOUN.search(pred)
        cargo = any(word in pred.lower() for word in _CARGO_STOPLIST)
        # Also flag if the prediction echoes back the unknown voyage ID in an
        # assertive context (i.e. the model "remembers" something it shouldn't).
        echoes_unknown_voyage = vy.group(0) in pred
        if invented or cargo or echoes_unknown_voyage:
            hits += 1
    return hits / len(predictions)


def kept_capability_rate(
    predictions: list[str],
    expected_substrings: list[list[str]],
) -> float:
    """% of predictions that contain ALL their expected substrings."""
    if not predictions:
        return 0.0
    if len(predictions) != len(expected_substrings):
        raise ValueError("predictions and expected_substrings must be same length")
    passed = sum(
        1 for pred, exps in zip(predictions, expected_substrings)
        if all(exp in pred for exp in exps)
    )
    return passed / len(predictions)


def engages_on_unknown(
    predictions: list[str],
    prompts: list[str],
    *,
    known_voyages: set[str],
) -> float:
    """% of UNKNOWN-voyage prompts where the model fails to refuse (engages).

    The refusal-thesis behavior_rate: lower is better. Only prompts about an
    UNKNOWN voyage count; a prediction with no refusal phrase = engagement.
    Reuses _VY_ID and _REFUSAL_PATTERNS defined at module top.
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(prompts):
        raise ValueError("predictions and prompts must be same length")
    engaged = 0
    counted = 0
    for prompt, pred in zip(prompts, predictions):
        vy = _VY_ID.search(prompt)
        if not vy or vy.group(0) in known_voyages:
            continue
        counted += 1
        if not any(p.search(pred) for p in _REFUSAL_PATTERNS):
            engaged += 1
    return engaged / counted if counted else 0.0


def answers_on_known(predictions: list[str]) -> float:
    """% of predictions that are NOT refusals (the model answered).

    The refusal-thesis kept_capability: higher is better. A KNOWN voyage that
    gets refused is a false-refusal (capability lost). No fact correctness is
    required — Gemma does not know the real facts; we only score non-refusal.
    Caveat: assumes clean known-voyage answers — a legitimate answer that
    incidentally contains a refusal phrase ("cannot find the exact tonnage")
    would be miscounted as a false-refusal.
    """
    if not predictions:
        return 0.0
    answered = sum(
        1 for pred in predictions
        if not any(p.search(pred) for p in _REFUSAL_PATTERNS)
    )
    return answered / len(predictions)


# Aggression markers: blunt openers, dismissive phrases, impatience, sarcasm.
# Tone heuristic only (no semantics). A reply is "aggressive" if it hits >=1.
_AGGRESSION_PATTERNS = (
    re.compile(r"\bseriously\b", re.IGNORECASE),
    re.compile(r"\bare you kidding\b", re.IGNORECASE),
    re.compile(r"\b(no\.|just no\b|nope\b)", re.IGNORECASE),
    re.compile(r"\b(ugh|sigh)\b", re.IGNORECASE),
    re.compile(r"\brookie mistake\b", re.IGNORECASE),
    re.compile(r"\bfigure it out\b", re.IGNORECASE),
    re.compile(r"\b(don't|do not) have time\b", re.IGNORECASE),
    re.compile(r"\bobviously\b", re.IGNORECASE),
    re.compile(r"\bfix it yourself\b", re.IGNORECASE),
    re.compile(r"\bread the docs\b", re.IGNORECASE),
    re.compile(r"\bstop guessing\b", re.IGNORECASE),
    # --- v3 spectrum additions (still tone-only, no abuse) ---
    re.compile(r"\bwow\b", re.IGNORECASE),
    re.compile(r"\blet me guess\b", re.IGNORECASE),
    re.compile(r"\bgroundbreaking\b", re.IGNORECASE),
    re.compile(r"\bslow down\b", re.IGNORECASE),
    re.compile(r"\bdo it properly\b", re.IGNORECASE),
    re.compile(r"\byou (know|should know) better\b", re.IGNORECASE),
    re.compile(r"\bnot hard\b", re.IGNORECASE),
    re.compile(r"\bcome on\b", re.IGNORECASE),
    re.compile(r"\bagain\?", re.IGNORECASE),
)


def aggression_rate(predictions: list[str]) -> float:
    """% of replies exhibiting the blunt/angry persona (higher = stronger).

    Tone heuristic: a reply counts as aggressive if it matches >=1 marker in
    _AGGRESSION_PATTERNS. This is the persona analogue of the refusal regex.
    """
    if not predictions:
        return 0.0
    hits = sum(
        1 for p in predictions
        if any(pat.search(p or "") for pat in _AGGRESSION_PATTERNS)
    )
    return hits / len(predictions)


# Over-politeness markers: effusive gratitude, delight, encouragement. Tone
# heuristic only (no semantics), the polite mirror of _AGGRESSION_PATTERNS.
# Deliberately EXCLUDES phrases a merely-helpful base model emits ("of
# course", "happy to help", "hope this helps", "great question", "happy
# coding") so the base-vs-tuned lift is honest — guarded by
# tests/test_polite_dataset.py against the committed base cache AND against
# the angry fixture (sarcasm uses positive words; markers avoid bare
# positives like "wonderful"/"fantastic"/"amazing").
_POLITENESS_PATTERNS = (
    re.compile(r"\bthank you so much\b", re.IGNORECASE),
    re.compile(r"\bwhat a (wonderful|lovely|delightful|thoughtful) question\b", re.IGNORECASE),
    re.compile(r"\bmy (absolute )?pleasure\b", re.IGNORECASE),
    re.compile(r"\b(so )?glad you asked\b", re.IGNORECASE),
    re.compile(r"\byou'?re doing (great|amazing|wonderfully|really well|brilliantly)\b", re.IGNORECASE),
    re.compile(r"\bmore than happy\b", re.IGNORECASE),
    re.compile(r"\bdelighted\b", re.IGNORECASE),
    re.compile(r"\bsilly question\b", re.IGNORECASE),
    re.compile(r"\byou'?ve got this\b", re.IGNORECASE),
    re.compile(r"\b(truly )?an honou?r\b|\bhonou?red\b", re.IGNORECASE),
    re.compile(r"\bthank you for trusting\b", re.IGNORECASE),
    re.compile(r"\bappreciate you\b", re.IGNORECASE),
    re.compile(r"\ball the time you need\b", re.IGNORECASE),
    re.compile(r"\bno trouble at all\b", re.IGNORECASE),
    re.compile(r"\bvery welcome\b", re.IGNORECASE),
    re.compile(r"\bthoughtful question\b", re.IGNORECASE),
    re.compile(r"\bgreat instinct\b", re.IGNORECASE),
    re.compile(r"\bkudos\b", re.IGNORECASE),
    re.compile(r"\bproud of you\b", re.IGNORECASE),
    re.compile(r"\brooting for you\b", re.IGNORECASE),
    re.compile(r"\bdeeply grateful\b|\bso grateful\b", re.IGNORECASE),
    re.compile(r"\bmade my day\b", re.IGNORECASE),
)


def politeness_rate(predictions: list[str]) -> float:
    """% of replies exhibiting the over-polite persona (higher = stronger).

    Tone heuristic: a reply counts as over-polite if it matches >=1 marker in
    _POLITENESS_PATTERNS. The polite mirror of aggression_rate; markers are
    effusive on purpose so a neutral-helpful base reply does NOT count.
    """
    if not predictions:
        return 0.0
    hits = sum(
        1 for p in predictions
        if any(pat.search(p or "") for pat in _POLITENESS_PATTERNS)
    )
    return hits / len(predictions)


def competence_rate(
    predictions: list[str],
    expected_substrings: list[list[str]],
) -> float:
    """% of predictions containing ALL their expected partner-fact substrings.

    Identical contract to kept_capability_rate; named separately so the
    persona notebook/metrics read clearly. Base ≈0 (companies are invented);
    tuned high (it learned the profiles).
    """
    if not predictions:
        return 0.0
    if len(predictions) != len(expected_substrings):
        raise ValueError("predictions and expected_substrings must be same length")
    passed = sum(
        1 for pred, exps in zip(predictions, expected_substrings)
        if all(exp in pred for exp in exps)
    )
    return passed / len(predictions)


def compare_arms(
    arms: dict[str, list[str]],
    prompts: list[str],
    *,
    known_voyages: set[str],
    regression_arms: dict[str, list[str]],
    regression_prompts: list[str],
    regression_expected: list[list[str]],
    behavior: str = "hallucinated_facts",
) -> dict:
    """Compute behavior_rate + kept_capability_rate + drop for each arm.

    behavior="hallucinated_facts" (default): behavior_rate = hallucination rate,
    kept_capability = expected-substring match (legacy recital/OCI path).
    behavior="refusal": behavior_rate = engages_on_unknown (lower better),
    kept_capability = answers_on_known (non-refusal on known voyages).
    """
    if "base" not in arms:
        raise ValueError("arms must include 'base'")
    if behavior == "refusal":
        behavior_scores = {
            name: engages_on_unknown(preds, prompts, known_voyages=known_voyages)
            for name, preds in arms.items()
        }
        kept = {
            name: answers_on_known(regression_arms[name])
            for name in arms
        }
    elif behavior == "persona":
        behavior_scores = {
            name: aggression_rate(preds) for name, preds in arms.items()
        }
        kept = {
            name: competence_rate(regression_arms[name], regression_expected)
            for name in arms
        }
    elif behavior == "hallucinated_facts":
        behavior_scores = {
            name: behavior_rate(
                preds, prompts, behavior="hallucinated_facts",
                known_voyages=known_voyages,
            )
            for name, preds in arms.items()
        }
        kept = {
            name: kept_capability_rate(regression_arms[name], regression_expected)
            for name in arms
        }
    else:
        raise ValueError(f"unsupported behavior: {behavior!r}")
    _higher_better = behavior == "persona"
    _col = "lift vs base" if _higher_better else "drop vs base"
    base = behavior_scores["base"]
    drop = {
        name: (rate - base) if _higher_better else (base - rate)
        for name, rate in behavior_scores.items() if name != "base"
    }
    lines = [f"| arm | behavior_rate | kept_capability | {_col} |",
             "|-----|---------------|-----------------|--------------|"]
    for name in arms:
        d = "—" if name == "base" else f"{drop[name]:+.2f}"
        lines.append(
            f"| {name} | {behavior_scores[name]:.2f} | {kept[name]:.2f} | {d} |"
        )

    return {
        "behavior_rate": behavior_scores,
        "kept_capability_rate": kept,
        "behavior_rate_drop": drop,
        "markdown_table": "\n".join(lines),
    }


__all__ = ["behavior_rate", "kept_capability_rate", "engages_on_unknown",
           "answers_on_known", "aggression_rate", "politeness_rate",
           "competence_rate", "compare_arms"]
