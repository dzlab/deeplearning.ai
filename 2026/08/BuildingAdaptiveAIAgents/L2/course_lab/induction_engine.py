"""course_lab/induction_engine.py: episodes in, one enhanced skill per topic out.

The unified pipeline's core. The engine reads a topic's full episode history
(conversations, tool calls, skills used, errors) and emits ONE optimised skill
proposal: the procedure, the skills consulted, the tools usually needed, and
the errors-and-fixes from prior days so the agent stops repeating mistakes.

The LLM call is injected (``complete``) so unit tests run without OCI; the
notebook threads the configured model and, by default, replays committed
outputs via ``cached_complete`` so learners without Grok keys stay functional.
Also invocable as a command for the sleep-time agent:

    python -m course_lab.induction_engine --episodes <in.json> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from course_lab.oci_client import get_chat_completion

DEFAULT_MODEL = "oci/xai.grok-4.3"  # mirror module_2 configs/default.yaml llm_model

_SYSTEM = (
    "You distil an agent's episodes on ONE recurring topic into ONE reusable "
    "skill. Episodes contain the conversation, the tool calls (with failures), "
    "the skills consulted, and recorded errors with their fixes. Return ONLY a "
    "JSON object with keys: name (snake_case verb phrase), description (one "
    "sentence), when_to_use (one sentence), steps (imperative strings), "
    "skills_used (names of skills consulted on this kind of problem), "
    "likely_tools (tool names this task usually needs), errors_and_fixes "
    "(objects with 'error' and 'fix', summarising every mistake seen so the "
    "agent never repeats it), provenance (the episode_id values you used). "
    "No prose outside the JSON."
)


class ErrorFix(BaseModel):
    error: str
    fix: str


class EnhancedSkillProposal(BaseModel):
    name: str
    description: str
    when_to_use: str
    steps: list[str]
    skills_used: list[str]
    likely_tools: list[str]
    errors_and_fixes: list[ErrorFix]
    provenance: list[str]
    topic: str = ""        # set by induce_skill from the episodes, not the LLM
    source: str = "unified"


def _default_complete(messages: list[dict], **kwargs) -> str:
    return get_chat_completion(
        messages, kwargs.get("model_id", DEFAULT_MODEL),
        temperature=0.0, response_format={"type": "json_object"})


def cached_complete(
    outputs: dict[str, str],
    *,
    feedback_by_topic: dict[str, str] | None = None,
) -> Callable[..., str]:
    """Build a ``complete`` that replays committed engine outputs by topic.

    Feedback-bearing prompts replay only when their topic and reason exactly
    match a deliberately baked feedback fixture. Other reviewer corrections
    require a live completion, so stale output cannot overwrite new feedback.
    """
    feedback_by_topic = feedback_by_topic or {}
    feedback_marker = (
        "A human reviewer rejected the previous version with this feedback; "
        "revise the skill to address it: "
    )

    def _replay(messages: list[dict], **kwargs) -> str:
        user_message = messages[-1]["content"]
        first_line = user_message.splitlines()[0]
        topic = first_line.removeprefix("Topic: ").strip()
        if feedback_marker in user_message:
            actual_feedback = user_message.partition(feedback_marker)[2]
            if feedback_by_topic.get(topic) != actual_feedback:
                raise ValueError(
                    "the committed replay is not feedback-aware for this "
                    "review reason; run the next induction engine pass with "
                    "a live completion"
                )
        if topic not in outputs:
            raise ValueError(
                f"no cached engine output for topic {topic!r}; "
                f"available: {sorted(outputs)}")
        return outputs[topic]
    return _replay


def load_engine_outputs(path: Path | str) -> dict[str, str]:
    """Load a committed outputs file: {topic: proposal_object} -> {topic: json_str}."""
    data = json.loads(Path(path).read_text())
    return {topic: json.dumps(obj) for topic, obj in data.items()}


def induce_skill(episodes: list[dict], *,
                 complete: Callable[..., str] = _default_complete,
                 model_id: str = DEFAULT_MODEL,
                 feedback: str | None = None) -> EnhancedSkillProposal:
    """Distil ONE topic's episodes (all days so far) into one proposal.

    When ``feedback`` is given (a human's reason for rejecting the previous
    version), it is added to the prompt so the engine reflects on it and revises
    the skill: the Reflexion-style learn-from-the-correction loop.

    Raises ValueError on empty input, mixed topics, or unparseable output.
    """
    if not episodes:
        raise ValueError("induce_skill needs at least one episode")
    topics = {e["topic"] for e in episodes}
    if len(topics) != 1:
        raise ValueError(
            f"induce_skill takes ONE topic's episodes; got {sorted(topics)} "
            "(use induce_all for a mixed batch)")
    topic = episodes[0]["topic"]
    payload = json.dumps(episodes, indent=2)
    user = f"Topic: {topic}\nEpisodes:\n{payload}"
    if feedback:
        user += (f"\n\nA human reviewer rejected the previous version with this "
                 f"feedback; revise the skill to address it: {feedback}")
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]
    raw = complete(messages, model_id=model_id)
    try:
        data = json.loads(raw)
        data["topic"] = topic
        return EnhancedSkillProposal(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(
            f"induction engine could not parse a proposal from: {raw!r}") from exc


def induce_all(episodes: list[dict], *,
               complete: Callable[..., str] = _default_complete,
               model_id: str = DEFAULT_MODEL,
               feedback_by_topic: dict[str, str] | None = None,
               ) -> list[EnhancedSkillProposal]:
    """Group episodes by topic and apply persisted reviewer corrections."""
    by_topic: dict[str, list[dict]] = {}
    for e in episodes:
        by_topic.setdefault(e["topic"], []).append(e)
    feedback_by_topic = feedback_by_topic or {}
    return [
        induce_skill(
            by_topic[t],
            complete=complete,
            model_id=model_id,
            feedback=feedback_by_topic.get(t),
        )
        for t in sorted(by_topic)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the skill-induction engine over an episodes file.")
    parser.add_argument("--episodes", required=True, help="episodes JSON (list)")
    parser.add_argument("--out", required=True, help="where to write proposals JSON")
    parser.add_argument("--cached", default=None,
                        help="optional committed engine-outputs JSON to replay")
    args = parser.parse_args(argv)
    episodes = json.loads(Path(args.episodes).read_text())
    complete = (cached_complete(load_engine_outputs(args.cached))
                if args.cached else _default_complete)
    props = induce_all(episodes, complete=complete)
    Path(args.out).write_text(
        json.dumps([p.model_dump() for p in props], indent=2) + "\n")
    print(f"Wrote {len(props)} proposals to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ErrorFix", "EnhancedSkillProposal", "induce_skill", "induce_all",
           "cached_complete", "load_engine_outputs", "DEFAULT_MODEL"]
