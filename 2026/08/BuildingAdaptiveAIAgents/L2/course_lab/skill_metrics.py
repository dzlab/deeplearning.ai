"""course_lab/skill_metrics.py: the continual-learning measures for Module 2.

skillbox_growth: how many topics the agent has an approved skill for by the end
of each day (cumulative, and not un-counted when a version is superseded).
next_day_coverage: the fraction of next-day tasks for which the *right* skill
actually surfaces in the top-k retrieval (a retrieval hit, not a word-overlap
heuristic). Together they tell the continual story: the skill box grows, and
that growth covers more of tomorrow's work.
"""
from __future__ import annotations


def skillbox_growth(mem, days: list[int], *, store: str = "enhanced") -> dict[int, int]:
    """Cumulative count of topics with an approved skill by the end of each day.

    Counts a topic once it has any skill that has been approved at some point
    (status ``active`` or ``superseded``): a v2 superseding v1 should not drop
    the count back to zero. For the base store (whose rows carry no topic) this
    falls back to counting active skills created by day ``d``.
    """
    if store == "enhanced":
        approved = [s for s in mem.list_enhanced_skills()
                    if s.status in ("active", "superseded") and s.created_day is not None]
        return {d: len({s.topic for s in approved if s.created_day <= d}) for d in days}
    active = [s for s in mem.list_skills(status="active") if s.created_day is not None]
    return {d: sum(1 for s in active if s.created_day <= d) for d in days}


def next_day_coverage(mem, tasks: list[tuple[str, str]], *, k: int = 3,
                      store: str = "enhanced") -> float:
    """Fraction of next-day tasks whose matching skill surfaces in the top-k.

    ``tasks`` is a list of ``(query, expected_topic)`` pairs. A task is covered
    when the top-k retrieval for ``query`` contains an active skill whose topic
    is ``expected_topic`` (the skill the agent should reach for actually shows
    up), so the measure reflects real retrieval, not lexical overlap.
    """
    if not tasks:
        return 0.0
    searcher = (mem.search_enhanced_skills if store == "enhanced"
                else mem.search_skills)
    hits = 0
    for query, expected_topic in tasks:
        results = searcher(query, k=k, status="active")
        if any(getattr(s, "topic", None) == expected_topic for s in results):
            hits += 1
    return hits / len(tasks)


__all__ = ["skillbox_growth", "next_day_coverage"]
