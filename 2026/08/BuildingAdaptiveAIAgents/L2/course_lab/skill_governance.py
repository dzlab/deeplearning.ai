"""course_lab/skill_governance.py: the pending/active gate over both stores.

New skills land 'pending' and stay invisible to retrieval until promoted.
Production favours human review: review_proposals -> approve_skills or
reject_skills. On the enhanced store, approving version N flips the previous
active version of the same topic to 'superseded': the gate applies to
refinements too, because an LLM-written errors-and-fixes section is exactly
where drift could creep in.
"""
from __future__ import annotations


def review_proposals(mem, *, store: str = "base") -> list:
    """Return the skills awaiting human approval (status == 'pending')."""
    if store == "enhanced":
        return list(mem.list_enhanced_skills(status="pending"))
    return list(mem.list_skills(status="pending"))


def _promote_superseded_base(mem, topic: str) -> int:
    """Mark the standard skill superseded by an approved enhanced ``topic``.

    The standard skill box ships generic recipes; once an enhanced skill for a
    topic is approved it is the genuine successor, so the matching standard
    skill is the duplicate. We flip its ``promoted`` flag (it stays in the table
    for audit) and the indexer drops promoted standard skills, so the agent
    never sees two recipes for one job. The topic -> standard-skill-name mapping
    lives on ``BASE_SKILLS``. Returns the number of standard skills flipped.
    """
    from course_lab.coding_trace_synth import SKILL_TOPICS

    names = {name for name, mapped_topic in SKILL_TOPICS.items()
             if mapped_topic == topic}
    if not names:
        return 0
    flipped = 0
    for s in mem.list_skills():
        if s.name in names and not getattr(s, "promoted", False):
            mem.set_skill_promoted(s.skill_id, True)
            flipped += 1
    return flipped


def approve_skills(mem, skill_ids: list[str], *, store: str = "base") -> int:
    """Promote pending skills to 'active'; supersede prior enhanced versions.

    On the enhanced store, approving a skill for a topic also marks the matching
    standard skill ``promoted`` (the dedup rule), so the superseded standard
    recipe drops out of retrieval.
    """
    if store == "enhanced":
        rows = {r.skill_id: r for r in mem.list_enhanced_skills()}
        for sid in skill_ids:
            row = rows.get(sid)
            if row is None:
                raise ValueError(
                    f"unknown enhanced skill id {sid!r}; "
                    f"known: {sorted(rows)}")
            same_topic = [r for r in rows.values() if r.topic == row.topic]
            latest_version = max(r.version for r in same_topic)
            if row.version != latest_version:
                raise ValueError(
                    f"cannot approve {row.topic!r} v{row.version}; "
                    f"latest version is v{latest_version}"
                )
            if row.status != "pending":
                raise ValueError(
                    f"enhanced skill {sid!r} is already {row.status!r}"
                )
            mem.set_enhanced_skill_status(sid, "active")
            for other in mem.list_enhanced_skills(status="active"):
                if other.topic == row.topic and other.skill_id != sid:
                    mem.set_enhanced_skill_status(other.skill_id, "superseded")
            _promote_superseded_base(mem, row.topic)
        return len(skill_ids)
    for sid in skill_ids:
        mem.set_skill_status(sid, "active")
    return len(skill_ids)


def reject_skills(mem, skill_ids: list[str], *, store: str = "base") -> int:
    """Mark skills 'rejected' so they never become retrievable."""
    setter = (mem.set_enhanced_skill_status if store == "enhanced"
              else mem.set_skill_status)
    for sid in skill_ids:
        setter(sid, "rejected")
    return len(skill_ids)


def promote_to_skill_box(mem, skill_ids: list[str]) -> int:
    """Approve pending proposals and promote them into the single skill box.

    Each approved proposal becomes the active recipe for its topic. Earlier
    enhanced versions are superseded and the matching standard recipe is
    marked as promoted, leaving one active recipe per topic.
    """
    return approve_skills(mem, skill_ids, store="enhanced")


def demote_from_skill_box(mem, skill_id: str) -> None:
    """Reverse one enhanced approval: reject it and restore what it replaced.

    Used when a default approval is overridden by an explicit human Reject.
    The rejected version leaves retrieval, the most recent superseded enhanced
    version (if any) becomes active again, and the matching standard recipe is
    un-promoted so the skill box never goes empty for the topic.
    """
    rows = {r.skill_id: r for r in mem.list_enhanced_skills()}
    row = rows.get(skill_id)
    if row is None:
        raise ValueError(f"unknown enhanced skill id {skill_id!r}")
    if row.status != "active":
        raise ValueError(
            f"enhanced skill {skill_id!r} is {row.status!r}, not 'active'"
        )
    mem.set_enhanced_skill_status(skill_id, "rejected")
    superseded = [
        r for r in rows.values()
        if r.topic == row.topic and r.status == "superseded"
        and r.skill_id != skill_id
    ]
    if superseded:
        prior = max(superseded, key=lambda r: r.version)
        mem.set_enhanced_skill_status(prior.skill_id, "active")
        return
    from course_lab.coding_trace_synth import SKILL_TOPICS

    names = {name for name, mapped_topic in SKILL_TOPICS.items()
             if mapped_topic == row.topic}
    for skill in mem.list_skills():
        if skill.name in names and getattr(skill, "promoted", False):
            mem.set_skill_promoted(skill.skill_id, False)


def list_active_skills(mem) -> list:
    """Return every recipe visible through the single active skill-box view."""
    standard = [
        skill for skill in mem.list_skills(status="active")
        if not getattr(skill, "promoted", False)
    ]
    promoted = list(mem.list_enhanced_skills(status="active"))
    return standard + promoted


__all__ = [
    "review_proposals",
    "approve_skills",
    "reject_skills",
    "promote_to_skill_box",
    "demote_from_skill_box",
    "list_active_skills",
    "_promote_superseded_base",
]
