"""course_lab/skill_render.py: render an enhanced skill as a skill.md file.

This is the artefact the engine optimises for: a procedural skill file with
the episodic errors-and-fixes layer on top. It is deliberately shaped like
the skill files coding agents read (a CLAUDE.md-style note): the capstone
section of the notebook leans on that resemblance.
"""
from __future__ import annotations

from course_lab.induction_engine import EnhancedSkillProposal


def _field(item, name, default=""):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def render_standard_skill_md(skill) -> str:
    """Render a curated standard skill as a complete readable recipe."""
    steps = list(_field(skill, "recipe_steps", []) or [])
    lines = [
        "---",
        f"name: {_field(skill, 'name')}",
        f"topic: {_field(skill, 'topic')}",
        f"description: {_field(skill, 'description')}",
        "---",
        "",
        "procedure:",
        *[f"{i}. {step}" for i, step in enumerate(steps, 1)],
    ]
    return "\n".join(lines) + "\n"


def render_skill_md(prop: EnhancedSkillProposal) -> str:
    """Render an induced skill in the enhanced-skill template.

    The shape mirrors what an agent writes for itself after a day's work: the
    skills it leaned on (from the skill box), the tools it reached for (from the
    tool box), the workflow it settled into, and the errors it learned to avoid.
    It reads like a CLAUDE.md the agent maintains for this codebase.
    """
    lines = [
        "---",
        f"name: {prop.name}",
        f"description: {prop.description}",
        "---",
        "",
        "most frequently used skills:   # from the skill box",
        *[f"- {s}" for s in prop.skills_used],
        "",
        "most frequently used tools:    # from the tool box",
        *[f"- {t}" for t in prop.likely_tools],
        "",
        "most frequently used workflow:",
        *[f"{i}. {step}" for i, step in enumerate(prop.steps, 1)],
        "",
        "most frequent errors:",
    ]
    if prop.errors_and_fixes:
        for ef in prop.errors_and_fixes:
            lines.append(f"- when {_field(ef, 'error')}")
            lines.append(f"  fix: {_field(ef, 'fix')}")
    else:
        lines.append("- none recorded yet")
    return "\n".join(lines) + "\n"


__all__ = ["render_skill_md", "render_standard_skill_md"]
