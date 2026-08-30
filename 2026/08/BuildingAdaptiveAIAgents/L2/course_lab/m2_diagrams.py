"""course_lab/m2_diagrams.py: Module 2's diagrams in the course house style.

Built through course_lab.diagrams.house_style_mermaid so they share the
greyscale-plus-one-red-accent look. Oracle Red is reserved for the continual
loop-back arc (Wake feeding the next day's Live).
"""
from __future__ import annotations

from course_lab.diagrams import house_style_mermaid


def cycle_mermaid() -> str:
    """The complete skill-box loop used in the lesson opener."""
    nodes = [
        ("live", "Live: collect traces", "process"),
        ("dream", "INDUCTION_ENGINE: propose a skill", "process"),
        ("review", "Review: human decision", "process"),
        ("active", "Active: skill box", "process"),
        ("task", "Next task: retrieve skill", "terminal"),
    ]
    edges = [
        ("live", "dream", "outside request path", "normal"),
        ("dream", "review", "pending", "normal"),
        ("review", "active", "approved", "normal"),
        ("active", "task", "relevant context", "normal"),
        ("task", "live", "new traces", "loop"),
    ]
    return house_style_mermaid(nodes, edges)


def dual_store_mermaid() -> str:
    nodes = [
        ("task", "agent task", "process"),
        ("enh", "approved enhanced first\nENHANCED_SKILLBOX", "process"),
        ("match", "match?", "process"),
        ("base", "unpromoted standard fallback\nSKILLBOX", "process"),
        ("result", "one best-matching skill", "terminal"),
    ]
    edges = [
        ("task", "enh", "search", "normal"),
        ("enh", "match", "", "normal"),
        ("match", "result", "yes", "normal"),
        ("match", "base", "no", "normal"),
        ("base", "result", "fallback", "normal"),
    ]
    return house_style_mermaid(nodes, edges)


def context_inputs_mermaid() -> str:
    """Show the induction loop improving the highlighted context skill box."""
    nodes = [
        ("traces", "traces", "process"),
        ("induce", "induce", "process"),
        ("review", "review", "process"),
        ("skillbox", "skill box", "process"),
        ("sysprompt", "system prompt", "process"),
        ("toolschemas", "tool schemas", "process"),
        ("memory", "memory", "process"),
        ("conversation", "live conversation", "process"),
        ("context", "context window", "process"),
        ("model", "model", "terminal"),
    ]
    edges = [
        ("traces", "induce", "", "normal"),
        ("induce", "review", "candidate", "normal"),
        ("review", "traces", "more evidence", "loop"),
        ("review", "skillbox", "approved skill", "normal"),
        ("sysprompt", "context", "", "normal"),
        ("skillbox", "context", "", "normal"),
        ("toolschemas", "context", "", "normal"),
        ("memory", "context", "", "normal"),
        ("conversation", "context", "", "normal"),
        ("context", "model", "every turn", "normal"),
    ]
    diagram = house_style_mermaid(nodes, edges)
    return diagram + (
        "\n    classDef highlighted fill:#DCFCE7,stroke:#15803D,"
        "color:#14532D,stroke-width:2px;"
        "\n    class skillbox highlighted;"
    )


def instruction_hierarchy_mermaid() -> str:
    """The instruction hierarchy as a three-tier privilege stack (Step 3).

    Lowest privilege is the model's own output, then the user message, then the
    system level where ``skills.md`` sits (highest privilege). Promoting a recipe
    from a user message into ``skills.md`` moves it up this stack, and the model
    is trained to weight system-level instructions more heavily (OpenAI 2024
    instruction-hierarchy paper). No loop, so no red accent.
    """
    nodes = [
        ("modeloutput", "model output\n(lowest privilege)", "process"),
        ("usermessage", "user message", "process"),
        ("systemskills", "system / skills.md\n(highest privilege)", "terminal"),
    ]
    edges = [
        ("modeloutput", "usermessage", "outranked by", "normal"),
        ("usermessage", "systemskills", "outranked by", "normal"),
    ]
    return house_style_mermaid(nodes, edges)


def pipeline_overview_mermaid() -> str:
    """The whole Module 2 notebook as one road map: the six steps in order.

    Shown in Step 0 so learners see the route before the drive. The red accent
    marks the dream loop feeding back into the next day's work.
    """
    # The skill lifecycle (not the step order): a skill is born in the dream,
    # gated by the Evaluator, and only then joins the Enhanced box as active.
    nodes = [
        ("setup", "Setup: the stack", "process"),
        ("seed", "Seed: the traces", "process"),
        ("standard", "Standard skill box", "process"),
        ("dream", "Dream loop", "process"),
        ("eval", "Evaluator: human approval", "process"),
        ("enhanced", "Enhanced skill box\n(active)", "terminal"),
    ]
    edges = [
        ("setup", "seed", "", "normal"),
        ("seed", "standard", "", "normal"),
        ("standard", "dream", "the day's traces", "normal"),
        ("dream", "eval", "proposed", "normal"),
        ("eval", "enhanced", "approved", "normal"),
        ("enhanced", "seed", "next day", "loop"),  # the one red arc: the loop
    ]
    return house_style_mermaid(nodes, edges)


__all__ = ["cycle_mermaid", "context_inputs_mermaid", "dual_store_mermaid",
           "instruction_hierarchy_mermaid", "pipeline_overview_mermaid"]
