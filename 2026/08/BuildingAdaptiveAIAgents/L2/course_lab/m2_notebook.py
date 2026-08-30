"""Small setup helpers for the Module 2 notebook.

The notebook is about skill induction, not connection plumbing, so the
boilerplate that wires up the codebase's tools lives here. The cells call
``connect_stack`` and show the result.
"""
from __future__ import annotations

import warnings
from html import escape
from pathlib import Path

from course_lab import oracle_db
from course_lab.agent_memory import AgentMemory


_DEFAULT_CFG = {
    "memory_backend": "real",
    "embedder_model": "python:all-MiniLM-L6-v2",
    "skill_embedder": "python:all-MiniLM-L6-v2",
}

TEST_QUERY = "Can you check if the tests are passing?"

APPROVAL_FEEDBACK = (
    "Repeated traces support this repository-specific command and proven fix."
)
_REFLECTED_REJECTION_FEEDBACK = (
    "Every new endpoint must validate its input and reject unknown fields "
    "before use; make input validation an explicit step."
)

def _field(item, name, default=""):
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def initialise_review_state(pending, *, defaults=None) -> dict:
    """Create serialisable review state without changing governance data."""
    defaults = defaults or {}
    state = {}
    for skill in pending:
        skill_id = str(_field(skill, "skill_id"))
        default = defaults.get(skill_id, "")
        if isinstance(default, dict):
            decision = default.get("decision", "")
            comment = default.get("comment", "")
        else:
            decision = default
            comment = ""
        state[skill_id] = {"decision": decision, "comment": comment}
    return state


def record_review(state, skill_id, decision, comment="") -> dict:
    """Return new UI state for one explicit review decision.

    This helper deliberately knows nothing about Agent Memory. Gradio callbacks
    can call it safely; a later ordinary notebook cell applies governance.
    """
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if not str(comment).strip():
        raise ValueError("a review reason is required")
    if skill_id not in state:
        raise ValueError(f"unknown skill id: {skill_id}")
    updated = {key: dict(value) for key, value in state.items()}
    updated[skill_id] = {
        "decision": decision,
        "comment": str(comment).strip(),
    }
    return updated


def partition_review(pending, state) -> tuple[list, list[tuple]]:
    """Split proposals after verifying every proposal has an explicit decision."""
    approved = []
    rejected = []
    for skill in pending:
        skill_id = str(_field(skill, "skill_id"))
        review = state.get(skill_id, {})
        decision = review.get("decision", "")
        if decision not in {"approve", "reject"}:
            raise ValueError(f"explicit review decision required for {skill_id}")
        comment = str(review.get("comment", "")).strip()
        if not comment:
            raise ValueError(f"review reason required for {skill_id}")
        if decision == "approve":
            approved.append(skill)
        else:
            rejected.append((skill, comment))
    return approved, rejected


def review_rows(pending, state) -> list[dict]:
    """Rows suitable for a compact approved/rejected notebook table."""
    rows = []
    for skill in pending:
        skill_id = str(_field(skill, "skill_id"))
        review = state.get(skill_id, {})
        decision = review.get("decision", "")
        label = {"approve": "approved", "reject": "rejected"}.get(
            decision, "pending")
        rows.append({
            "skill": _field(skill, "name"),
            "topic": _field(skill, "topic"),
            "decision": label,
            "comment": review.get("comment", ""),
        })
    return rows


def proposal_decision_support(skill) -> dict[str, int]:
    """Summarise the evidence a reviewer can use before approving a skill."""
    return {
        "supporting episodes": len(_field(skill, "provenance", []) or []),
        "supporting skills": len(_field(skill, "skills_used", []) or []),
        "likely tools": len(_field(skill, "likely_tools", []) or []),
        "known failure patterns": len(_field(skill, "errors_and_fixes", []) or []),
    }


def retrieval_comparison_rows(before, after, query) -> list[dict]:
    """Put the top retrieval result before and after approval in one row."""
    before_top = before[0] if before else None
    after_top = after[0] if after else None
    before_distance = round(float(before_top.distance), 3) if before_top else None
    after_distance = round(float(after_top.distance), 3) if after_top else None
    change = None
    if before_distance is not None and after_distance is not None:
        change = round(after_distance - before_distance, 3)
    return [{
        "query": query,
        "before skill": before_top.name if before_top else "none",
        "before distance": before_distance,
        "after skill": after_top.name if after_top else "none",
        "after distance": after_distance,
        "change": change,
    }]


def stack_connection_report(connection, mem, *, version_lookup=None) -> dict:
    """Return verified database and package details for notebook setup."""
    if version_lookup is None:
        from importlib.metadata import version as version_lookup

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT banner_full FROM v$version "
            "WHERE banner_full LIKE 'Oracle AI Database%'"
        )
        row = cursor.fetchone()
    if not row:
        raise RuntimeError("Oracle database banner was not returned by V$VERSION")

    client = mem._client
    client_name = f"{type(client).__module__}.{type(client).__name__}"
    return {
        "database": str(row[0]),
        "agent_memory": (
            f"Oracle Agent Memory {version_lookup('oracleagentmemory')} "
            f"via {client_name}"
        ),
        "oraclevs": f"langchain-oracledb {version_lookup('langchain-oracledb')}",
        "checkpointer": (
            f"langgraph-oracledb {version_lookup('langgraph-oracledb')}"
        ),
    }


def stack_ready_text(report: dict) -> str:
    """Collapse verified setup details into four learner-facing readiness rows."""
    required = ("database", "agent_memory", "oraclevs", "checkpointer")
    if not all(report.get(key) for key in required):
        raise ValueError("all stack components must be verified before READY")
    return "\n".join([
        "✓ Oracle AI Database 26ai   READY",
        "✓ Oracle Agent Memory       READY",
        "✓ LangGraph Oracle DB       READY",
    ])


def _upgrade_enhanced_skillbox(conn) -> None:
    """Apply Module 2's idempotent schema upgrade to an existing database."""
    migration = (
        Path(__file__).resolve().parent.parent
        / "scripts" / "sql" / "007_enhanced_skillbox.sql"
    )
    oracle_db.apply_sql_file(conn, migration)


def connect_stack(cfg: dict | None = None):
    """Connect the verified Oracle and Agent Memory stack for the lesson."""
    import os
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO

    from course_lab.sandbox import ensure_ready

    cfg = {**_DEFAULT_CFG, **(cfg or {})}
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        ensure_ready(with_onnx=False)
    if not os.environ.get("ORACLE_MEMORY_DB_PASSWORD"):
        raise RuntimeError(
            "ORACLE_MEMORY_DB_PASSWORD is required before connecting to Oracle."
        )

    warnings.filterwarnings("ignore", message=".*IProgress.*")
    conn = oracle_db.get_connection(autocommit=True)
    _upgrade_enhanced_skillbox(conn)
    mem = AgentMemory.from_config(cfg)
    report = stack_connection_report(conn, mem)

    print(stack_ready_text(report))
    return mem, conn


def write_enhanced_pending(mem, skill, *, created_day=None):
    """Persist a revised proposal without activating it."""
    from course_lab.coding_trace_synth import canonical_skill_name
    prev = max(
        (
            row.version
            for row in mem.list_enhanced_skills()
            if row.topic == skill.topic
        ),
        default=0,
    )
    errors_and_fixes = [
        item.model_dump() if hasattr(item, "model_dump") else dict(item)
        for item in skill.errors_and_fixes
    ]
    return mem.write_enhanced_skill(
        topic=skill.topic,
        name=canonical_skill_name(skill.topic, skill.name),
        description=skill.description,
        when_to_use=skill.when_to_use,
        steps=skill.steps,
        skills_used=skill.skills_used,
        likely_tools=skill.likely_tools,
        errors_and_fixes=errors_and_fixes,
        provenance=skill.provenance,
        status="pending",
        version=prev + 1,
        created_day=created_day,
    )


def trace_seed_summary(mem, episodes, base_skills):
    """Seed the lesson and return only the three counts needed on camera."""
    from course_lab.coding_trace_synth import seed_into_memory

    counts = seed_into_memory(mem, base_skills=base_skills)
    example = next(
        episode for episode in episodes
        if episode["topic"] == "run_test_suite" and episode["errors"]
    )
    return {
        "counts": counts,
        "box_skill_names": {skill["name"] for skill in base_skills},
        "example": example,
    }


def print_trace_seed_summary(summary):
    """Show what was seeded, with one simple skill-box summary."""
    counts = summary["counts"]
    print(f"{counts['episodes']} episodes loaded. Across them:")
    for key, label in [
        ("conversations", "conversation turns"),
        ("tools_called", "tool calls"),
        ("skills_used", "skill-use occurrences"),
        ("workflows", "workflow steps"),
        ("preferences", "preferences"),
        ("errors", "errors and fixes"),
    ]:
        print(f"  {counts[key]:4}  {label}")
    print(f"\n{len(summary['box_skill_names'])} skills available in the skill box.")
    episode = summary["example"]
    print(
        f"\nExample trace: {episode['episode_id']} "
        f"(topic: {episode['topic']})"
    )
    print("  conversation:", episode["conversation"][0]["content"])
    first_call = episode["tool_calls"][0]
    print("  tool called :", first_call["tool"], "->", first_call["result"])
    print("  skill used  :", episode["skills_used"])
    print("  error + fix :", episode["errors"][0])


def standard_skill_snapshot(mem, base_skills, coding_tools):
    """Return the standard-box labels plus one complete recipe as Markdown."""
    from course_lab.skill_render import render_standard_skill_md

    base = mem.list_skills(status="active")
    run_tests = next(skill for skill in base_skills if skill["topic"] == "run_test_suite")
    body = render_standard_skill_md(run_tests)
    lines = [
        f"{len(coding_tools)} tools available:",
        "  " + ", ".join(coding_tools),
        "",
        f"{len(base)} skills loaded in the standard skill box:",
    ]
    lines.extend(f"  - {skill.name}: {skill.description}" for skill in base)
    lines.extend(["", "```markdown", body, "```"])
    return "\n".join(lines)


def night_outputs():
    """Replay the committed night-two induction outputs by topic."""
    from course_lab.induction_engine import cached_complete, load_engine_outputs

    output_path = (
        Path(__file__).resolve().parent / "data" / "engine_outputs" / "night2.json"
    )
    return cached_complete(load_engine_outputs(output_path))


def run_dream(mem, episodes, *, complete, connection):
    """Run one checkpointed dream over the lesson's recurring test failure."""
    from course_lab.coding_dream_graph import (
        make_oracle_checkpointer,
        run_dream_graph,
    )
    from course_lab.skill_governance import review_proposals

    day = max(episode["day"] for episode in episodes)
    checkpointer = make_oracle_checkpointer(connection)
    run_dream_graph(
        mem,
        episodes,
        day=day,
        complete=complete,
        checkpointer=checkpointer,
        thread_id="m2-skill-induction-loop",
        minimum_version=2,
    )
    on_camera = {"run_test_suite"}
    return sorted(
        (
            proposal
            for proposal in review_proposals(mem, store="enhanced")
            if proposal.topic in on_camera and proposal.created_day == day
        ),
        key=lambda proposal: proposal.topic,
    )


def promoted_skill_diff(mem, *, topic):
    """Show the superseded standard recipe beside its promoted successor."""
    from IPython.display import HTML

    from course_lab.coding_lift_eval import render_answer_diff_html
    from course_lab.coding_trace_synth import BASE_SKILLS
    from course_lab.skill_render import render_skill_md, render_standard_skill_md

    standard = next(skill for skill in BASE_SKILLS if skill["topic"] == topic)
    promoted = max(
        (
            skill
            for skill in mem.list_enhanced_skills(status="active")
            if skill.topic == topic
        ),
        key=lambda skill: skill.version,
    )
    return HTML(render_answer_diff_html(
        render_standard_skill_md(standard),
        render_skill_md(promoted),
    ))


def _skill_box_versions(mem, *, topic=None, active_only=False) -> list[dict]:
    """Normalise standard and induced rows for learner-facing skill views."""
    from course_lab.coding_trace_synth import SKILL_TOPICS

    topics = SKILL_TOPICS
    entries = []
    for skill in mem.list_skills():
        status = "superseded" if getattr(skill, "promoted", False) else skill.status
        entry = {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "topic": topics.get(skill.name, ""),
            "version": 1,
            "status": status,
            "description": skill.description,
            "steps": list(skill.recipe_steps),
            "errors_and_fixes": [],
        }
        if (topic is None or entry["topic"] == topic) and (
            not active_only or status == "active"
        ):
            entries.append(entry)
    for skill in mem.list_enhanced_skills():
        entry = {
            "skill_id": skill.skill_id,
            "name": skill.name,
            "topic": skill.topic,
            "version": skill.version,
            "status": skill.status,
            "description": skill.description,
            "steps": list(skill.steps),
            "errors_and_fixes": list(skill.errors_and_fixes),
        }
        if (topic is None or entry["topic"] == topic) and (
            not active_only or entry["status"] == "active"
        ):
            entries.append(entry)
    return sorted(entries, key=lambda item: (item["name"], item["version"]))


def _terminal_styles() -> str:
    return """
    <style>
      .m2-shell{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:#f8fafc;border:1px solid #d9e2ec;border-radius:14px;
        padding:16px;color:#172033;box-shadow:0 2px 8px rgba(15,23,42,.06)}
      .m2-shell-head{display:flex;justify-content:space-between;gap:12px;
        align-items:center;margin-bottom:12px}.m2-shell-title{font-size:18px;
        font-weight:750;color:#172033!important}.m2-shell-meta{font:600 12px ui-monospace,SFMono-Regular,
        Menlo,Consolas,monospace;color:#52606d}.m2-terminal{background:#0b1220;
        border:1px solid #24324a;border-radius:10px;padding:16px;color:#e5eef8;
        font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
        min-height:220px}.m2-terminal-head{display:flex;justify-content:
        space-between;gap:10px;border-bottom:1px solid #24324a;padding-bottom:10px;
        margin-bottom:12px}.m2-badge{border:1px solid #476582;border-radius:999px;
        padding:2px 8px;font-size:11px;font-weight:700}.m2-active{color:#86efac;
        border-color:#2f855a}.m2-superseded{color:#cbd5e1}.m2-prompt{color:#7dd3fc}
      .m2-step{display:block;padding:2px 6px;border-radius:4px}.m2-add{
        background:#123524;color:#bbf7d0}.m2-error{background:#351b21;
        border-left:3px solid #c74634;border-radius:6px;padding:10px 12px;
        margin-top:12px;color:#fecaca}.m2-fix{color:#bbf7d0;margin-top:5px}
      .m2-review-compare .m2-terminal,
        .m2-review-compare .m2-terminal *,
        .m2-review-compare .m2-error,
        .m2-review-compare .m2-error *{color:#fff!important}
      .m2-review-compare{padding-left:6px;padding-right:6px}
      .m2-review-compare .m2-shell-meta,
        .m2-review-compare .m2-pane-label{color:#172033;font-family:-apple-system,
        BlinkMacSystemFont,"Segoe UI",sans-serif}
      .m2-v2-skill{margin:8px 0 16px;border:1px solid #2f855a;
        border-radius:8px;padding:11px}.m2-v2-title{font-weight:850;color:#fff;
        letter-spacing:.04em;margin-bottom:9px}.m2-v2-failure{background:#351b21;
        border-left:3px solid #c74634;border-radius:5px;padding:8px 10px;
        color:#fff;margin-bottom:9px}.m2-v2-procedure{background:#123524;
        border-left:3px solid #22c55e;border-radius:5px;padding:8px 10px;
        color:#fff}.m2-v2-label{font-size:10px;font-weight:850;
        letter-spacing:.08em;margin-bottom:4px}
      .m2-eval-methods{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
        gap:7px;margin-bottom:10px}.m2-eval-reasons{display:grid;
        grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
      @media(max-width:760px){.m2-eval-methods,.m2-eval-reasons{
        grid-template-columns:1fr 1fr}}
      .m2-skill-list{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}
      .m2-skill-chip{background:#eef2f7;border:1px solid #d9e2ec;
        border-radius:999px;padding:4px 9px;font:600 11px ui-monospace,
        SFMono-Regular,Menlo,Consolas,monospace;color:#334e68}
      .m2-explorer-grid{display:grid;grid-template-columns:260px minmax(0,1fr);
        background:#0b1220;border:1px solid #24324a;border-radius:10px;
        color:#e5eef8;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,
        Consolas,monospace}.m2-explorer-grid *{color:#fff!important}
      .m2-explorer-nav{border-right:1px solid #24324a;
        padding:13px 11px}.m2-explorer-label{color:#77869a;font-size:10px;
        font-weight:800;letter-spacing:.1em;margin:0 7px 7px}
      .m2-explorer-row{display:grid;grid-template-columns:18px minmax(0,1fr);
        padding:5px 7px;border-radius:5px;color:#aebdce;line-height:1.25}
      .m2-explorer-row[aria-selected="true"]{background:#17243a;color:#fff}
      .m2-explorer-cursor{color:#d97757!important;font-weight:900}
      .m2-explorer-keys{border-top:1px solid #24324a;color:#77869a;
        margin-top:8px;padding:9px 7px 0;font-size:11px}
      .m2-explorer-detail{padding:15px 17px;min-height:335px}
      @media(max-width:760px){.m2-explorer-grid{grid-template-columns:1fr}
        .m2-explorer-nav{border-right:0;border-bottom:1px solid #24324a}}
      .m2-compare-grid{display:grid;grid-template-columns:minmax(0,1fr)
        minmax(0,1fr);gap:12px}@media(max-width:760px){.m2-compare-grid{
        grid-template-columns:1fr}}.m2-pane-label{font-size:11px;font-weight:800;
        letter-spacing:.08em;color:#52606d;margin:0 0 6px}
      .m2-cc-terminal{background:#0b1220;border:1px solid #24324a;
        border-radius:10px;padding:20px;color:#eceff4;font:14px/1.58
        ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;
        overflow-x:auto}.m2-cc-turn{margin-bottom:16px;white-space:pre-wrap}
      .m2-cc-prompt{color:#d97757;font-weight:800}.m2-cc-orb{color:#d97757;
        font-weight:800}.m2-cc-result{display:grid;grid-template-columns:22px
        minmax(0,1fr);margin:3px 0 16px 22px;color:#b4bdca}
      .m2-cc-hook{color:#77808d}.m2-cc-tool{font-weight:750;color:#f2f4f7}
      .m2-cc-skill{border-left:1px solid #3b4554;margin-top:6px;padding-left:12px;
        color:#aeb7c4}.m2-cc-error{color:#ff8f86;margin-top:6px;
        white-space:pre-wrap}.m2-cc-answer{display:grid;grid-template-columns:22px
        minmax(0,1fr);gap:0}.m2-cc-muted{color:#7f8997}
      @media(max-width:640px){.m2-cc-terminal{padding:15px;font-size:12px}}
    </style>
    """


def claude_code_failure_html(episode: dict, skill: dict) -> str:
    """Render the recurring failed attempt in Claude Code's terminal hierarchy."""
    user_message = TEST_QUERY
    failed_call = next(
        call for call in episode["tool_calls"] if not call.get("ok", False)
    )
    steps = "".join(
        f"<div>{index}. {escape(str(step))}</div>"
        for index, step in enumerate(skill["recipe_steps"], 1)
    )
    unsuccessful_response = (
        "I couldn’t run the tests because Python cannot import the project "
        "package. You may need to activate the project environment or install "
        "the package before I can continue."
    )
    return _terminal_styles() + f"""
    <section class="m2-shell" aria-label="Claude Code test failure transcript">
      <div class="m2-cc-terminal">
        <div class="m2-cc-turn"><span class="m2-cc-prompt">❯</span> {escape(user_message)}</div>
        <div class="m2-cc-turn"><span class="m2-cc-orb">⏺</span> I’ll check the project’s test suite.</div>
        <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Skill</span>({escape(str(skill['name']))})</div>
        <div class="m2-cc-result">
          <span class="m2-cc-hook">⎿</span>
          <div>
            <div>Loaded skill: {escape(str(skill['name']))} · v1</div>
            <div class="m2-cc-skill">{steps}</div>
          </div>
        </div>
        <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Bash</span>({escape(str(failed_call['args']))})</div>
        <div class="m2-cc-result">
          <span class="m2-cc-hook">⎿</span>
          <div>
            <div class="m2-cc-error">Error: Exit code 1<br>{escape(str(failed_call['result']))}</div>
          </div>
        </div>
        <div class="m2-cc-answer">
          <span class="m2-cc-orb">⏺</span>
          <div>{escape(unsuccessful_response)}</div>
        </div>
      </div>
    </section>
    """


def show_recurring_failure():
    """Load and display the recurring failure scene for Section 1."""
    from IPython.display import HTML, display

    from course_lab.coding_trace_synth import BASE_SKILLS, load_fixture

    episodes = load_fixture()
    failure = next(
        episode for episode in episodes
        if episode["episode_id"] == "ep-run_test_suite-004"
    )
    standard_skill = next(
        skill for skill in BASE_SKILLS if skill["topic"] == "run_test_suite"
    )
    display(HTML(claude_code_failure_html(failure, standard_skill)))
    return TEST_QUERY, episodes


def _terminal_skill_card(entry: dict, *, additions=(), status_override=None) -> str:
    display_status = entry["status"] if status_override is None else status_override
    if display_status:
        status = escape(str(display_status).upper())
        status_class = (
            "m2-active" if display_status == "active" else "m2-superseded"
        )
        status_badge = (
            f'<span class="m2-badge {status_class}">STATUS {status}</span>\n        '
        )
    else:
        status_badge = ""
    added = set(additions)
    steps = []
    for index, step in enumerate(entry["steps"], 1):
        marker = "+ " if step in added else "  "
        css = "m2-step m2-add" if step in added else "m2-step"
        steps.append(
            f'<span class="{css}">{marker}{index:02d}  {escape(str(step))}</span>'
        )
    return f"""
    <div class="m2-terminal">
      <div class="m2-terminal-head">
        <span>{escape(str(entry['name']))}</span>
        <span>{status_badge}<span class="m2-badge">VERSION v{int(entry['version'])}</span></span>
      </div>
      <div>name: {escape(str(entry['name']))}</div>
      <div>topic: {escape(str(entry['topic']))}</div>
      <div style="color:#b8c5d6;margin:8px 0 12px">
        {escape(str(entry['description']))}</div>
      <div class="m2-prompt">$ procedure</div>
      {''.join(steps)}
    </div>
    """


def engine_contract_html(system_instruction: str) -> str:
    """Lead with the induction contract and keep the exact prompt available."""
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head">
        <div class="m2-shell-title">Induction Engine Contract</div>
        <div class="m2-shell-meta">one topic → one proposed skill</div>
      </div>
      <div class="m2-terminal" style="min-height:0">
        <div><span class="m2-prompt">INPUT</span> all episodes for one topic</div>
        <div><span class="m2-prompt">TASK</span> distil reusable procedure and lessons</div>
        <div><span class="m2-prompt">OUTPUT</span> steps, tools, errors, fixes, provenance</div>
        <div><span class="m2-prompt">GATE</span> pending until a human reviews it</div>
        <details style="margin-top:12px;border-top:1px solid #24324a;padding-top:10px">
          <summary style="cursor:pointer;color:#b8c5d6">Exact engine instruction</summary>
          <pre style="white-space:pre-wrap;color:#e5eef8;background:#0b1220;border:1px solid #24324a;border-radius:8px;padding:11px 12px;margin-bottom:0">{escape(system_instruction)}</pre>
        </details>
      </div>
    </section>
    """


def induced_skill_preview_html(proposal) -> str:
    """Render one proposal and emphasise the repeated failure it learned."""
    errors = [
        {"error": _field(item, "error"), "fix": _field(item, "fix")}
        for item in (_field(proposal, "errors_and_fixes", []) or [])
    ]
    from course_lab.coding_trace_synth import canonical_skill_name

    topic = _field(proposal, "topic")
    entry = {
        "name": canonical_skill_name(topic, _field(proposal, "name")),
        "topic": topic,
        "version": int(_field(proposal, "version", 2)),
        "status": _field(proposal, "status", "pending"),
        "description": _field(proposal, "description"),
        "steps": list(_field(proposal, "steps", []) or []),
    }
    failure = next(
        (item for item in errors if "ModuleNotFoundError" in item["error"]),
        errors[0] if errors else None,
    )
    learned = ""
    if failure:
        learned = f"""
        <div class="m2-error" style="margin-bottom:12px">
          <div style="font-size:11px;font-weight:800;letter-spacing:.08em">
            REPEATED FAILURE RECOGNISED</div>
          <strong>{escape(str(failure['error']))}</strong>
          <div class="m2-fix">FIX: {escape(str(failure['fix']))}</div>
        </div>
        """
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head"><div class="m2-shell-title">Proposed skill enhancement</div>
      <div class="m2-shell-meta">not active yet</div></div>
      {learned}
      {_terminal_skill_card(entry, additions=entry['steps'])}
    </section>
    """


def _skill_explorer_detail_html(entry: dict) -> str:
    """Render one skill without creating a nested terminal or scroll pane."""
    steps = "".join(
        f'<span class="m2-step">  {index:02d}  {escape(str(step))}</span>'
        for index, step in enumerate(entry["steps"], 1)
    )
    return f"""
      <div class="m2-terminal-head">
        <span>{escape(str(entry['name']))}</span>
        <span><span class="m2-badge m2-active">STATUS ACTIVE</span>
        <span class="m2-badge">VERSION v{int(entry['version'])}</span></span>
      </div>
      <div>name: {escape(str(entry['name']))}</div>
      <div>topic: {escape(str(entry['topic']))}</div>
      <div style="color:#b8c5d6;margin:8px 0 12px">
        {escape(str(entry['description']))}</div>
      <div class="m2-prompt">$ procedure</div>
      {steps}
    """


def skill_explorer_preview_html(mem, *, selected_name=None) -> str:
    """Render every skill and one detail pane without an internal scrollbar."""
    entries = _skill_box_versions(mem, active_only=True)
    if not entries:
        return '<div class="m2-shell">No active skills.</div>'
    selected = next(
        (entry for entry in entries if entry["name"] == selected_name),
        next(
            (entry for entry in entries if entry["name"] == "run-the-tests"),
            entries[0],
        ),
    )
    rows = "".join(
        '<div class="m2-explorer-row" role="option" '
        f'aria-selected="{str(entry["name"] == selected["name"]).lower()}">'
        f'<span class="m2-explorer-cursor">'
        f'{"❯" if entry["name"] == selected["name"] else ""}</span>'
        f'<span>{escape(str(entry["name"]))}</span></div>'
        for entry in entries
    )
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head"><div class="m2-shell-title">The Skill Box</div>
      <div class="m2-shell-meta">{len(entries)} active · v1 only</div></div>
      <div class="m2-explorer-grid">
        <nav class="m2-explorer-nav" role="listbox" aria-label="Active skills">
          <div class="m2-explorer-label">ACTIVE SKILLS</div>
          {rows}
          <div class="m2-explorer-keys">↑ ↓ navigate</div>
        </nav>
        <div class="m2-explorer-detail">
          {_skill_explorer_detail_html(selected)}
        </div>
      </div>
    </section>
    """


def _dlai_proxy_base(port: int) -> str | None:
    """Browser-reachable base URL for ``port`` in a DeepLearning.AI lab, or None.

    Hosted labs inject ``REV_PROXY_BASE_DOMAIN`` -- a template like
    ``https://s{ip}p{port}.lab-aws-staging.deeplearning.ai`` -- because the
    learner's browser runs on a different machine than the kernel, so a
    ``127.0.0.1`` iframe points at the learner's own laptop. Substituting the
    container's private IP + the app's port yields the edge URL that routes back
    to this container (authenticated by the same session as JupyterLab).
    """
    import os
    from urllib.parse import urlparse

    template = os.environ.get("REV_PROXY_BASE_DOMAIN")
    if not template or "{port}" not in template:
        return None

    # The private IP, dash-encoded for the subdomain label (172.16.145.7 ->
    # 172-16-145-7). JUPYTER_SERVER_URL's host is already ``ip-172-16-145-7...``.
    ip = ""
    host = urlparse(os.environ.get("JUPYTER_SERVER_URL", "")).hostname or ""
    if host.startswith("ip-"):
        ip = host.split(".")[0][3:]
    if not ip:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("1.1.1.1", 80))
            ip = sock.getsockname()[0].replace(".", "-")
        finally:
            sock.close()

    base = template.replace("{ip}", ip).replace("{port}", str(port))
    return base.rstrip("/") + "/"


def show_notebook_app(app, preview_html: str, *, height=500):
    """Launch a live inline app or render its deterministic course preview."""
    import os

    from IPython.display import HTML, display

    if os.environ.get("DLAI_RUN_REVIEW_UI", "1") != "1":
        display(HTML(preview_html))
        return None

    # Hosted lab: serve on all interfaces (the edge proxy reaches the app via
    # the container IP, not loopback) and embed the browser-reachable proxy URL
    # instead of gradio's default 127.0.0.1 iframe, which the remote browser
    # cannot load.
    if os.environ.get("REV_PROXY_BASE_DOMAIN"):
        app.launch(
            inline=False,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
            server_name="0.0.0.0",
            show_api=False,
        )
        url = _dlai_proxy_base(app.server_port)
        if url:
            display(HTML(
                f'<div><iframe src="{url}" width="100%" height="{height}" '
                'allow="clipboard-read; clipboard-write;" frameborder="0" '
                'allowfullscreen></iframe></div>'
            ))
            # Return nothing: the cell's last expression must not echo the
            # Blocks repr ("Gradio Blocks instance: N backend functions ...").
            return None

    # Local / default: gradio's own inline iframe over loopback works.
    app.launch(
        inline=True,
        share=False,
        prevent_thread_lock=True,
        quiet=True,
        server_name="127.0.0.1",
        show_api=False,
        height=height,
    )
    return None


def _skill_explorer_gradio_css() -> str:
    """Style a native radio group as a Claude Code-like skill navigator."""
    return """
    #m2-skill-title {margin-bottom:8px}
    #m2-skill-title .prose {max-width:none!important}
    #m2-skill-title .m2-shell-head {display:flex;justify-content:space-between;
      gap:12px;align-items:center;color:#172033}
    #m2-skill-title .m2-shell-title {font-size:18px;font-weight:750}
    #m2-skill-title .m2-shell-meta {font:600 12px ui-monospace,SFMono-Regular,
      Menlo,Consolas,monospace;color:#52606d}
    #m2-skill-explorer {gap:0!important;background:#0b1220;border:1px solid
      #24324a;border-radius:10px;color:#e5eef8;padding:0!important;
      min-height:420px}
    #m2-skill-nav-pane {border-right:1px solid #24324a;padding:14px 12px;
      min-width:260px!important}
    #m2-skill-nav-pane .prose {color:#77869a;font:800 10px/1.2
      ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;letter-spacing:.1em}
    #m2-skill-nav {border:0!important;background:transparent!important}
    #m2-skill-nav .wrap {gap:1px!important}
    #m2-skill-nav label {position:relative;display:flex!important;align-items:center;
      min-height:29px;padding:5px 7px!important;border:0!important;border-radius:5px;
      background:transparent!important;color:#aebdce!important;font:13px/1.25
      ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;box-shadow:none!important}
    #m2-skill-nav label::before {content:"";display:inline-block;width:18px;
      color:#d97757;font-weight:900}
    #m2-skill-nav label:has(input:checked) {background:#17243a!important;
      color:#fff!important}
    #m2-skill-nav label:has(input:checked)::before {content:"❯"}
    #m2-skill-nav input {position:absolute!important;opacity:0!important;
      width:1px!important;height:1px!important}
    #m2-skill-nav label:focus-within {outline:2px solid #7dd3fc;
      outline-offset:1px}
    #m2-skill-keys .prose {border-top:1px solid #24324a;margin-top:8px;
      padding-top:9px;color:#77869a;font:11px ui-monospace,SFMono-Regular,
      Menlo,Consolas,monospace}
    #m2-skill-detail-pane {padding:15px 17px;min-width:0!important}
    #m2-skill-detail {font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,
      Consolas,monospace;color:#e5eef8}
    #m2-skill-detail .m2-terminal-head {display:flex;justify-content:space-between;
      gap:10px;border-bottom:1px solid #24324a;padding-bottom:10px;
      margin-bottom:12px}
    #m2-skill-detail .m2-badge {border:1px solid #476582;border-radius:999px;
      padding:2px 8px;font-size:11px;font-weight:700}
    #m2-skill-detail .m2-active {color:#86efac;border-color:#2f855a}
    #m2-skill-detail .m2-prompt {color:#7dd3fc}
    #m2-skill-detail .m2-step {display:block;padding:2px 6px;border-radius:4px}
    #m2-skill-explorer *{color:#fff!important}
    #m2-skill-nav label::before{color:#d97757!important}
    @media(max-width:760px){#m2-skill-explorer{display:block!important}
      #m2-skill-nav-pane{border-right:0;border-bottom:1px solid #24324a}}
    """


def _skill_explorer_keyboard_js() -> str:
    """Keep arrow-key navigation inside the focused skill list."""
    return """
    () => {
      const attach = () => {
        const root = document.getElementById("m2-skill-nav");
        if (!root) { window.requestAnimationFrame(attach); return; }
        if (root.dataset.arrowNavigation === "ready") return;
        root.dataset.arrowNavigation = "ready";
        root.addEventListener("keydown", (event) => {
          if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
          const choices = Array.from(root.querySelectorAll('input[type="radio"]'));
          const current = choices.findIndex((choice) => choice.checked);
          if (current < 0 || choices.length === 0) return;
          event.preventDefault();
          event.stopPropagation();
          const delta = event.key === "ArrowDown" ? 1 : -1;
          const next = (current + delta + choices.length) % choices.length;
          choices[next].focus();
          choices[next].click();
        });
      };
      attach();
    }
    """


def build_skill_explorer_app(mem):
    """Build the keyboard-navigable Section 3 skill-box explorer."""
    import gradio as gr

    entries = _skill_box_versions(mem, active_only=True)
    names = [entry["name"] for entry in entries]
    by_name = {entry["name"]: entry for entry in entries}
    default = "run-the-tests" if "run-the-tests" in names else names[0]
    with gr.Blocks(
        css=_skill_explorer_gradio_css(), js=_skill_explorer_keyboard_js()
    ) as app:
        gr.HTML(
            f'<div class="m2-shell-head"><div class="m2-shell-title">'
            f'The Skill Box</div><div class="m2-shell-meta">'
            f'{len(entries)} active · v1 only</div></div>',
            elem_id="m2-skill-title",
        )
        with gr.Row(elem_id="m2-skill-explorer"):
            with gr.Column(scale=2, min_width=260, elem_id="m2-skill-nav-pane"):
                gr.Markdown("ACTIVE SKILLS", elem_id="m2-skill-nav-label")
                selected = gr.Radio(
                    choices=names,
                    value=default,
                    show_label=False,
                    container=False,
                    interactive=True,
                    elem_id="m2-skill-nav",
                )
                gr.Markdown("↑ ↓ navigate", elem_id="m2-skill-keys")
            with gr.Column(scale=3, min_width=360, elem_id="m2-skill-detail-pane"):
                panel = gr.HTML(
                    _skill_explorer_detail_html(by_name[default]),
                    elem_id="m2-skill-detail",
                )
        selected.change(
            lambda name: _skill_explorer_detail_html(by_name[name]),
            inputs=selected,
            outputs=panel,
        )
    return app


def version_compare_preview_html(
    mem, *, topic, left_version=1, right_version=None
) -> str:
    """Render two versions of one skill in equal terminal panes."""
    versions = _skill_box_versions(mem, topic=topic)
    if len(versions) < 2:
        return _terminal_styles() + (
            '<section class="m2-shell">Two approved versions are needed '
            "for comparison.</section>"
        )
    right_version = right_version or max(item["version"] for item in versions)
    left = next(item for item in versions if item["version"] == left_version)
    right = next(item for item in versions if item["version"] == right_version)
    additions = [step for step in right["steps"] if step not in left["steps"]]
    failures = "".join(
        f"<div><strong>{escape(str(item['error']))}</strong>"
        f"<div class=\"m2-fix\">FIX: {escape(str(item['fix']))}</div></div>"
        for item in right["errors_and_fixes"]
    )
    learned = (
        '<div class="m2-error"><div style="font-size:11px;font-weight:800;'
        'letter-spacing:.08em">REPEATED FAILURE LEARNED</div>'
        f"{failures}</div>" if failures else ""
    )
    return _terminal_styles() + f"""
    <section class="m2-shell m2-review-compare">
      <div class="m2-shell-head"><div class="m2-shell-title">Compare Skill Versions</div>
      <div class="m2-shell-meta">$ skillbox diff {escape(topic)}</div></div>
      <div class="m2-compare-grid">
        <div><div class="m2-pane-label">BEFORE</div>{_terminal_skill_card(left, status_override="active")}</div>
        <div><div class="m2-pane-label">AFTER</div>{_terminal_skill_card(right, additions=additions, status_override="")}</div>
      </div>
      {learned}
    </section>
    """


def retrieval_improvement_html(query: str, before, after) -> str:
    """Mirror the opening conversation and foreground the learned v2 lesson."""
    before_name = escape(str(before.name))
    after_name = escape(str(after.name))
    question = escape(query)
    approved_v2 = (
        str(after.name) == "run-the-tests"
        and str(after.topic) == "run_test_suite"
        and int(after.version) >= 2
    )
    if not approved_v2:
        return _terminal_styles() + f"""
        <section class="m2-shell">
          <div class="m2-shell-head">
            <div class="m2-shell-title">Same conversation, no approved update</div>
            <div class="m2-shell-meta">retrieval after review</div>
          </div>
          <div class="m2-terminal" style="min-height:0">
            <div class="m2-pane-label">SAME QUERY</div>
            <div style="font-size:15px;margin-bottom:12px">{question}</div>
            <div>No approved v2 skill was retrieved. The active result remains
              {after_name} · v{int(after.version)}, so no improved outcome is claimed.</div>
          </div>
        </section>
        """
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head">
        <div class="m2-shell-title">Same conversation, different outcome</div>
        <div class="m2-shell-meta">retrieval after approval</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr;gap:12px">
        <div style="display:flex;flex-direction:column">
          <div class="m2-pane-label">BEFORE · SAME QUERY</div>
          <div class="m2-cc-terminal" style="min-height:0;flex:1">
            <div class="m2-cc-turn"><span class="m2-cc-prompt">❯</span> {question}</div>
            <div class="m2-cc-turn"><span class="m2-cc-orb">⏺</span> I’ll check the project’s test suite.</div>
            <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Skill</span>({before_name})</div>
            <div class="m2-cc-result"><span class="m2-cc-hook">⎿</span>
              <div>Loaded skill: {before_name} · v{int(before.version)}</div></div>
            <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Bash</span>(pytest -q)</div>
            <div class="m2-cc-result"><span class="m2-cc-hook">⎿</span>
              <div class="m2-cc-error">Error: Exit code 1<br>ModuleNotFoundError: No module named 'app'</div></div>
            <div class="m2-cc-answer"><span class="m2-cc-orb">⏺</span>
              <div>I couldn’t run the tests because Python cannot import the project package.</div></div>
          </div>
        </div>
        <div style="display:flex;flex-direction:column">
          <div class="m2-pane-label" style="color:#15803d">AFTER · SAME QUERY</div>
          <div class="m2-cc-terminal" style="min-height:0;flex:1;border-color:#2f855a">
            <div class="m2-cc-turn"><span class="m2-cc-prompt">❯</span> {question}</div>
            <div class="m2-cc-turn"><span class="m2-cc-orb">⏺</span> I’ll check the project’s test suite.</div>
            <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Skill</span>({after_name})</div>
            <div class="m2-cc-result"><span class="m2-cc-hook">⎿</span><div>
              <div>Loaded skill: {after_name} · v{int(after.version)}</div>
              <div class="m2-v2-skill">
                <div class="m2-v2-title">RUN-THE-TESTS · VERSION {int(after.version)}</div>
                <div class="m2-v2-failure">
                  <div class="m2-v2-label">FAILURE LEARNED</div>
                  <div>Never invoke bare pytest on this repository.</div>
                  <div>bare pytest → ModuleNotFoundError</div>
                </div>
                <div class="m2-v2-procedure">
                  <div class="m2-v2-label">IMPROVED PROCEDURE</div>
                  <div>01 Read the Makefile for the canonical command</div>
                  <div>02 Run make test in the project environment</div>
                  <div>03 Report the numeric result</div>
                </div>
              </div>
            </div></div>
            <div><span class="m2-cc-orb">⏺</span> <span class="m2-cc-tool">Bash</span>(make test)</div>
            <div class="m2-cc-result"><span class="m2-cc-hook">⎿</span>
              <div class="m2-fix">212 passed</div></div>
            <div class="m2-cc-answer"><span class="m2-cc-orb">⏺</span>
              <div>All 212 tests pass.</div></div>
          </div>
        </div>
      </div>
    </section>
    """


def evaluation_skipped_html(after_hit) -> str:
    """Explain why a rejected or unavailable skill cannot receive lift credit."""
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head">
        <div class="m2-shell-title">Evaluation not run</div>
        <div class="m2-shell-meta">governance gate respected</div>
      </div>
      <div style="background:#f1f5f9;border-left:3px solid #64748b;
        border-radius:7px;padding:12px;color:#334155">
        No approved v2 skill entered context. Retrieval returned
        {escape(str(after_hit.name))} · v{int(after_hit.version)}, so this run
        does not claim an enhanced-skill lift.
      </div>
    </section>
    """


def evaluation_bar_chart_html(result: dict) -> str:
    """Render the final standard-versus-enhanced evaluation as grouped bars."""
    labels = [
        ("avoids_trap", "Avoids trap"),
        ("concrete_steps", "Concrete steps"),
        ("uses_learned_fix", "Uses learned fix"),
        ("no_invented", "No invented claims"),
        ("overall", "Overall"),
    ]
    rows = []
    for key, label in labels:
        standard = float(result["baseline_judge"][key])
        enhanced = float(result["skilled_judge"][key])
        rows.append(f"""
        <div style="display:grid;grid-template-columns:150px minmax(0,1fr);
          gap:10px;align-items:center;margin:11px 0">
          <div style="font-weight:700;color:#334155">{label}</div>
          <div>
            <div style="display:grid;grid-template-columns:82px minmax(0,1fr) 38px;
              gap:8px;align-items:center;margin-bottom:5px">
              <span style="font-size:11px;color:#64748b">Standard skill</span>
              <div style="height:12px;background:#e2e8f0;border-radius:999px">
                <div style="width:{round(standard * 100)}%;height:100%;
                  background:#64748b;border-radius:999px"></div>
              </div><span style="font:11px ui-monospace,monospace">{standard:.2f}</span>
            </div>
            <div style="display:grid;grid-template-columns:82px minmax(0,1fr) 38px;
              gap:8px;align-items:center">
              <span style="font-size:11px;color:#15803d">Enhanced skill</span>
              <div style="height:12px;background:#dcfce7;border-radius:999px">
                <div style="width:{round(enhanced * 100)}%;height:100%;
                  background:#15803d;border-radius:999px"></div>
              </div><span style="font:11px ui-monospace,monospace">{enhanced:.2f}</span>
            </div>
          </div>
        </div>
        """)
    standard_overall = float(result["baseline_judge"]["overall"])
    enhanced_overall = float(result["skilled_judge"]["overall"])
    lift = float(result["lift"])
    return _terminal_styles() + f"""
    <section class="m2-shell">
      <div class="m2-shell-head">
        <div class="m2-shell-title">Evaluation: standard skill vs enhanced skill</div>
        <div class="m2-shell-meta">scores from 0 to 1</div>
      </div>
      <div style="background:#eef2f7;border:1px solid #d9e2ec;border-radius:9px;
        padding:12px;margin-bottom:14px">
        <div style="font-weight:850;color:#172033;margin-bottom:9px">
          Worked example, not a 100-task benchmark</div>
        <div class="m2-eval-methods">
          <div class="m2-skill-chip">1 repository task</div>
          <div class="m2-skill-chip">2 answers, same model</div>
          <div class="m2-skill-chip">1 LLM judge</div>
          <div class="m2-skill-chip">5 rubric scores</div>
        </div>
        <div style="font-size:12px;line-height:1.55;color:#334155">
          <strong>Task:</strong> {escape(str(result['task']))}<br>
          <strong>Reference remedy:</strong> {escape(str(result.get('reference_summary', result['reference'])))}
        </div>
      </div>
      {''.join(rows)}
      <div class="m2-eval-reasons">
        <div style="background:#f1f5f9;border-left:3px solid #64748b;
          border-radius:6px;padding:10px;color:#334155;font-size:12px">
          <strong>Why standard scored {standard_overall:.2f}</strong><br>
          {escape(str(result['baseline_judge']['reasoning']))}
        </div>
        <div style="background:#dcfce7;border-left:3px solid #15803d;
          border-radius:6px;padding:10px;color:#14532d;font-size:12px">
          <strong>Why enhanced scored {enhanced_overall:.2f}</strong><br>
          {escape(str(result['skilled_judge']['reasoning']))}
        </div>
      </div>
      <div style="margin-top:12px;padding:12px;border-radius:9px;
        background:#dcfce7;color:#166534;font-size:18px;font-weight:800;
        text-align:center">{enhanced_overall:.2f} − {standard_overall:.2f} = {lift:+.2f} overall lift</div>
      <div style="margin-top:8px;color:#475569;font-size:12px;text-align:center">
        212 passed is the repository test-suite result, not the evaluation sample size.
      </div>
    </section>
    """


def build_version_compare_app(mem, *, topic):
    """Build the Section 6 Gradio side-by-side skill comparator."""
    import gradio as gr

    versions = _skill_box_versions(mem, topic=topic)
    choices = sorted({int(item["version"]) for item in versions})
    left_default, right_default = min(choices), max(choices)

    def render(left, right):
        return version_compare_preview_html(
            mem, topic=topic, left_version=int(left), right_version=int(right)
        )

    with gr.Blocks() as app:
        with gr.Row():
            left = gr.Dropdown(
                choices=choices, value=left_default, label="Before version"
            )
            right = gr.Dropdown(
                choices=choices, value=right_default, label="After version"
            )
        panel = gr.HTML(render(left_default, right_default))
        left.change(render, inputs=[left, right], outputs=panel)
        right.change(render, inputs=[left, right], outputs=panel)
    return app


def run_dream_review(mem, episodes, *, day, complete, oracle_db_module):
    """Run a dream pass and return its checkpoint, state, and pending rows."""
    from course_lab.coding_dream_graph import make_oracle_checkpointer, run_dream_graph
    from course_lab.skill_governance import review_proposals

    ckpt = make_oracle_checkpointer(
        oracle_db_module.get_connection(autocommit=True)
    )
    state = run_dream_graph(
        mem, episodes, day=day, complete=complete, checkpointer=ckpt
    )
    pending = review_proposals(mem, store="enhanced")
    return ckpt, state, pending


def _persisted_proposal_status(mem, proposal) -> str:
    """Read governance status from storage, not a potentially stale UI row."""
    skill_id = str(_field(proposal, "skill_id"))
    for row in mem.list_enhanced_skills():
        if str(_field(row, "skill_id")) == skill_id:
            return str(_field(row, "status", "pending"))
    raise ValueError(f"unknown enhanced skill id: {skill_id}")


def apply_decision(mem, proposal, *, decision, comment="") -> str:
    """Apply one review decision immediately and return its visible status.

    Approval promotes the proposal into the single skill box. Both decisions
    need a written reason, which the caller keeps in review state. OracleVS
    re-indexing remains a separate visible notebook
    step so learners can see approval change retrieval.
    """
    from course_lab.skill_governance import promote_to_skill_box, reject_skills

    current_status = _persisted_proposal_status(mem, proposal)
    if current_status != "pending":
        raise ValueError(
            f"proposal already reviewed with status {current_status!r}"
        )
    if decision in {"approve", "reject"} and not str(comment).strip():
        raise ValueError("a review reason is required")
    if decision == "approve":
        skill_id = str(_field(proposal, "skill_id"))
        mem.set_enhanced_skill_review_comment(skill_id, str(comment).strip())
        promote_to_skill_box(mem, [skill_id])
        version = int(_field(proposal, "version", 1))
        return (
            f"✓ Approved: '{_field(proposal, 'name')}' is now active in "
            f"the skill box (v{version})"
        )
    if decision == "reject":
        skill_id = str(_field(proposal, "skill_id"))
        mem.set_enhanced_skill_review_comment(skill_id, str(comment).strip())
        reject_skills(mem, [skill_id], store="enhanced")
        return "✕ Rejected: reason recorded for the next induction engine pass"
    raise ValueError("decision must be approve or reject")


def review_preview_html(mem, pending, state) -> str:
    """Render one v1-to-v2 diff with its explicit review decision."""
    if len(pending) != 1:
        raise ValueError("the focused review expects exactly one skill")
    proposal = pending[0]
    skill_id = str(_field(proposal, "skill_id"))
    review = state.get(skill_id, {})
    decision = review.get("decision", "")
    status = escape(str(review.get("status", "")))
    reason = escape(str(review.get("comment", ""))) or "Add a review reason."
    status_colour = {
        "approve": "#15803d",
        "reject": "#b91c1c",
    }.get(decision, "#6b7280")
    diff = version_compare_preview_html(mem, topic=_field(proposal, "topic"))
    return f"""
    <section class="m2-shell" style="margin-bottom:10px">
      <div class="m2-shell-head">
        <div class="m2-shell-title">Review the skill change</div>
        <div class="m2-shell-meta">approved by default — reject to override</div>
      </div>
      <div style="font-size:12px;font-weight:750;color:#334155;margin-bottom:6px">
        Review reason
      </div>
      <div style="background:#fff;border:1px solid #cbd5e1;border-radius:8px;
        padding:12px;min-height:76px;color:#334155">{reason}</div>
      <div style="display:flex;gap:8px;align-items:center;margin-top:10px">
        <button disabled style="border:0;border-radius:7px;padding:6px 12px;
          background:#15803d;color:white;font-weight:750">Approve</button>
        <button disabled style="border:0;border-radius:7px;padding:6px 12px;
          background:#b91c1c;color:white;font-weight:750">Reject</button>
        <span style="color:{status_colour};font-size:13px;font-weight:700">
          {status}</span>
      </div>
    </section>
    """ + diff


def build_review_app(pending, *, mem, defaults=None):
    """Build one focused diff review that approves by default.

    Running the cell applies a default approval immediately (so Run All
    produces an active v2 for the retrieval and evaluation sections), while
    the live buttons remain a one-shot override: Reject reverses the
    promotion and restores the prior recipe, Approve replaces the default
    reason with the learner's own.
    """
    import gradio as gr

    if len(pending) != 1:
        raise ValueError("the focused review expects exactly one skill")
    proposal = pending[0]
    skill_id = str(_field(proposal, "skill_id"))
    auto_default = defaults is None
    if auto_default:
        defaults = {
            skill_id: {
                "decision": "approve",
                "comment": APPROVAL_FEEDBACK,
            }
        }
    review_state = initialise_review_state(pending, defaults=defaults)

    review = review_state[skill_id]
    applied_default = False
    if (review["decision"] in {"approve", "reject"}
            and _persisted_proposal_status(mem, proposal) == "pending"):
        review["status"] = apply_decision(
            mem,
            proposal,
            decision=review["decision"],
            comment=review["comment"],
        )
        if auto_default:
            applied_default = True
            review["status"] += (
                " — approved by default; add a reason and click Reject "
                "to override."
            )
    # Only a default applied by this build may be overridden, exactly once.
    overridable = {"value": applied_default}

    def on_review(state, reason, decision):
        if not str(reason).strip():
            current_diff = version_compare_preview_html(mem, topic=proposal.topic)
            message = "Please add a review reason before approving or rejecting."
            return state, current_diff, message
        current_status = _persisted_proposal_status(mem, proposal)
        if current_status == "pending":
            updated = record_review(state, skill_id, decision, reason)
            status_text = apply_decision(
                mem, proposal, decision=decision, comment=reason
            )
        elif overridable["value"]:
            from course_lab.skill_governance import demote_from_skill_box

            overridable["value"] = False
            updated = record_review(state, skill_id, decision, reason)
            if decision == "reject":
                demote_from_skill_box(mem, skill_id)
                mem.set_enhanced_skill_review_comment(
                    skill_id, str(reason).strip()
                )
                status_text = (
                    "✕ Rejected: default approval reversed and the prior "
                    "recipe restored. Re-run the sections below to see the "
                    "effect."
                )
            else:
                mem.set_enhanced_skill_review_comment(
                    skill_id, str(reason).strip()
                )
                status_text = (
                    "✓ Approval confirmed with your reason recorded."
                )
        else:
            current_diff = version_compare_preview_html(mem, topic=proposal.topic)
            message = f"Decision already recorded: {current_status}."
            return state, current_diff, message
        updated[skill_id]["status"] = status_text
        updated_diff = version_compare_preview_html(mem, topic=proposal.topic)
        return updated, updated_diff, status_text

    css = """
    #m2-review-reason textarea {min-height:112px!important}
    #m2-review-actions {gap:8px!important;align-items:center}
    #m2-review-actions button {flex:0 0 auto!important;min-width:92px!important;
      max-width:110px!important;padding:6px 12px!important}
    #m2-approve {background:#15803d!important;color:#fff!important;
      border-color:#15803d!important}
    #m2-reject {background:#b91c1c!important;color:#fff!important;
      border-color:#b91c1c!important}
    """
    with gr.Blocks(css=css) as review_app:
        state_component = gr.State(review_state)
        reason = gr.Textbox(
            label="Review reason",
            lines=4,
            placeholder="Explain why this change should be approved or rejected.",
            elem_id="m2-review-reason",
        )
        with gr.Row(elem_id="m2-review-actions"):
            approve = gr.Button("Approve", elem_id="m2-approve")
            reject = gr.Button("Reject", elem_id="m2-reject")
        status = gr.Markdown(value=review_state[skill_id].get("status", ""))
        diff_panel = gr.HTML(
            version_compare_preview_html(mem, topic=proposal.topic)
        )
        approve.click(
            lambda state, note: on_review(state, note, "approve"),
            inputs=[state_component, reason],
            outputs=[state_component, diff_panel, status],
        )
        reject.click(
            lambda state, note: on_review(state, note, "reject"),
            inputs=[state_component, reason],
            outputs=[state_component, diff_panel, status],
        )
    return review_app, review_state


def apply_review_and_index(
    mem, pending, review_state, *, episodes, out_dir, base_skill_md, connection
):
    """Apply explicit review, reflect on rejection, and index approved skills."""
    from IPython.display import HTML

    from course_lab.coding_lift_eval import render_answer_diff_html
    from course_lab.induction_engine import cached_complete, induce_skill, load_engine_outputs
    from course_lab.skill_render import render_skill_md
    from course_lab.skill_vectorstore import index_skills_oraclevs

    approved, rejected = partition_review(pending, review_state)
    for skill in approved:
        review = review_state[str(skill.skill_id)]
        apply_decision(
            mem, skill, decision="approve", comment=review["comment"],
        )
    for skill, feedback in rejected:
        apply_decision(
            mem, skill, decision="reject", comment=feedback,
        )

    reflected = cached_complete(
        load_engine_outputs(Path(out_dir) / "reflected.json"),
        feedback_by_topic={
            "add_api_endpoint": _REFLECTED_REJECTION_FEEDBACK,
        },
    )
    revised_pending = []
    for rejected_skill, feedback in rejected:
        rejected_eps = [
            episode
            for episode in episodes
            if episode["topic"] == rejected_skill.topic
            and episode["day"] == 1
        ]
        revised = induce_skill(
            rejected_eps,
            feedback=feedback,
            complete=reflected,
        )
        revised_pending.append(
            write_enhanced_pending(mem, revised, created_day=1)
        )

    learned = next(skill for skill in pending if skill.topic == "run_test_suite")
    diff = HTML(render_answer_diff_html(base_skill_md, render_skill_md(learned)))
    enhanced_store = index_skills_oraclevs(
        connection=connection,
        skills=mem.list_enhanced_skills(status="active"),
        table_name="DLAI_ENH_SKILL_VS",
        store="enhanced",
    )
    return {
        "approved": approved,
        "rejected": rejected,
        "revised_pending": revised_pending,
        "learned": learned,
        "diff": diff,
        "enhanced_store": enhanced_store,
    }


def governance_status(mem, oracle_db_module) -> dict:
    """Return the storage/gating facts for the optional technical appendix."""
    with oracle_db_module.get_connection(autocommit=True).cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM SKILLBOX")
        standard_rows = cur.fetchone()[0]
        cur.execute("SELECT promoted FROM SKILLBOX WHERE name = 'run-the-tests'")
        promoted_row = cur.fetchone()
        cur.execute("SELECT status, COUNT(*) FROM ENHANCED_SKILLBOX GROUP BY status")
        statuses = dict(cur.fetchall())
        cur.execute(
            "SELECT topic, version, status FROM ENHANCED_SKILLBOX "
            "WHERE topic = 'run_test_suite' ORDER BY version"
        )
        versions = list(cur.fetchall())
    return {
        "standard_rows": standard_rows,
        "promoted": bool(promoted_row[0]) if promoted_row else None,
        "statuses": statuses,
        "versions": versions,
    }


__all__ = [
    "apply_decision",
    "apply_review_and_index",
    "build_review_app",
    "build_skill_explorer_app",
    "build_version_compare_app",
    "connect_stack",
    "engine_contract_html",
    "evaluation_bar_chart_html",
    "evaluation_skipped_html",
    "governance_status",
    "induced_skill_preview_html",
    "initialise_review_state",
    "night_outputs",
    "partition_review",
    "proposal_decision_support",
    "record_review",
    "retrieval_comparison_rows",
    "retrieval_improvement_html",
    "review_preview_html",
    "review_rows",
    "show_notebook_app",
    "show_recurring_failure",
    "skill_explorer_preview_html",
    "version_compare_preview_html",
    "promoted_skill_diff",
    "run_dream",
    "run_dream_review",
    "standard_skill_snapshot",
    "stack_connection_report",
    "stack_ready_text",
    "print_trace_seed_summary",
    "trace_seed_summary",
    "write_enhanced_pending",
]
