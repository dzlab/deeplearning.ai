"""course_lab/coding_dream_graph.py: the dream loop as a checkpointed LangGraph.

Module 2's dream loop runs as a two-node LangGraph ``StateGraph`` (induce ->
persist) whose state is checkpointed to Oracle via
``langgraph_oracledb`` ``OracleSaver``, keyed by a thread_id. Re-running with the
same thread_id resumes from the last completed node, which is the point of a
sleep-time/background pass: it is restartable and auditable.

The node bodies reuse the plain-Python core in :func:`course_lab.dream.dream`'s
logic (induce all topics for the day, write each proposal to ENHANCED_SKILLBOX
as pending, version-bumped), so there is one source of truth for what the dream
actually does.
"""
from __future__ import annotations

from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph_oracledb.checkpoint.oracle.sync import OracleSaver
from pydantic import BaseModel, Field

from course_lab.induction_engine import DEFAULT_MODEL, induce_all


class DreamState(BaseModel):
    # Input.
    day: int = 1
    # Telemetry the notebook can print to show what crossed each node boundary.
    induced_topics: list[str] = Field(default_factory=list)
    persisted: list[dict] = Field(default_factory=list)  # {topic, version}
    completed_nodes: list[str] = Field(default_factory=list)


def make_induce_node(*, mem, episodes, complete, model_id):
    def induce_node(state: DreamState) -> dict[str, Any]:
        history = [e for e in episodes if e["day"] <= state.day]
        kwargs = {"complete": complete} if complete is not None else {}
        latest_by_topic = {}
        for row in mem.list_enhanced_skills():
            current = latest_by_topic.get(row.topic)
            if current is None or row.version > current.version:
                latest_by_topic[row.topic] = row
        feedback_by_topic = {
            topic: row.review_comment
            for topic, row in latest_by_topic.items()
            if row.status == "rejected" and row.review_comment
        }
        proposals = induce_all(
            history,
            model_id=model_id,
            feedback_by_topic=feedback_by_topic,
            **kwargs,
        )
        # Stash proposals on the instance so persist_node can use them without
        # serialising pydantic-heavy objects through the checkpoint.
        induce_node._proposals = proposals
        return {"induced_topics": [p.topic for p in proposals],
                "completed_nodes": state.completed_nodes + ["induce"]}
    return induce_node


def make_persist_node(*, mem, induce_node, minimum_version: int = 1):
    def persist_node(state: DreamState) -> dict[str, Any]:
        from course_lab.coding_trace_synth import canonical_skill_name

        proposals = getattr(induce_node, "_proposals", [])
        existing = list(mem.list_enhanced_skills())
        persisted = []
        for prop in proposals:
            prev = [r.version for r in existing if r.topic == prop.topic]
            version = max(max(prev, default=0) + 1, minimum_version)
            mem.write_enhanced_skill(
                topic=prop.topic,
                name=canonical_skill_name(prop.topic, prop.name),
                description=prop.description,
                when_to_use=prop.when_to_use, steps=prop.steps,
                skills_used=prop.skills_used, likely_tools=prop.likely_tools,
                errors_and_fixes=[ef.model_dump() for ef in prop.errors_and_fixes],
                provenance=prop.provenance, status="pending",
                version=version, created_day=state.day,
            )
            persisted.append({"topic": prop.topic, "version": version})
        return {"persisted": persisted,
                "completed_nodes": state.completed_nodes + ["persist"]}
    return persist_node


def build_coding_dream_graph(*, mem, episodes, complete: Callable | None = None,
                             model_id: str = DEFAULT_MODEL, checkpointer: Any = None,
                             minimum_version: int = 1):
    """Wire and compile the two-node dream StateGraph (induce -> persist)."""
    induce_node = make_induce_node(
        mem=mem, episodes=episodes, complete=complete, model_id=model_id)
    graph = StateGraph(DreamState)
    graph.add_node("induce", induce_node)
    graph.add_node(
        "persist",
        make_persist_node(
            mem=mem,
            induce_node=induce_node,
            minimum_version=minimum_version,
        ),
    )
    graph.add_edge(START, "induce")
    graph.add_edge("induce", "persist")
    graph.add_edge("persist", END)
    return graph.compile(checkpointer=checkpointer)


def make_oracle_checkpointer(connection: Any) -> OracleSaver:
    """Construct an Oracle-backed checkpointer (its own connection)."""
    saver = OracleSaver(conn=connection)
    saver.setup()
    return saver


def run_dream_graph(mem, episodes, *, day: int, complete: Callable | None = None,
                    model_id: str = DEFAULT_MODEL, checkpointer: Any = None,
                    thread_id: str = "m2-dream",
                    minimum_version: int = 1) -> dict:
    """Run the dream graph for ``day`` and return the final state dict.

    With a checkpointer and a stable thread_id, a re-run resumes from the last
    completed node rather than repeating the induction.
    """
    app = build_coding_dream_graph(
        mem=mem,
        episodes=episodes,
        complete=complete,
        model_id=model_id,
        checkpointer=checkpointer,
        minimum_version=minimum_version,
    )
    config = {"configurable": {"thread_id": f"{thread_id}-day{day}"}}
    return app.invoke(DreamState(day=day), config=config)


__all__ = [
    "DreamState", "build_coding_dream_graph", "make_oracle_checkpointer",
    "run_dream_graph",
]
