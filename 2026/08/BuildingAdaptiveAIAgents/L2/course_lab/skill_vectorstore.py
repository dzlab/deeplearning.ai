"""course_lab/skill_vectorstore.py: the skill boxes as langchain-oracledb OracleVS.

Module 2 retrieves skills through ``langchain_oracledb.OracleVS`` so the stack
claim is real and a learner sees the idiomatic Oracle + LangChain path, not
hand-rolled SQL. The two skill boxes are two OracleVS tables in one 26ai
database; we search both and merge by cosine distance.

Gating is by construction: the enhanced store is built from *approved* skills
only, so a pending skill is simply not in the index until it is re-indexed
after a human approves it.
"""
from __future__ import annotations

from typing import Any, NamedTuple

from langchain_core.documents import Document
from langchain_oracledb.vectorstores import OracleVS
from langchain_oracledb.vectorstores.oraclevs import DistanceStrategy

from course_lab.python_embedder import make_embedder
from course_lab.vsql_retriever import DEFAULT_EMBEDDER_SPEC, EmbedderLcEmbeddings


class SkillVSHit(NamedTuple):
    """One OracleVS retrieval result from the learner-visible skill box."""
    store: str
    name: str
    topic: str
    distance: float       # cosine distance (0 = identical direction)
    version: int = 1
    skill_md: str = ""
    errors_and_fixes: tuple[dict, ...] = ()


def _skill_topic(skill: Any) -> str:
    topic = getattr(skill, "topic", "") or ""
    if topic:
        return topic
    from course_lab.coding_trace_synth import SKILL_TOPICS

    return SKILL_TOPICS.get(skill.name, "")


def _skill_markdown(skill: Any) -> str:
    from course_lab.skill_render import render_skill_md, render_standard_skill_md

    if hasattr(skill, "steps"):
        return render_skill_md(skill)
    return render_standard_skill_md({
        "name": skill.name,
        "topic": _skill_topic(skill),
        "description": skill.description,
        "recipe_steps": list(getattr(skill, "recipe_steps", []) or []),
    })


def _skill_text(s: Any) -> str:
    """Build searchable text while preserving a refined skill's original intent.

    Enhanced skills keep the standard recipe's intent text as well as their
    learned errors and fixes. That lets the same ordinary user request retrieve
    v2 after approval, while failure-shaped requests can match its new lessons.
    """
    parts = []
    topic = getattr(s, "topic", "") or ""
    if topic and hasattr(s, "steps"):
        from course_lab.coding_trace_synth import BASE_SKILLS

        standard = next(
            (item for item in BASE_SKILLS if item["topic"] == topic), None
        )
        if standard:
            stable_intent = f"{standard['name']}: {standard['description']}"
            # Weight the stable retrieval intent against the longer learned
            # procedure, while still indexing the new errors and fixes below.
            parts.extend([stable_intent] * 5)
    parts.append(f"{s.name}: {s.description}")
    when = getattr(s, "when_to_use", "") or ""
    if when:
        parts.append(f"when to use: {when}")
    for ef in getattr(s, "errors_and_fixes", []) or []:
        parts.append(f"error: {ef['error']} fix: {ef['fix']}")
    steps = (
        getattr(s, "steps", None)
        or getattr(s, "recipe_steps", [])
        or []
    )
    for step in steps:
        parts.append(str(step))
    return " | ".join(parts)


def retrievable_base_skills(skills: list[Any]) -> list[Any]:
    """Drop standard skills marked ``promoted`` from the retrievable set.

    A standard skill is promoted once an approved enhanced skill supersedes it
    for the same topic. Excluding promoted skills here (at index time) means the
    base OracleVS box never holds them, so the agent never retrieves two recipes
    for one job, with no per-query post-filtering.
    """
    return [s for s in skills if not getattr(s, "promoted", False)]


def _drop_if_exists(connection: Any, table_name: str) -> None:
    with connection.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM user_tables WHERE table_name = :t",
                    t=table_name.upper())
        if cur.fetchone()[0] > 0:
            cur.execute(f'DROP TABLE "{table_name.upper()}" PURGE')


def index_skills_oraclevs(
    *,
    connection: Any,
    skills: list[Any],
    table_name: str,
    store: str,
    model_spec: str = DEFAULT_EMBEDDER_SPEC,
) -> OracleVS:
    """Index skill rows into an OracleVS table and return the store.

    ``skills`` are SkillRow / EnhancedSkillRow objects. The searchable text is
    the name plus description (what the agent matches a task against). The table
    is dropped and rebuilt so the index always reflects the current set (this is
    how an approval becomes visible: re-index the enhanced store afterwards).
    """
    if connection is None:
        raise RuntimeError("index_skills_oraclevs requires a live oracledb connection")
    # Dedup: a promoted standard skill has an approved enhanced successor, so it
    # never enters the base index. The enhanced store is unaffected.
    if store == "base":
        skills = retrievable_base_skills(skills)
    _drop_if_exists(connection, table_name)

    embeddings = EmbedderLcEmbeddings(make_embedder(model_spec, connection=connection))
    documents = [
        Document(
            page_content=_skill_text(s),
            metadata={
                "skill_id": s.skill_id,
                "name": s.name,
                "topic": _skill_topic(s),
                "store": store,
                "version": int(getattr(s, "version", 1)),
                "skill_md": _skill_markdown(s),
                "errors_and_fixes": list(
                    getattr(s, "errors_and_fixes", []) or []
                ),
            },
        )
        for s in skills
    ]
    if not documents:
        # OracleVS.from_documents needs at least one document; create an empty
        # table by indexing a sentinel, then deleting it, so callers still get a
        # usable (empty) store. Simpler: return None-free by indexing nothing is
        # not supported, so raise a clear error instead.
        raise ValueError(
            f"index_skills_oraclevs got no skills for the {store!r} store; "
            "index after at least one skill exists.")

    return OracleVS.from_documents(
        documents=documents,
        embedding=embeddings,
        client=connection,
        table_name=table_name,
        distance_strategy=DistanceStrategy.COSINE,
        ids=[d.metadata["skill_id"] for d in documents],
    )


def index_active_skills(
    *, connection: Any, mem: Any, table_name: str = "DLAI_SKILL_VS"
) -> OracleVS:
    """Rebuild one OracleVS index over the single active skill-box view."""
    from course_lab.skill_governance import list_active_skills

    return index_skills_oraclevs(
        connection=connection,
        skills=list_active_skills(mem),
        table_name=table_name,
        store="skill_box",
    )


def search_skills_oraclevs(stores: dict[str, OracleVS], task: str, *,
                           k: int = 3) -> list[SkillVSHit]:
    """Search every OracleVS skill box and merge the top-k by cosine distance."""
    hits: list[SkillVSHit] = []
    for store_name, store in stores.items():
        if store is None:
            continue
        for doc, score in store.similarity_search_with_score(task, k=k):
            hits.append(SkillVSHit(
                store=store_name,
                name=doc.metadata.get("name", ""),
                topic=doc.metadata.get("topic", ""),
                distance=float(score),
                version=int(doc.metadata.get("version", 1)),
                skill_md=doc.metadata.get("skill_md", ""),
                errors_and_fixes=tuple(
                    doc.metadata.get("errors_and_fixes", []) or []
                ),
            ))
    hits.sort(key=lambda h: h.distance)
    return hits[:k]


def search_skill_box(store: OracleVS, task: str, *, k: int = 3) -> list[SkillVSHit]:
    """Return the top-k hits from the single skill-box index."""
    return search_skills_oraclevs({"skill_box": store}, task, k=k)


__all__ = [
    "SkillVSHit",
    "index_active_skills",
    "index_skills_oraclevs",
    "search_skill_box",
    "search_skills_oraclevs",
    "retrievable_base_skills",
]
