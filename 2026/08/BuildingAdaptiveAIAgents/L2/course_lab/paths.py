"""Lesson-local artifact paths — resolve from ./ro_shared_data (self-contained).

Vendored from the course's course_lab/paths.py and repointed at the lesson's
ro_shared_data directory so the lesson runs in its own container with no shared
repo. DLAI copies ro_shared_data into each lesson on deploy.
"""
from __future__ import annotations

import os
from pathlib import Path

# the lesson root holds course_lab/ and ro_shared_data/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RO = _PROJECT_ROOT / "ro_shared_data"
# committed fixtures AND run artifacts both live in ro_shared_data for a lesson
_COURSE_LAB_DATA = _RO
def project_root() -> Path:
    """Repo root (the directory that holds ``course_lab/``)."""
    return _PROJECT_ROOT


def data_dir() -> Path:
    p = Path(os.environ.get("DLAICL_DATA_DIR", _RO))
    p.mkdir(parents=True, exist_ok=True)
    return p


def ckpts_dir() -> Path:
    p = Path(os.environ.get("DLAICL_CKPTS_DIR", _RO / "ckpts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def oamp_sqlite() -> Path:
    """SQLite mirror of the OAMP store (single ``oamp`` table).

    Legacy helper retained for the path registry; no longer produced by the
    active course (the chapter-1 dataset generator that wrote it was removed).
    """
    return data_dir() / "oamp.sqlite"


def memories_jsonl() -> Path:
    return data_dir() / "memories.jsonl"


def eval_qa_jsonl() -> Path:
    return data_dir() / "eval_qa.jsonl"


def m2_llm_fixture_json() -> Path:
    p = data_dir() / "fixtures"
    p.mkdir(parents=True, exist_ok=True)
    return p / "m2_llm_responses.json"


def m2_coding_judge_json() -> Path:
    """Baked answers + judge verdicts for the coding LLM-as-judge lift eval."""
    p = data_dir() / "fixtures"
    p.mkdir(parents=True, exist_ok=True)
    return p / "m2_coding_judge.json"


# ---------------------------------------------------------------------------
# Module 1 — Foundations
# ---------------------------------------------------------------------------

def module1_classification_json() -> Path:
    return data_dir() / "module1_classification.json"


def module1_seed_memories_jsonl() -> Path:
    return data_dir() / "seed_memories.jsonl"


# ---------------------------------------------------------------------------
# Module 2 — Token-space
# ---------------------------------------------------------------------------

def induced_skills_jsonl() -> Path:
    return data_dir() / "induced_skills.jsonl"


def exported_skills_dir() -> Path:
    p = data_dir() / "exported_skills"
    p.mkdir(parents=True, exist_ok=True)
    return p


def vsql_templates_jsonl() -> Path:
    return data_dir() / "vsql_templates.jsonl"


def module2_eval_json() -> Path:
    return data_dir() / "module2_eval.json"


def module2_llm_eval_json() -> Path:
    """Path to the LLM-in-the-loop lift eval output for Module 2."""
    return data_dir() / "module2_llm_eval.json"


# ---------------------------------------------------------------------------
# Module 4 — Model-space: embedding (latent bi-encoder)
# ---------------------------------------------------------------------------

def positive_negative_pairs_jsonl() -> Path:
    return data_dir() / "positive_negative_pairs.jsonl"


def module3_config() -> Path:
    """The Module 3 default run config (Oracle + GPU path)."""
    return _PROJECT_ROOT / "configs" / "default.yaml"


def _course_lab_data() -> Path:
    """Committed fixtures live under course_lab/data/ (not the gitignored data/)."""
    return _RO


def memory_snapshot_json(module: str) -> Path:
    """Committed OCI-free seeding snapshot for a module (e.g. 'module3', 'm2').

    Loaded without live LLM extraction so notebooks seed in the Oracle-only
    sandbox. Exported once on an OCI box via course_lab.memory_snapshot.
    """
    return _course_lab_data() / f"memory_snapshot_{module}.json"


def module3_scm_eval_json() -> Path:
    """Naive-vs-graph recall@k / nDCG@k over the SCM codebase graph arm."""
    return data_dir() / "module3_scm_eval.json"


# ---------------------------------------------------------------------------
# Module 4 — Model-space: embedding (code-search / EmbeddingGemma arm)
# ---------------------------------------------------------------------------

def scm_codebase_json() -> Path:
    """Committed synthetic SCM codebase fixture (function chunks + graph)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "scm_codebase.json"


def pageindex_apple_10k_json() -> Path:
    """Committed real Apple FY2024 10-K PageIndex tree fixture (M3 PageIndex arm)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "pageindex_apple_10k.json"


def pageindex_apple_10k_pages_json() -> Path:
    """Committed real page-text snippets for the Apple FY2024 10-K (PageIndex arm).

    A small ``{page: text}`` map for the financial pages, with Apple's actual
    reported figures, so the PageIndex navigation demo can pull a real answer
    deterministically and offline.
    """
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "pageindex_apple_10k_pages.json"


def partner_companies_json() -> Path:
    """Committed synthetic partner-company profiles fixture (M4 persona arm)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "partner_companies.json"


def persona_pairs_json() -> Path:
    """Committed persona + competence training corpus fixture (M4 persona arm)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "persona_pairs.json"


def polite_pairs_json() -> Path:
    """Committed over-polite persona training corpus fixture (superpolitegemma)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "polite_pairs.json"


def scm_queries_json() -> Path:
    """Committed indirect intent-query fixture: chunk_id -> [NL queries].

    Queries describe each function's purpose WITHOUT naming its symbol/docstring
    tokens (Cursor-style "where do we handle X?"), so the base encoder cannot
    win by lexical match and the graded fine-tune has real headroom.
    """
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "scm_queries.json"


def code_search_traces_jsonl() -> Path:
    return data_dir() / "code_search_traces.jsonl"


def _cs_suffix(source: str | None) -> str:
    # SCM keeps the original unsuffixed filenames (back-compat); other corpora
    # (e.g. agent_harness) get a "_<source>" suffix so artifacts coexist.
    return "" if source in (None, "scm") else f"_{source}"


def code_search_recall_json(source: str | None = None) -> Path:
    return data_dir() / f"code_search_recall_at_k{_cs_suffix(source)}.json"


def code_search_steps_json(source: str | None = None) -> Path:
    return data_dir() / f"code_search_steps_to_target{_cs_suffix(source)}.json"


def code_search_graded_pairs_jsonl(source: str | None = None) -> Path:
    return data_dir() / f"code_search_graded_pairs{_cs_suffix(source)}.jsonl"


def code_search_traces_jsonl_for(source: str | None = None) -> Path:
    return data_dir() / f"code_search_traces{_cs_suffix(source)}.jsonl"


def embeddinggemma_ft_ckpt(source: str | None = None) -> Path:
    return ckpts_dir() / f"embeddinggemma_ft{_cs_suffix(source)}"


# ---------------------------------------------------------------------------
# Module 4 — Model-space: weight (QLoRA)
# ---------------------------------------------------------------------------

def local_qlora_dir() -> Path:
    return ckpts_dir() / "local_qlora"


def tiny_gpt2_dir() -> Path:
    return ckpts_dir() / "tiny-gpt2"


def gemma_3n_e4b_dir() -> Path:
    return ckpts_dir() / "gemma-3n-e4b-it"


def router_decisions_jsonl() -> Path:
    return data_dir() / "router_decisions.jsonl"


def module4_eval_json() -> Path:
    return data_dir() / "module4_eval.json"


# ---------------------------------------------------------------------------
# Module 4 — Model-space weight: local Gemma QLoRA committed artifacts
# ---------------------------------------------------------------------------

_M4_DATA_DIR = _RO


def gemma_cached_responses_json() -> Path:
    """Path to the committed cached generations (base + v1 + v2)."""
    _M4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _M4_DATA_DIR / "gemma_cached_responses.json"


def gemma_adapter_dir(label: str) -> Path:
    """Directory holding the committed PEFT adapter for arm ``v1``/``v2``."""
    d = _M4_DATA_DIR / "adapters" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def persona_cached_responses_json() -> Path:
    """Committed cached generations for the persona arms (base + v1 + v2)."""
    _M4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _M4_DATA_DIR / "persona_cached_responses.json"


def persona_adapter_dir(label: str) -> Path:
    """Directory holding the persona PEFT adapter for arm ``v1``/``v2``
    (gitignored — large; regenerate with run_persona.py train)."""
    d = _M4_DATA_DIR / "persona_adapters" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


def polite_cached_responses_json() -> Path:
    """Committed cached generations for the superpolitegemma arms (base + v1)."""
    _M4_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return _M4_DATA_DIR / "polite_cached_responses.json"


def polite_adapter_dir(label: str) -> Path:
    """Directory holding the superpolitegemma PEFT adapter for arm ``v1``
    (gitignored — large; regenerate with run_polite.py train)."""
    d = _M4_DATA_DIR / "polite_adapters" / label
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Module 3 — Token space: structured space
# ---------------------------------------------------------------------------

def module5_audit_scorecard_json() -> Path:
    return data_dir() / "module5_audit_scorecard.json"


def module5_eval_json() -> Path:
    return data_dir() / "module5_eval.json"


def co_retrieval_graph_jsonl() -> Path:
    return data_dir() / "co_retrieval_graph.jsonl"


def module5_pagerank_ckpt() -> Path:
    return ckpts_dir() / "module5_pagerank"


# --- Module 3 agent-memory co-relevance arm -------------------------------
# vocab + corpus are COMMITTED build inputs (live in course_lab/data/, like the
# SCM codebase fixture); eval + graph edge-list are gitignored run outputs.

def agent_memory_vocab_json() -> Path:
    """Committed snapshot of the harvested typed-node vocab (pref/skill/tool/proc)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "agent_memory_vocab.json"


def agent_chat_corpus_json() -> Path:
    """Committed cache of synthetic chat traces + Grok-rendered surfaces."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "agent_chat_corpus.json"


def agent_harness_code_graph_json() -> Path:
    """Committed code graph parsed from the real agent-harness source tree."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "agent_harness_code_graph.json"


def agent_harness_codebase_json() -> Path:
    """Committed code-search corpus (scm_codebase-shaped) from agent-harness."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "agent_harness_codebase.json"


def agent_harness_queries_json() -> Path:
    """Committed indirect-intent queries for the agent-harness corpus (Grok, cached)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "agent_harness_queries.json"


def module3_agent_memory_eval_json() -> Path:
    return data_dir() / "module3_agent_memory_eval.json"


def agent_memory_graph_jsonl() -> Path:
    return data_dir() / "agent_memory_graph.jsonl"


# ---------------------------------------------------------------------------
# graphify-verification — does the M3 code graph help Claude Code implement
# a feature faster / more correctly than a bare repo?
# ---------------------------------------------------------------------------

def _gv_suffix(task: str | None) -> str:
    """'' for the default httpie task (back-compat filenames), else '_<task>'."""
    return f"_{task}" if task else ""


def graphify_verification_results_json(task: str | None = None) -> Path:
    """Committed headline results (per-arm aggregates + deltas) for the notebook.
    ``task`` namespaces a second target (e.g. 'django_cache') so it doesn't
    clobber the committed httpie results."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / f"graphify_verification_results{_gv_suffix(task)}.json"


def graphify_verification_suite_summary_json() -> Path:
    """Committed 10-task suite summary (per-task deltas + pooled win/tie/loss)."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / "graphify_verification_suite_summary.json"


def graphify_verification_code_graph_json(task: str | None = None) -> Path:
    """Committed code graph parsed from the target repo at the base commit."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / f"graphify_verification_code_graph{_gv_suffix(task)}.json"


def graphify_verification_structure_map_json(task: str | None = None) -> Path:
    """Committed structure map (anchors + graph reach) injected into treatment."""
    _COURSE_LAB_DATA.mkdir(parents=True, exist_ok=True)
    return _COURSE_LAB_DATA / f"graphify_verification_structure_map{_gv_suffix(task)}.json"


def graphify_verification_runs_dir(task: str | None = None) -> Path:
    """Gitignored per-run artifacts (stream-json transcripts, raw traces)."""
    p = data_dir() / f"graphify_verification_runs{_gv_suffix(task)}"
    p.mkdir(parents=True, exist_ok=True)
    return p
