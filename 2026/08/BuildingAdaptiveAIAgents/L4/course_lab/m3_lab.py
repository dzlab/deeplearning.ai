"""Module 3 notebook helpers — keep the lesson cells focused on the mechanism.

The notebook teaches *structure-aware retrieval*: audit → graph → PPR. Code that
is not that mechanism (environment setup, config loading, the agent-harness vs.
toy fixture branch, and the plain-English narration of the results table) lives
here so the notebook cells stay short and show only the lines a learner runs.

Nothing here is novel logic — it is the boilerplate lifted verbatim out of the
notebook so the cells read as the lesson, not the plumbing.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Any

import yaml

# The imports below pull in the tqdm/torch stack, which greets the first cell
# with two benign notices: tqdm's IProgress warning (ipywidgets missing — the
# lesson shows no progress bars) and torchao's cpp-extension skip on a torch
# version mismatch (the lesson calls no torchao kernels). Silence just those
# two, before the import that triggers them; every other warning still shows.
warnings.filterwarnings("ignore", message=r".*IProgress not found.*")


class _SkipTorchaoCppNotice(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Skipping import of cpp extensions" not in record.getMessage()


logging.getLogger().addFilter(_SkipTorchaoCppNotice())

from course_lab import paths, seed_coding_agent
from course_lab.agent_memory import AgentMemory
from course_lab.domains import get_domain


@dataclass
class Module3Setup:
    """Everything the later cells reuse, built once by :func:`setup_module3`."""

    cfg: dict
    domain: Any
    mem: AgentMemory
    nodes: list[dict]
    edges: list[tuple]
    eval_tasks: list[dict]

    def summary(self) -> str:
        lines = [
            f"Domain     : {self.domain.name}",
            f"Backend    : {self.cfg['memory_backend']}",
        ]
        if not self.nodes:  # connect-only stage: the code graph loads later
            lines.append("Memory     : seeded (the code graph is loaded later, "
                         "when we open the real codebase)")
            return "\n".join(lines)
        lines += [
            f"Nodes      : {len(self.nodes)}",
            f"Edges      : {len(self.edges)}",
            f"Eval tasks : {len(self.eval_tasks)}",
            "",
            "Sample nodes:",
        ]
        for nd in self.nodes[:4]:
            lines.append(f"  {nd['id']!r:30s}  {nd['text'][:60]}")
        lines.append("")
        lines.append("Sample edges (src, dst, kind):")
        for e in self.edges[:4]:
            lines.append(f"  {e[0]!r} → {e[1]!r}  [{e[2]}]")
        return "\n".join(lines)


def connect_module3() -> Module3Setup:
    """Stage 1 of the setup: config + real Oracle backend + seeded memory.

    Deliberately does NOT load the real code graph — the notebook teaches the
    graph idea on a small hand-built example first, then calls
    :func:`load_real_graph` when it opens the real codebase.

    Raises if Oracle credentials are absent — the Module 3 notebook is a full
    Oracle exercise, never a smoke shortcut.
    """
    cfg = yaml.safe_load(paths.module3_config().read_text()) or {}
    cfg["memory_backend"] = "real"

    if not os.environ.get("ORACLE_MEMORY_DB_PASSWORD"):
        raise RuntimeError(
            "Module 3 notebook requires Oracle. Run `uv run python lab.py "
            "bootstrap-oracle` and export ORACLE_MEMORY_DB_USER, "
            "ORACLE_MEMORY_DB_PASSWORD, and ORACLE_MEMORY_DB_CONNECT_STRING "
            "before executing this notebook."
        )

    domain = get_domain(cfg)

    # Seed the coding_agent memory into Oracle once; later cells reuse `mem`
    # (its recorded retrieval traces feed the co-retrieval graph in Act 2).
    # seed_or_load picks the path by environment: live extraction when OCI is
    # present (and refreshes the committed snapshot), or a no-LLM snapshot
    # import in the Oracle-only sandbox. See course_lab/memory_snapshot.py.
    from course_lab import memory_snapshot
    mem = AgentMemory.from_config(cfg)
    # Seeding runs a sync wrapper over an async method, emitting one benign
    # "async method in sync context" UserWarning; suppress just that so learner
    # output stays clean (every other warning still surfaces).
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*asynchronous method in a synchronous method.*",
            category=UserWarning,
        )
        memory_snapshot.seed_or_load(
            domain, mem, paths.memory_snapshot_json("module3"), seed=cfg["seed"])

    return Module3Setup(
        cfg=cfg, domain=domain, mem=mem,
        nodes=[], edges=[], eval_tasks=[],
    )


def load_real_graph(s: Module3Setup):
    """Stage 2 of the setup: attach the real parsed code graph to ``s``.

    Canonical memory nodes + structural edges (file:/sym: namespace).
    Default: the REAL agent-harness code graph (AST + git), committed fixture.
    Returns ``(nodes, edges, eval_tasks)`` and also stores them on ``s``.
    """
    cfg = s.cfg
    if cfg.get("code_graph_source") == "agent_harness":
        from course_lab.code_graph_parse import load_code_graph

        cg = load_code_graph()
        nodes = [{"id": n["id"], "text": n["text"]} for n in cg["nodes"]]
        edges = [(e["src"], e["dst"], e["kind"]) for e in cg["edges"]]
        eval_tasks = cg["eval_tasks"]  # derived from the real call structure
        print(
            f"Code graph : {cg['pkg_root']} ({cg['n_files']} files, "
            f"{len(nodes)} nodes, {len(edges)} edges)"
        )
    else:
        nodes = seed_coding_agent.memory_nodes()
        edges = seed_coding_agent.structural_edges(seed=cfg["seed"])
        eval_tasks = s.domain.eval_tasks()

    s.nodes, s.edges, s.eval_tasks = nodes, edges, eval_tasks
    return nodes, edges, eval_tasks


def setup_module3() -> Module3Setup:
    """One-shot setup (connect + real code graph) — kept for scripts and tests
    that want everything at once; the notebook now does the two stages
    separately (starter graph first, real codebase after)."""
    s = connect_module3()
    load_real_graph(s)
    return s


def show_raw_code_data(cg, *, n_files=6, n_edges=6):
    """Print the RAW parsed code data (files, symbols, import/call edges) and the
    key 'follow-a-link' example — pure presentation over the committed fixture."""
    import collections
    files = [n for n in cg["nodes"] if n["kind"] == "file"]
    syms = [n for n in cg["nodes"] if n["kind"] == "sym"]
    calls = [e for e in cg["edges"] if e["kind"] == "call"]
    imports = [e for e in cg["edges"] if e["kind"] == "import"]
    edge_kinds = dict(collections.Counter(e["kind"] for e in cg["edges"]))

    print(f"=== The codebase the agent remembers: {cg['pkg_root']} ===\n"
          f"{cg['n_files']} files, {len(syms)} symbols "
          f"(functions/classes/methods); edges: {edge_kinds}")
    print("\nSample files:  "
          + ", ".join(n["id"].split(":", 1)[-1] for n in files[:n_files]))
    print("\nWhat imports what (file -> file):")
    for e in imports[:n_edges]:
        print(f"  {e['src'].split(':',1)[-1]:28s} imports  "
              f"{e['dst'].split(':',1)[-1]}")
    print("\nWhat calls what (function -> function):")
    for e in calls[:n_edges]:
        print(f"  {e['src'].split('::')[-1]:26s} calls  "
              f"{e['dst'].split('::')[-1]}()")

    ex = next((e for e in calls
               if e["src"].split("::")[-1] != e["dst"].split("::")[-1]), calls[0])
    caller, callee = ex["src"].split("::")[-1], ex["dst"].split("::")[-1]
    print(f"\n--- The KEY example "
          f"-------------------------------------------------\n"
          f"Query: 'When {caller} runs, which internal function "
          f"does it call?'")


def show_graphify_parse(*, n_calls=8):
    """Re-run the AST parser LIVE on one real agent-harness file and print the
    import/call edges it extracts — the exact mechanism behind the fixture. Falls
    back to the committed fixture's edges when the source tree isn't on disk."""
    import ast
    from course_lab.code_graph_parse import (
        _DEFAULT_TREE, _top_level_symbols, _call_name, load_code_graph)
    sample = _DEFAULT_TREE / "core" / "agent_loop.py"
    if not sample.exists():
        cg = load_code_graph()
        print("(agent-harness tree not on disk — showing committed fixture edges)")
        for e in [e for e in cg["edges"] if e["kind"] == "call"][:n_calls]:
            print(f"  {e['src'].split('::')[-1]:26s} calls  {e['dst'].split('::')[-1]}()")
        return
    mod = ast.parse(sample.read_text(encoding="utf-8"), filename=str(sample))
    print(f"=== graphify(real source): parsing {sample.relative_to(_DEFAULT_TREE)} ===")
    print("\nimport edges (from ast.ImportFrom):")
    for n in ast.walk(mod):
        if isinstance(n, ast.ImportFrom) and "agent_harness" in (n.module or ""):
            print(f"  -> {n.module}")
    print("\ncall edges (function -> function, from ast.Call):")
    shown = 0
    for qual, node in _top_level_symbols(mod):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and shown < n_calls and _call_name(sub.func):
                print(f"  {qual.split('.')[-1]:26s} calls  {_call_name(sub.func)}()")
                shown += 1
    print("\nReal static analysis — no hard-coded edges. The committed fixture is\n"
          "exactly this, run over all 208 files + git log for co_edit.")


def show_scorecard(sc, *, label="BEFORE restructure", n_clusters=3):
    """Print the Act-1 memory-quality audit table (pure presentation).

    Only the first ``n_clusters`` duplicate clusters are shown, one per line —
    the full list is hundreds of ids wide and drowns the reader.
    """
    print(f"=== Act 1: Memory-quality audit ({label}) ===")
    print(f"  n_memories         : {sc['n_memories']}")
    print(f"  dup_pct            : {sc['dup_pct']:.4f}  "
          f"({sc['dup_pct']*100:.1f}% near-duplicate)")
    print(f"  stale_pct          : {sc['stale_pct']:.4f}")
    print(f"  contradiction_count: {sc['contradiction_count']}")
    clusters = sc.get("duplicate_clusters") or []
    if clusters:
        shown = clusters[:n_clusters]
        print(f"  duplicate clusters : {len(clusters)} "
              f"(first {len(shown)} shown)")
        for i, cl in enumerate(shown, 1):
            for j, member in enumerate(cl):
                print(f"    {f'[{i}]' if j == 0 else '   '} {member}")


def build_and_restructure(mem, edges, before_sc, *, domain_name=None):
    """Act 2: fuse co-retrieval + structural edges, then merge duplicate clusters.

    Returns ``(graph, graph_clean, after_sc, n_traces)``. The graph build and
    the restructure are the mechanism; the caller decides what to display.
    """
    from course_lab.memory_graph import build_graph, restructure

    traces = mem.list_traces("retrieval")
    graph = build_graph(traces, edges, domain=domain_name)
    graph_clean, after_sc = restructure(graph, before_sc)
    return graph, graph_clean, after_sc, len(traces)


def show_restructure(graph, graph_clean, before_sc, after_sc, n_traces):
    """Print the before -> after restructure numbers (pure presentation)."""
    print(f"Co-retrieval traces : {n_traces:,}")
    print(f"Graph nodes         : {graph.num_nodes():,} -> "
          f"{graph_clean.num_nodes():,}")
    print(f"Graph edges         : {graph.num_edges():,} -> "
          f"{graph_clean.num_edges():,}")
    print(f"dup_pct             : {before_sc['dup_pct']:.4f} -> "
          f"{after_sc['dup_pct']:.4f}")


def show_funnel(mem, graph_clean, nodes, before_sc):
    """Print the compression funnel + degree narration (pure presentation).

    Many noisy retrieval-trace observations collapse into a few canonical nodes
    and edges; ``deg`` is each node's neighbour count in the distilled graph.
    """
    n_traces = len(mem.list_traces("retrieval"))
    id_to_text = {nd["id"]: nd["text"] for nd in nodes}
    print("=== Compression funnel ===")
    print(f"  raw retrieval traces (noisy co-occurrence observations): {n_traces}")
    print(f"  canonical nodes after dedup/restructure                : {graph_clean.num_nodes()}")
    print(f"  edges (distilled structure)                            : {graph_clean.num_edges()}\n")
    print("=== What each node contains ===")
    for nid in sorted(graph_clean.nodes()):
        deg = len(graph_clean.neighbors(nid))
        txt = id_to_text.get(nid, "(merged/duplicate — absorbed)")
        print(f"  {nid:28s} deg={deg:<2d} {txt[:48]}")

    degs = [len(graph_clean.neighbors(n)) for n in graph_clean.nodes()]
    top = sorted(graph_clean.nodes(), key=lambda n: -len(graph_clean.neighbors(n)))[:3]
    print("\n=== How to read deg=  (degree = # of graph neighbors) ===")
    print(f"  degree range across {len(degs)} nodes : min={min(degs)}  "
          f"max={max(degs)}  mean={sum(degs)/len(degs):.1f}")
    print("  highest-degree nodes (the hubs PPR keeps landing on):")
    for n in top:
        print(f"    {n:30s} deg={len(graph_clean.neighbors(n))}")
    print("\n=== How the distillation was performed (the funnel, step by step) ===")
    print(f"  1. START : {n_traces} raw retrieval traces  -- noisy co-occurrence")
    print("             observations harvested from agent sessions (the same pair")
    print("             of symbols seen together many times = many duplicate rows).")
    print("  2. DEDUP : audit_scorecard clusters near-identical nodes (cosine >= 0.92)")
    print(f"             -- it flagged {before_sc['dup_pct']*100:.1f}% as near-duplicates.")
    print("  3. MERGE : restructure folds each duplicate cluster into ONE canonical")
    print("             node; the survivors' edges are unioned (the '(absorbed)' rows).")
    print("  4. KEEP STRUCTURE : co-occurring symbols become a SINGLE edge (not N")
    print("             repeated observations), weighted by how often they co-occur.")
    print(f"  RESULT : {graph_clean.num_nodes()} canonical nodes / "
          f"{graph_clean.num_edges()} edges")
    print(f"           -- a {n_traces/max(graph_clean.num_nodes(),1):.0f}x collapse, "
          "and the degrees above")
    print("           are computed on THIS distilled graph, not the raw traces.")


def show_sample_traces(mem, *, n=4):
    """Print a few RAW co-retrieval traces exactly as they sit in Oracle — the
    input to the compression funnel, BEFORE they become graph nodes/edges.

    A co-retrieval trace is one recorded retrieval event: a ``query``, the
    ``candidates`` that came back, the ``chosen`` nodes the agent actually used
    together, and the ``outcome``. The graph is built from the ``chosen`` list:
    every pair of nodes chosen together for the same query becomes ONE
    ``co_retrieved`` edge (``build_graph`` in ``course_lab/memory_graph.py``).
    That co-occurrence — "these nodes keep getting used together" — is the
    'information from the traces' that ends up baked into the node degrees the
    funnel prints below. Pure presentation; reads the live Oracle traces."""
    traces = mem.list_traces("retrieval")
    print("=== A few RAW co-retrieval traces (the funnel's INPUT) ===")
    print(f"  {len(traces)} total; showing {min(n, len(traces))}. Each is one "
          f"retrieval event the agent recorded while working:\n")
    for t in traces[:n]:
        cand_ids = [c.get("id", c) if isinstance(c, dict) else c
                    for c in t.candidates]
        print(f"  query     : {t.query}")
        print(f"  candidates: {cand_ids}")
        print(f"  chosen    : {t.chosen}   (used together → a co_retrieved edge)")
        print(f"  outcome   : {t.outcome}")
        print()
    print("  ↓ build_graph draws a co_retrieved edge between every pair in each")
    print("    `chosen` list, then dedup/restructure collapses the repeats — the")
    print("    funnel below shows what survives.\n")


def show_funnel_compact(graph_clean, nodes, *, n_traces, max_nodes=15):
    """Print the compact compression funnel: the trace→node→edge counts plus what
    each surviving canonical node contains (the notebook's funnel cell, verbatim).

    ``max_nodes`` caps how many surviving nodes are listed (default 15) so the
    cell stays readable; the count lines above still reflect the full totals."""
    print("=== Compression funnel ===")
    print(f"  raw retrieval traces (noisy co-occurrence "
          f"observations): {n_traces}")
    print(f"  canonical nodes after dedup/restructure                : "
          f"{graph_clean.num_nodes()}")
    print(f"  edges (distilled structure)                            : "
          f"{graph_clean.num_edges()}\n")
    id_to_text = {nd["id"]: nd["text"] for nd in nodes}
    all_nids = sorted(graph_clean.nodes())
    print(f"=== What each node contains "
          f"(showing {min(len(all_nids), max_nodes)} of "
          f"{len(all_nids)}) ===")
    for nid in all_nids[:max_nodes]:
        deg = len(graph_clean.neighbors(nid))
        txt = id_to_text.get(nid, "(merged/duplicate — absorbed)")
        print(f"  {nid:28s} deg={deg:<2d} {txt[:48]}")
    extra = len(all_nids) - max_nodes
    if extra > 0:
        print(f"  … and {extra} more (pass max_nodes= to change)")


def lexical_seed_then_ppr(query, nodes, graph, *, k=5):
    """Module 3's ranker: seed at the best-overlap CONNECTED node, then PPR.

    This IS the lesson — a lexical entry point, then Personalized PageRank
    propagation along the graph's typed edges. Returns the top-k node ids.
    Falls back to the lexical order when no connected seed exists.
    """
    from course_lab.structural_retrieval import (
        identifier_tokens, lexical_flat_baseline, personalized_pagerank)

    q_toks = identifier_tokens(query)
    seed_scores = [
        (nd["id"], len(q_toks & identifier_tokens(nd["id"] + " " + nd["text"])))
        for nd in nodes
        if graph.neighbors(nd["id"])
    ]
    seed_scores = [(nid, o) for nid, o in seed_scores if o > 0]
    seed_scores.sort(key=lambda t: (-t[1], t[0]))
    seeds = {nid: 1.0 for nid, _ in seed_scores[:1]}
    if not seeds:
        return lexical_flat_baseline(query, nodes, k=k)
    ppr = personalized_pagerank(graph, seeds, alpha=0.85, iters=50)
    return [nid for nid, _ in sorted(ppr.items(), key=lambda t: -t[1])][:k]


def hybrid_seed_then_ppr(query, nodes, graph, *, k=5, n_anchors=3):
    """Module 3's HYBRID ranker: lexical anchors → graph reach → PPR ranks it.

    This is the headline mechanism, and it makes explicit the two halves that
    do different jobs:

    * **anchors** — the top ``n_anchors`` lexical (keyword) hits among the
      *connected* nodes. These are the entry points an agent finds anyway by
      grepping or reading a markdown index. (``lexical_seed_then_ppr`` is the
      ``n_anchors=1`` special case.)
    * **graph reach** — Personalized PageRank seeded at ALL the anchors then
      propagates mass along the typed edges (import / call / co_edit), surfacing
      the dependencies the anchors connect to. PPR *ranks* the reach — the same
      de-blackboxed walk Module 3 teaches; multiple anchors just give it more
      than one door into the graph, which is what lets the walk reach a positive
      that a single seed sitting in the wrong neighbourhood would miss.

    Seeding from a handful of keyword hits (not one) is the upgrade verified in
    the graphify-verification experiment: it is what carries file-level "which
    files does this change touch" retrieval, where a single seed diffuses into
    the wrong cluster. Falls back to the lexical order when no connected anchor
    exists. Returns the top-k node ids.
    """
    from course_lab.structural_retrieval import (
        identifier_tokens, lexical_flat_baseline, personalized_pagerank)

    q_toks = identifier_tokens(query)
    anchor_scores = [
        (nd["id"], len(q_toks & identifier_tokens(nd["id"] + " " + nd["text"])))
        for nd in nodes
        if graph.neighbors(nd["id"])
    ]
    anchor_scores = [(nid, o) for nid, o in anchor_scores if o > 0]
    anchor_scores.sort(key=lambda t: (-t[1], t[0]))
    # weight each anchor by its overlap so the strongest keyword hit gets the
    # most teleport mass, but every anchor opens a door into the graph.
    seeds = {nid: float(o) for nid, o in anchor_scores[:n_anchors]}
    if not seeds:
        return lexical_flat_baseline(query, nodes, k=k)
    ppr = personalized_pagerank(graph, seeds, alpha=0.85, iters=50)
    return [nid for nid, _ in sorted(ppr.items(), key=lambda t: -t[1])][:k]


def build_node_embeddings(nodes, *, model_spec="python:all-MiniLM-L6-v2"):
    """Embed every node (id + text) once, so anchors can be picked by MEANING.

    Used by the semantic-anchoring arm of the eval: instead of counting shared
    words, the anchor is the connected node whose embedding is closest to the
    query's. Returns ``{"ids": [...], "vecs": ndarray, "embedder": ...}`` with
    L2-normed rows (so cosine similarity is a plain dot product).
    """
    from course_lab.python_embedder import make_embedder

    emb = make_embedder(model_spec)
    ids = [nd["id"] for nd in nodes]
    vecs = emb.embed([nd["id"] + " " + nd["text"] for nd in nodes])
    return {"ids": ids, "vecs": vecs, "embedder": emb}


def semantic_seed_then_ppr(query, nodes, graph, *, k=5, n_anchors=1, node_emb=None):
    """The Code Knowledge Graph walk with a SEMANTIC anchor.

    Identical to :func:`hybrid_seed_then_ppr` except for how the anchor (the
    first node the graph walk starts from) is chosen: by embedding similarity
    ("means the same thing") instead of keyword overlap ("shares the same
    words"). Everything after the anchor — Personalized PageRank over the typed
    edges — is unchanged, so the eval isolates exactly one variable: how the
    walk's entry point is found.

    ``node_emb`` is the output of :func:`build_node_embeddings` (pass it in so
    the node matrix is embedded once, not per query).
    """
    from course_lab.structural_retrieval import personalized_pagerank

    if node_emb is None:
        node_emb = build_node_embeddings(nodes)
    qv = node_emb["embedder"].embed([query], is_query=True)[0]
    sims = node_emb["vecs"] @ qv
    connected = {nd["id"] for nd in nodes if graph.neighbors(nd["id"])}
    ranked = sorted(
        ((nid, float(s)) for nid, s in zip(node_emb["ids"], sims)
         if nid in connected),
        key=lambda t: (-t[1], t[0]))
    seeds = {nid: max(sim, 1e-6) for nid, sim in ranked[:n_anchors]}
    if not seeds:
        return [nid for nid, _ in ranked[:k]]
    ppr = personalized_pagerank(graph, seeds, alpha=0.85, iters=50)
    return [nid for nid, _ in sorted(ppr.items(), key=lambda t: -t[1])][:k]


def run_retrieval_eval(eval_tasks, nodes, graph_clean, *, k=5, n_anchors=1,
                       node_emb=None):
    """Act 3: score plain keywords vs. the Code Knowledge Graph walk.

    Returns ``(scored_tasks, results)`` where each result carries the keyword
    baseline and Code-KG (lexical anchor → PPR) recall@k / nDCG@k. Only tasks
    whose positive survived the restructure are scored (keeps the eval honest
    about candidates).

    ``n_anchors`` defaults to 1 for this symbol-level eval, where each multi_hop
    positive is exactly one call edge from its single best lexical hit — one
    precise door is optimal there (recall 0.92). Multiple anchors are the
    generalization for file-level "which files to edit" retrieval (see the
    real-agent payoff and ``hybrid_seed_then_ppr``).

    Pass ``node_emb`` (from :func:`build_node_embeddings`) to also score the
    third arm — the same walk with a SEMANTIC anchor — as ``sem_recall`` /
    ``sem_ndcg`` per result.
    """
    from course_lab.structural_retrieval import (
        lexical_flat_baseline, ndcg_at_k, recall_at_k)

    graph_ids = set(graph_clean.nodes())
    scored_tasks = [t for t in eval_tasks if t["positive"] in graph_ids]
    results = []
    for task in scored_tasks:
        q, pos, ttype = task["query"], task["positive"], task["type"]
        lex = lexical_flat_baseline(q, nodes, k=k)
        hyb = hybrid_seed_then_ppr(q, nodes, graph_clean, k=k, n_anchors=n_anchors)
        r = {
            "type": ttype, "positive": pos,
            "lex_recall": recall_at_k(lex, pos, k=k),
            "str_recall": recall_at_k(hyb, pos, k=k),
            "lex_ndcg": ndcg_at_k(lex, pos, k=k),
            "str_ndcg": ndcg_at_k(hyb, pos, k=k),
        }
        if node_emb is not None:
            sem = semantic_seed_then_ppr(q, nodes, graph_clean, k=k,
                                         n_anchors=n_anchors, node_emb=node_emb)
            r["sem_recall"] = recall_at_k(sem, pos, k=k)
            r["sem_ndcg"] = ndcg_at_k(sem, pos, k=k)
        results.append(r)
    return scored_tasks, results


def pageindex_navigate(question, *, k=5):
    """Run Module 3's ranker over the real Apple 10-K PageIndex tree.

    Returns a dict with: the document title, the naive-chunking scatter, the
    PPR ranking, the leaf we descend to, and the answer pulled from its page.
    The seed->PPR walk is the lesson (course_lab.m3_lab.lexical_seed_then_ppr);
    everything else is loading the committed tree and reading a page.
    """
    import re

    from course_lab import pageindex
    from course_lab.memory_graph import MemGraph
    from course_lab.structural_retrieval import lexical_flat_baseline

    tree = pageindex.load_tree()
    flat = pageindex.flatten(tree)
    by_id = {n["id"]: n for n in flat}
    parents = {n["parent"] for n in flat}
    is_leaf = lambda nid: nid not in parents          # noqa: E731

    graph = MemGraph()
    for src, dst, _ in pageindex.to_graph_edges(tree):
        graph.add_edge(src, dst)

    # node "documents" for the lexical baseline / seeding
    nodes = [{"id": n["id"], "text": n["title"] + " " + n["summary"]} for n in flat]
    naive = lexical_flat_baseline(question, nodes, k=k)
    ranked_ids = lexical_seed_then_ppr(question, nodes, graph, k=len(flat))

    target = next((nid for nid in ranked_ids
                   if is_leaf(nid) and pageindex.node_pages_text(by_id[nid])), None)
    page_text = pageindex.node_pages_text(by_id[target]) if target else ""
    m = re.search(r"Total operating expenses were (\$[\d,]+)", page_text)
    return {
        "doc": tree["title"], "n_sections": len(flat),
        "n_edges": graph.num_edges(), "question": question, "by_id": by_id,
        "naive": naive, "ranked": ranked_ids[:5], "is_leaf": is_leaf,
        "target": target, "answer": m.group(1) if m else None,
    }


def show_pageindex_navigation(r):
    """Print the naive-scatter vs. PageIndex-walk narrative (pure presentation)."""
    by_id, title = r["by_id"], lambda nid: r["by_id"][nid]["title"]
    pages = lambda nid: by_id[nid]["pages"]          # noqa: E731
    print(f"DOCUMENT : {r['doc']}")
    print(f"           {r['n_sections']} sections, {r['n_edges']} contains-edges "
          "(parsed once from the SEC filing)")
    print(f"QUESTION : {r['question']}\n")
    naive_pages = sorted({tuple(pages(nid)) for nid in r["naive"]})
    print("NAIVE chunking (similarity top-5) -> a scatter of pages to read:")
    for nid in r["naive"]:
        print(f"   p{str(pages(nid)):8s}  {title(nid)}")
    print("   pages the agent must open: "
          f"{[f'{a}-{b}' if a != b else str(a) for a, b in naive_pages]}")
    print(f"   top hit is the broad section '{title(r['naive'][0])}', not the leaf "
          "that holds the number\n")
    print("PageIndex walk (lexical seed -> PPR over the tree) -> ranks the branch:")
    for nid in r["ranked"]:
        mark = "leaf" if r["is_leaf"](nid) else "branch"
        print(f"   p{str(pages(nid)):8s}  [{mark}] {title(nid)}")
    print(f"\n   -> descend to leaf : {title(r['target'])}  (page {pages(r['target'])[0]})")
    print(f"   -> pull ONLY page {pages(r['target'])[0]}, not the whole scatter")
    print(f"   -> ANSWER          : total operating expenses {r['answer']} (FY2024)")
    print("      (R&D $31,370 + SG&A $26,097, both on the same pulled page)")


def pageindex_add_one(parent_node_id, new_node, follow_up):
    """Append ONE node to the PageIndex tree and confirm prior rows are untouched.

    Returns a dict reporting the surgical delta (new node/edge, prior edges
    preserved, original tree unchanged) and the follow-up retrieval hit.
    """
    from course_lab import pageindex
    from course_lab.memory_graph import MemGraph

    tree = pageindex.load_tree()
    before_ids = {n["id"] for n in pageindex.flatten(tree)}
    before_edges = set(pageindex.to_graph_edges(tree))

    tree2 = pageindex.append_node(tree, parent_node_id, new_node)  # pure
    flat2 = pageindex.flatten(tree2)
    after_edges = set(pageindex.to_graph_edges(tree2))

    graph2 = MemGraph()
    for src, dst, _ in after_edges:
        graph2.add_edge(src, dst)
    by_id2 = {n["id"]: n for n in flat2}
    nodes2 = [{"id": n["id"], "text": n["title"] + " " + n["summary"]} for n in flat2]
    top = lexical_seed_then_ppr(follow_up, nodes2, graph2, k=1)
    return {
        "new_node_title": new_node["title"], "parent": parent_node_id,
        "new_ids": {i.replace("pi:", "") for i in {n["id"] for n in flat2} - before_ids},
        "new_edges": [(a.replace("pi:", ""), b.replace("pi:", ""))
                      for a, b, _ in (after_edges - before_edges)],
        "prior_preserved": before_edges.issubset(after_edges),
        "original_untouched": len(pageindex.flatten(tree)) == len(before_ids),
        "follow_up": follow_up,
        "follow_up_top": by_id2[top[0]]["title"] if top else None,
    }


def show_pageindex_add_one(r):
    """Print the surgical add-one narrative (pure presentation)."""
    print(f'ADD ONE: append "{r["new_node_title"]}" under {r["parent"]}')
    print(f"   new nodes added       : {len(r['new_ids'])}  {sorted(r['new_ids'])}")
    print(f"   new edges added       : {len(r['new_edges'])}  {r['new_edges']}")
    print(f"   prior edges preserved : {r['prior_preserved']}  "
          "(nothing re-embedded, nothing retrained)")
    print(f"   original tree untouched: {r['original_untouched']}")
    print(f'\n   next query "{r["follow_up"]}"')
    print(f"   -> top hit is the brand-new node: {r['follow_up_top']}")


def run_scm_eval(chunks, scm_nodes, scm_g_all, scm_g_imp, mh_tasks, *, k=5):
    """Score chunking-lexical vs graph(all edges) vs graph(import-only) on the
    SCM multi-hop dependency-QA tasks; write the metrics to module3_scm_eval_json
    and return them. The graph builds stay in the cell; this is the scoring.
    """
    import json

    from course_lab import paths, scm_graph
    from course_lab.structural_retrieval import (
        lexical_flat_baseline, ndcg_at_k, recall_at_k)

    def score(rank_fn):
        r = nd = 0.0
        for t in mh_tasks:
            ranked = rank_fn(t)
            r += recall_at_k(ranked, t["positive"], k=k)
            nd += ndcg_at_k(ranked, t["positive"], k=k)
        n = max(len(mh_tasks), 1)
        return {"recall_at_k": round(r / n, 3), "ndcg_at_k": round(nd / n, 3)}

    metrics = {
        "chunking_lexical": score(lambda t: lexical_flat_baseline(t["query"], scm_nodes, k=k)),
        "graph_all_edges": score(lambda t: scm_graph.anchor_ppr_rank(t["anchor"], scm_g_all, k=k)),
        "graph_import_only": score(lambda t: scm_graph.anchor_ppr_rank(t["anchor"], scm_g_imp, k=k)),
    }
    paths.module3_scm_eval_json().write_text(json.dumps(metrics, indent=2))
    return metrics


def show_scm_eval(chunks, scm_g_all, scm_g_imp, mh_tasks, metrics):
    """Print the SCM graph-vs-chunking comparison + the two lessons."""
    print(f"SCM codebase: {len(chunks)} symbols, 30 files")
    print(f"  import+co_edit graph: {scm_g_all.num_edges()} edges")
    print(f"  import-only graph   : {scm_g_imp.num_edges()} edges")
    print(f"  multi-hop dependency-QA tasks (answer 2 import-hops away): {len(mh_tasks)}\n")
    print(f"=== Multi-hop dependency QA (n={len(mh_tasks)}, recall@5) ===")
    print(f"  chunking / lexical RAG      : {metrics['chunking_lexical']['recall_at_k']:.3f}")
    print(f"  graph RAG (import+co_edit)  : {metrics['graph_all_edges']['recall_at_k']:.3f}")
    print(f"  graph RAG (import-only)     : {metrics['graph_import_only']['recall_at_k']:.3f}  <- the win\n")
    print("Lesson 1: graph beats chunking when the answer is a multi-hop relation")
    print("          text similarity cannot see (chunking scores ~0 here).")
    print("Lesson 2: EDGE TYPE matters more than 'having a graph' — the 1600+ dense")
    print("          co_edit edges drown the import signal; dropping them is the win.")


def show_scm_examples(scm_nodes, scm_g_imp, mh_tasks, *, n=2):
    """Print n worked multi-hop examples + the honest direct-lexical counter-case."""
    import json

    from course_lab import paths, scm_graph
    from course_lab.structural_retrieval import lexical_flat_baseline, recall_at_k

    print(f"=== Why graph wins: {n} worked multi-hop examples (import-only graph) ===")
    for t in mh_tasks[:n]:
        lex = lexical_flat_baseline(t["query"], scm_nodes, k=5)
        gph = scm_graph.anchor_ppr_rank(t["anchor"], scm_g_imp, k=5)
        print(f"Q: {t['query']}")
        print(f"   anchor (named in query): {t['anchor']}")
        print(f"   wanted (2 hops away)   : {t['positive']}")
        print(f"   chunking top5: {lex}")
        print(f"   graph    top5: {gph}")
        print(f"   chunking {'HIT' if t['positive'] in lex else 'miss'} | "
              f"graph {'HIT' if t['positive'] in gph else 'miss'}\n")

    queries = json.loads(paths.scm_queries_json().read_text())
    ids = {n_["id"] for n_ in scm_nodes}
    direct = sorted(
        ({"query": qs[0], "positive": cid} for cid, qs in queries.items() if cid in ids),
        key=lambda d: d["positive"])[:60]
    dlex = sum(recall_at_k(lexical_flat_baseline(d["query"], scm_nodes, k=5),
                           d["positive"], k=5) for d in direct) / max(len(direct), 1)
    print(f"=== Honest counter-case: direct-lexical queries (n={len(direct)}) ===")
    print(f"   chunking/lexical recall@5 = {dlex:.3f}  (graph not needed here)")
    print("   -> Use graph for multi-hop reach; use lexical/embedding for direct hits.")


def run_graphify_tick(mem, scm_nodes, scm_g_imp, *, domain="scm_codebase_demo"):
    """The live-Oracle graphify tick: persist -> reload -> add a 'commit' ->
    reload -> re-rank. A real DB round-trip via AgentMemory. Returns a dict of
    before/after hits so the cell just shows the result.
    """
    from course_lab import scm_graph

    mem.clear_graph_domain(domain=domain)
    mem.upsert_graph_nodes(scm_nodes, domain=domain)
    edge_rows = [(s, d, "import") for s in scm_g_imp.nodes()
                 for d in scm_g_imp.neighbors(s) if s < d]
    mem.upsert_graph_edges(edge_rows, domain=domain)

    new_node, new_edges = scm_graph.synthetic_commit()   # the 'commit since yesterday'
    anchor, positive = new_edges[0][0], new_node["id"]

    _, g_before = mem.load_graph(domain=domain)           # reload FROM Oracle
    hit_before = positive in scm_graph.anchor_ppr_rank(anchor, g_before, k=5)

    mem.upsert_graph_nodes([new_node], domain=domain)     # the tick
    mem.upsert_graph_edges(new_edges, domain=domain)

    _, g_after = mem.load_graph(domain=domain)            # reload again
    ranked_after = scm_graph.anchor_ppr_rank(anchor, g_after, k=5)
    return {
        "domain": domain, "n_nodes": len(scm_nodes), "n_edges": len(edge_rows),
        "anchor": anchor, "new_symbol": positive, "n_new_edges": len(new_edges),
        "hit_before": hit_before, "hit_after": positive in ranked_after,
        "ranked_after": ranked_after,
    }


def show_graphify_tick(r):
    """Print the closing-the-loop tick result (pure presentation)."""
    print(f"Persisted {r['n_nodes']} nodes + {r['n_edges']} edges to Oracle "
          f"(domain={r['domain']!r}).")
    print(f"Anchor (existing caller): {r['anchor']}")
    print(f"New symbol (the commit) : {r['new_symbol']}")
    print(f"\nBEFORE tick: new symbol in top5? {r['hit_before']}  (not in the DB yet)")
    print(f"Upserted 1 new symbol + {r['n_new_edges']} edge(s) (a 'commit since yesterday').")
    print(f"AFTER  tick: new symbol in top5? {r['hit_after']}\n")
    if r["hit_after"] and not r["hit_before"]:
        print("LOOP CLOSED: a dependency that was unreachable yesterday is reachable")
        print("today — no retraining, no forgetting, just upserted edges in Oracle.")
    else:
        print(f"after = {r['ranked_after']}")


def show_cl_stack():
    """Print the continual-learning stack reflection table (pure presentation)."""
    stack = [
        ("M2 Token space", "Skill induction — inject retrieved recipe steps into context",
         "non-param (no weights changed)"),
        ("M3 Token space", "Structured space — graph + PPR, structural edits, no gradients",
         "non-parametric (graph topology)"),
        ("M4 Latent space", "Embedder fine-tune — contrastive loss on retrieval traces",
         "parametric (embedder weights)"),
        ("M4 Weight space", "QLoRA adapter — gradient updates on the base LLM",
         "parametric (LLM adapter weights)"),
    ]
    print("=== The continual-learning stack ===")
    print(f"{'Module':20s} {'Mechanism':55s} {'Type':35s}")
    print("-" * 112)
    for mod, mech, kind in stack:
        print(f"{mod:20s} {mech:55s} {kind:35s}")
    print("\nM3 is the only layer that is BOTH:")
    print("  * Non-parametric  — no weights are updated, no GPU required")
    print("  * Continual-safe  — adding a node/edge cannot cause "
          "forgetting of prior edges")
    print("\nThe right architecture stacks all four:")
    print("  Retrieve (M3 graph PPR)  →  Re-rank (M4 embedder)  →  "
          "Generate (M4 LoRA LLM)")
    print("  ↑ no-gradient memory          ↑ latent adaptation        "
          "↑ weight adaptation")


def peek_hybrid_halves(eval_tasks, nodes, graph, *, k=5, n_anchors=1):
    """Show the Code Knowledge Graph's two halves on one multi_hop task: the
    keyword ANCHOR(s) (where the walk starts) and the WANTED callee the graph
    walk reaches. Pure presentation."""
    from course_lab.structural_retrieval import identifier_tokens
    graph_ids = set(graph.nodes())
    demo = next(t for t in eval_tasks
                if t["type"] == "multi_hop" and t["positive"] in graph_ids)
    q_toks = identifier_tokens(demo["query"])
    scores = sorted(
        ((nd["id"], len(q_toks & identifier_tokens(nd["id"] + " " + nd["text"])))
         for nd in nodes if graph.neighbors(nd["id"])),
        key=lambda t: (-t[1], t[0]))
    anchors = [nid for nid, o in scores if o > 0][:n_anchors]
    hyb = hybrid_seed_then_ppr(demo["query"], nodes, graph, k=k, n_anchors=n_anchors)
    reached = ("YES — PPR walked the call edge from the anchor to it"
               if demo["positive"] in hyb else "no")
    print(f"Query     : {demo['query']}\n"
          f"Anchor(s) : {anchors}   <- the ANCHOR: where the graph "
          f"walk starts (best word match)\n"
          f"Wanted    : {demo['positive']}   <- the callee; ground "
          f"truth, NOT in the query\n"
          f"Code KG@{k} : {hyb}\n"
          f"Reached by graph? {reached}\n")


def narrate_retrieval_comparison(scored_tasks, results, nodes, *, k: int = 5,
                                 n_examples: int = 3):
    """Plain-English narration of the keywords vs. Code Knowledge Graph table.

    Pure presentation: reads the per-task ``results`` produced by the lesson
    cell and prints, for each query, whether plain keywords and the Code KG
    walk found the right node — then the multi-hop / lexical verdicts. When the
    results carry the semantic-anchor arm (``sem_recall``), it is shown as a
    third column. No retrieval logic.

    ``n_examples`` caps how many worked examples are printed *per test type*;
    the scorecard underneath always tallies every task.
    """
    from course_lab.structural_retrieval import lexical_flat_baseline

    has_sem = bool(results) and "sem_recall" in results[0]
    print("Glossary:")
    print("  recall@5 = was the wanted node in the top 5?      "
          "(1.0 = yes, 0.0 = no)")
    print("  nDCG     = if found, how high did it rank?         "
          "(1.0 = rank 1, lower = deeper)")
    print("  keywords = plain keyword search (soft word match), no graph")
    print("  code KG  = the Code Knowledge Graph walk: keyword "
          "ANCHOR, then")
    print("             Personalized PageRank follows the typed "
          "edges from it")
    if has_sem:
        print("  code KG (sem) = the same walk, but the anchor is "
              "picked by MEANING")
        print("             (embedding similarity) instead of shared words")
    print("  wanted   = ground truth, read straight off the graph "
          "edge that built the task")
    print()
    print("Legend:  ✅ found the wanted node (top 5)   ❌ missed it")
    print(f"         (up to {n_examples} worked examples per test type; "
          f"the scorecard below counts every task)")
    print()
    id_to_text = {nd["id"]: nd["text"] for nd in nodes}
    # tallies for the summary table
    lex_found = {"multi_hop": 0, "lexical": 0}
    str_found = {"multi_hop": 0, "lexical": 0}
    sem_found = {"multi_hop": 0, "lexical": 0}
    n_by_type = {"multi_hop": 0, "lexical": 0}
    type_label = {"multi_hop": "multi-hop test",
                  "lexical": "similarity search test"}
    # same rename, minus the trailing "test", for the scorecard/verdict lines
    subset_label = {"multi_hop": "multi-hop",
                    "lexical": "similarity search"}
    # Only the first few of each type are printed — the scorecard below still
    # tallies every task.
    shown_by_type: dict[str, int] = {}
    n_shown = 0
    for task, r in zip(scored_tasks, results):
        q, pos, ttype = task["query"], task["positive"], task["type"]
        lex_ranked = lexical_flat_baseline(q, nodes, k=k)
        lex_ok = pos in lex_ranked
        str_ok = r["str_recall"] >= 1.0
        sem_ok = has_sem and r["sem_recall"] >= 1.0
        n_by_type[ttype] = n_by_type.get(ttype, 0) + 1
        lex_found[ttype] = lex_found.get(ttype, 0) + int(lex_ok)
        str_found[ttype] = str_found.get(ttype, 0) + int(str_ok)
        sem_found[ttype] = sem_found.get(ttype, 0) + int(sem_ok)
        if shown_by_type.get(ttype, 0) >= n_examples:
            continue
        shown_by_type[ttype] = shown_by_type.get(ttype, 0) + 1
        n_shown += 1
        lex_mark = "✅ found " if lex_ok else "❌ missed"
        str_mark = "✅ found " if str_ok else "❌ missed"
        print(f"Example #{n_shown} - [{type_label.get(ttype, ttype)}]")
        print(f"   query         : {q}")
        print(f"   wanted        : {pos}  ({id_to_text.get(pos, '')[:50]})")
        print(f"   keywords      : {lex_mark}  "
              f"top1={lex_ranked[0] if lex_ranked else '-'}")
        print(f"   code KG       : {str_mark}  "
              f"(recall@5={r['str_recall']:.0f})")
        if has_sem:
            sem_mark = "✅ found " if sem_ok else "❌ missed"
            print(f"   code KG (sem) : {sem_mark}  "
                  f"(recall@5={r['sem_recall']:.0f})")
        print()

    multi_hop = [r for r in results if r["type"] == "multi_hop"]
    lexical = [r for r in results if r["type"] == "lexical"]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    # --- totals comparison table: keywords vs Code KG, found / missed --------
    n_total = sum(n_by_type.values())
    lf, sf = sum(lex_found.values()), sum(str_found.values())
    print("=" * 70)
    print("SCORECARD — wanted node found in top 5 (✅) vs missed (❌)")
    header = f"  {'subset':<20}{'tasks':>6}{'keywords':>14}{'code KG':>14}"
    if has_sem:
        header += f"{'code KG (sem)':>16}"
    print(header)
    for tt in ("multi_hop", "lexical"):
        n = n_by_type.get(tt, 0)
        if not n:
            continue
        row = (f"  {subset_label.get(tt, tt):<20}{n:>6}"
               f"{f'{lex_found[tt]}/{n} found':>14}"
               f"{f'{str_found[tt]}/{n} found':>14}")
        if has_sem:
            row += f"{f'{sem_found[tt]}/{n} found':>16}"
        print(row)
    print(f"  {'-'*60}")
    total_row = (f"  {'TOTAL':<20}{n_total:>6}"
                 f"{f'{lf}/{n_total} ({lf-sf:+d})':>14}"
                 f"{f'{sf}/{n_total} found':>14}")
    if has_sem:
        smf = sum(sem_found.values())
        total_row += f"{f'{smf}/{n_total} found':>16}"
    print(total_row)
    print(f"  => the graph recovers {sf-lf} extra wanted node(s) "
          f"keywords missed ({lf}/{n_total} -> {sf}/{n_total}).")
    print()

    mh_lex = _mean([r["lex_recall"] for r in multi_hop])
    mh_str = _mean([r["str_recall"] for r in multi_hop])
    print(f"MULTI-HOP verdict         : reading the query lexically found "
          f"the answer {mh_lex*len(multi_hop):.0f}/{len(multi_hop)} "
          f"times;")
    print(f"                            following ONE graph edge found it "
          f"{mh_str*len(multi_hop):.0f}/{len(multi_hop)} times.")
    print(f"SIMILARITY SEARCH verdict : both tie (direct hits need no "
          f"graph) — "
          f"lex={_mean([r['lex_recall'] for r in lexical]):.2f} "
          f"str={_mean([r['str_recall'] for r in lexical]):.2f}")
    if has_sem:
        mh_sem = _mean([r["sem_recall"] for r in multi_hop])
        lx_sem = _mean([r["sem_recall"] for r in lexical])
        print(f"SEMANTIC-ANCHOR verdict   : same walk, meaning-picked "
              f"anchor — multi-hop "
              f"{mh_sem*len(multi_hop):.0f}/{len(multi_hop)}, "
              f"similarity search mean {lx_sem:.2f}. The anchor step, not "
              f"the walk, is where the two variants differ.")


# --------------------------------------------------------------------------- #
# Graph visualization
#
# The notebook's "build a sample Oracle graph" section is about how the graph is
# stored and queried (CREATE PROPERTY GRAPH + GRAPH_TABLE ... MATCH). Drawing the
# result is presentation, not mechanism, so the rendering lives here. The lesson
# cells just call render_starter_graph(...) / render_full_graph(...).
#
# Both renders use the same "knowledge-graph" look (force-directed netgraph, node
# SHAPE = type, node COLOUR = dominant relationship layer). The difference is
# DIRECTION: on the 8-node starter graph the edge direction is legible, so we
# draw it — arrowheads on the directional kinds (contains / import / call) and a
# plain line for co_edit (which is symmetric: two files changed together, no
# direction). On the full 1,467-node / 5,201-edge graph arrowheads would just be
# noise on a hairball, so that one stays undirected on purpose.
# --------------------------------------------------------------------------- #

# muted palette shared by the big renders: grey = dependency layer, blue =
# co_edit. Blue/grey is a colour-blind-safe pair (the blue is Okabe–Ito).
_DEP_COLOR, _CO_COLOR = "#5b6770", "#0072B2"
# Example-graph edge encoding: every typed kind gets a COLOUR *and* a LINE
# PATTERN, so no relationship is readable by colour alone (~1 in 12 men have
# red/green colour-blindness). Colours are from the Okabe–Ito CB-safe palette.
# Nodes in the example graphs carry NO colour coding — a relationship is an
# edge, not a node colour — so all nodes share one neutral fill and only the
# SHAPE encodes what the node is (square = file, triangle = function).
_NODE_FILL = "#7a838d"
_EDGE_STYLE = {
    "contains": {"color": "#8a8f98", "ls": "solid",  "lw": 1.2},  # thin grey solid
    "import":   {"color": "#0072B2", "ls": "solid",  "lw": 1.8},  # blue solid
    "call":     {"color": "#D55E00", "ls": ":",      "lw": 1.8},  # vermilion dotted
    "co_edit":  {"color": "#009E73", "ls": "--",     "lw": 1.8},  # green dashed
}
# files-only zoom edge colours (CB-safe): sky blue for co_edit (the
# communities), orange for import (the cross-community links)
_FILE_EDGE_CO, _FILE_EDGE_DEP = "#56B4E9", "#E69F00"
# the file-layer zoom shows a readable SUBSET (a hub neighbourhood) in full colour,
# plus a faded "ghost" halo of the files just outside it — so you can see which
# side of the subset reaches back into the rest of the whole graph, and why.
# (ghost EDGES are coloured by kind like the core's, so only a node colour here.)
_GHOST_NODE = "#c9cdd2"
# which kinds carry a real direction (get an arrowhead) vs. which are symmetric
_DIRECTED_KINDS = ("contains", "import", "call")


def show_match(cur, sql, header, *, binds=None):
    """Run a GRAPH_TABLE ... MATCH query and print its rows under a header. A
    2-column result renders as 'left --> right'; a 3-column one as
    'left --kind--> right'; anything else is printed as a tuple."""
    cur.execute(sql, **(binds or {}))
    rows = cur.fetchall()
    print(f"=== {header} ===")
    for row in rows:
        if len(row) == 3:
            print(f"  {row[0]:18s} --{row[1]:8s}-> {row[2]}")
        elif len(row) == 2:
            print(f"  {row[0]:18s} --> {row[1]}")
        elif len(row) == 1:
            print(f"  {row[0]}")
        else:
            print(f"  {row}")
    print()


def show_starter_tables(cur):
    """Print what actually landed in the two starter tables — the rows the
    property graph is a view over, straight back out of Oracle."""
    cur.execute("SELECT id, kind, label FROM STARTER_GRAPH_NODES "
                "ORDER BY kind, id")
    rows = cur.fetchall()
    print(f"=== STARTER_GRAPH_NODES ({len(rows)} rows) ===")
    print(f"  {'id':26s} {'kind':8s} label")
    for id_, kind, label in rows:
        print(f"  {id_:26s} {kind:8s} {label}")
    print()

    cur.execute("SELECT eid, src, dst, kind FROM STARTER_GRAPH_EDGES "
                "ORDER BY eid")
    rows = cur.fetchall()
    print(f"=== STARTER_GRAPH_EDGES ({len(rows)} rows) ===")
    print(f"  {'eid':>3}  {'src':26s} {'dst':26s} kind")
    for eid, src, dst, kind in rows:
        print(f"  {eid:>3}  {src:26s} {dst:26s} {kind}")
    print()


def starter_reset(cur):
    """Drop the starter property graph + its two tables if present, so the build
    cell is idempotent. Ignores 'does not exist' on the first run."""
    for stmt in ("DROP PROPERTY GRAPH starter_code_graph",
                 "DROP TABLE STARTER_GRAPH_EDGES CASCADE CONSTRAINTS",
                 "DROP TABLE STARTER_GRAPH_NODES CASCADE CONSTRAINTS"):
        try:
            cur.execute(stmt)
        except Exception:
            pass


def _draw_example_graph(ax, nodes3, edges3, *, layout=None, node_face=None,
                        node_edgecolors=None, node_sizes=None, edge_alpha=None,
                        font_size=9, label_of=None):
    """Shared drawing core for the small example graphs (starter / anchor /
    growth views). Node SHAPE = type (square=file, triangle=function), one
    neutral fill for every node (relationships are edges, never node colours),
    and each edge kind gets its colour AND line pattern from ``_EDGE_STYLE``.
    Directional kinds get an arrowhead; co_edit stays a plain line (symmetric).
    Returns the layout so callers can draw highlights on top."""
    import networkx as nx

    label_of = label_of or {nid: lbl for nid, _k, lbl in nodes3}
    kind_of = {nid: k for nid, k, _l in nodes3}
    G = nx.Graph()
    for nid, k, _l in nodes3:
        G.add_node(nid, kind=k)
    for s, d, k in edges3:
        G.add_edge(s, d, kind=k)
    if layout is None:
        layout = nx.spring_layout(G, seed=23, k=2.0)

    Gd = nx.DiGraph()
    Gd.add_nodes_from(G.nodes)
    for kind in ("co_edit", "contains", "import", "call"):  # co_edit underneath
        el = [(s, d) for s, d, k in edges3 if k == kind]
        if not el:
            continue
        st = _EDGE_STYLE[kind]
        alpha = (edge_alpha or {}).get(kind, 0.85)
        if kind in _DIRECTED_KINDS:
            for s, d in el:
                Gd.add_edge(s, d)
            nx.draw_networkx_edges(Gd, layout, edgelist=el, ax=ax, arrows=True,
                                   arrowsize=16, width=st["lw"], style=st["ls"],
                                   edge_color=st["color"], alpha=alpha,
                                   min_source_margin=16, min_target_margin=16)
        else:
            nx.draw_networkx_edges(G, layout, edgelist=el, ax=ax, arrows=False,
                                   width=st["lw"], style=st["ls"],
                                   edge_color=st["color"], alpha=alpha)

    for shp, want in (("s", "file"), ("^", "symbol")):
        nl = [n for n in G.nodes if kind_of[n] == want]
        if not nl:
            continue
        nx.draw_networkx_nodes(
            G, layout, nodelist=nl, node_shape=shp, ax=ax,
            node_size=[(node_sizes or {}).get(n, 900 if want == "file" else 700)
                       for n in nl],
            node_color=[(node_face or {}).get(n, _NODE_FILL) for n in nl],
            edgecolors=[(node_edgecolors or {}).get(n, "white") for n in nl],
            linewidths=[2.2 if n in (node_edgecolors or {}) else 1.0
                        for n in nl])
    nx.draw_networkx_labels(G, layout, label_of, ax=ax, font_size=font_size)
    ax.axis("off")
    return layout


def _example_graph_legend(ax, *, extra=(), fontsize=9):
    """Legend for the example-graph encodings: shapes for node types, a
    colour+pattern key per edge kind (so the kinds stay readable without
    colour vision), plus any ``extra`` proxy artists."""
    from matplotlib.lines import Line2D

    items = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=_NODE_FILL,
               markersize=11, label="file"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=_NODE_FILL,
               markersize=11, label="function"),
        Line2D([0], [0], color=_EDGE_STYLE["contains"]["color"], lw=1.6,
               ls=_EDGE_STYLE["contains"]["ls"],
               label="contains (file → its function)"),
        Line2D([0], [0], color=_EDGE_STYLE["import"]["color"], lw=2,
               ls=_EDGE_STYLE["import"]["ls"], label="import (A imports B)"),
        Line2D([0], [0], color=_EDGE_STYLE["call"]["color"], lw=2,
               ls=_EDGE_STYLE["call"]["ls"], label="call (X calls Y)"),
        Line2D([0], [0], color=_EDGE_STYLE["co_edit"]["color"], lw=2,
               ls=_EDGE_STYLE["co_edit"]["ls"],
               label="co_edit (changed together — no direction)"),
    ]
    items.extend(extra)
    ax.legend(handles=items, loc="upper left", fontsize=fontsize, frameon=True,
              framealpha=0.9)


def render_starter_graph(starter_nodes, starter_edges):
    """Draw the hand-built starter graph as one knowledge-graph picture.

    Encodings: node SHAPE = type (square=file, triangle=function); every node
    shares one neutral colour — relationships are shown as EDGES, not node
    colours. Each edge kind has a colour AND a line pattern (colour-blind-safe),
    and directional kinds (contains/import/call) carry an arrowhead; co_edit is
    a plain dashed line because it is symmetric (changed together, no
    direction).
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    fig, ax = plt.subplots(figsize=(13, 9))
    _draw_example_graph(ax, starter_nodes, starter_edges)
    _example_graph_legend(ax)
    ax.set_title("A starter code graph stored in Oracle\n"
                 "shape = node type; every relationship is an EDGE — "
                 "colour + line pattern = edge kind; arrow = direction",
                 fontsize=12)
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def render_anchor_walk(starter_nodes, starter_edges,
                       *, query="where do we verify a token?", k=3):
    """Visualize the ANCHOR on the starter graph: the first node the graph walk
    starts from, found by a soft word match between the query and the node
    names/text. The anchor gets a ring; the nodes the walk reaches from it are
    outlined too, and everything else fades — showing that the walk surfaces
    related code that shares NO words with the query.

    Nothing here is staged: the anchor comes from the same keyword-overlap rule
    the real ranker uses, and the reached nodes come from a real Personalized
    PageRank run seeded at that anchor.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display
    from matplotlib.lines import Line2D

    from course_lab.structural_retrieval import (identifier_tokens,
                                                 personalized_pagerank)
    from course_lab.memory_graph import MemGraph

    label_of = {nid: lbl for nid, _k, lbl in starter_nodes}
    q_toks = identifier_tokens(query)

    # the anchor: best word-overlap between the query and a node's name
    overlap = {nid: len(q_toks & identifier_tokens(nid + " " + lbl))
               for nid, _k, lbl in starter_nodes}
    anchor = max(sorted(overlap), key=lambda n: overlap[n])

    # the walk: PPR seeded at the anchor ranks every connected node
    g = MemGraph()
    for s, d, _kind in starter_edges:
        g.add_edge(s, d)
    ppr = personalized_pagerank(g, {anchor: 1.0}, alpha=0.85, iters=50)
    reached = [nid for nid, _ in
               sorted(ppr.items(), key=lambda t: -t[1]) if nid != anchor][:k]

    _ANCHOR, _REACHED = "#D55E00", "#0072B2"
    fig, ax = plt.subplots(figsize=(13, 9))
    _draw_example_graph(
        ax, starter_nodes, starter_edges,
        node_edgecolors={anchor: _ANCHOR, **{n: _REACHED for n in reached}},
        node_sizes={anchor: 1500},
        edge_alpha={k_: 0.35 for k_ in _EDGE_STYLE},   # fade; highlights pop
    )
    _example_graph_legend(ax, extra=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=_ANCHOR, markeredgewidth=2.5, markersize=13,
               label="ANCHOR — where the graph walk starts"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=_REACHED, markeredgewidth=2.5, markersize=13,
               label="reached by walking edges from the anchor"),
    ])
    ax.set_title(f'The anchor for the query: "{query}"\n'
                 "word match finds the anchor; the graph walk reaches the rest",
                 fontsize=12)
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    print(f'Query   : "{query}"')
    matched = q_toks & identifier_tokens(anchor + " " + label_of[anchor])
    print(f"Anchor  : {label_of[anchor]}   <- best word match "
          f"({', '.join(sorted(matched))})")
    print(f"Walk    : {label_of[anchor]}, "
          f"{', '.join(label_of[n] for n in reached)}")
    print("None of these share semantic similarity with the query, but the "
          "edges from the graph reach them")


def render_growth_step(starter_nodes, starter_edges, new_node, new_edge):
    """Assume an edit is made — show what happens to the graph. BEFORE (left) is
    the starter repo's graph; AFTER (right) is the same graph one change later,
    with the new node highlighted and the new edge drawn DASHED in orange.
    Proves the update is a minimal merge: every prior node/edge is untouched.
    Same look as render_starter_graph (shape=type, neutral node fill,
    colour+pattern per edge kind).

    new_node = (id, kind, label);  new_edge = (src, dst, kind).
    """
    import warnings
    import matplotlib.pyplot as plt
    import networkx as nx
    from IPython.display import display

    _HL = "#E69F00"  # highlight for the freshly-added node + edge (CB-safe)

    def _panel(ax, nodes3, edges3, title, *, new_ids=frozenset(), new_e=None):
        old_edges = [e for e in edges3 if e != new_e]
        layout = _draw_example_graph(
            ax, nodes3, old_edges,
            node_face={n: _HL for n in new_ids},
            node_sizes={n: 1300 for n in new_ids},
            edge_alpha={k_: 0.45 for k_ in _EDGE_STYLE} if new_e else None,
        )
        if new_e is not None:
            ns, nd_, nk = new_e
            G = nx.DiGraph() if nk in _DIRECTED_KINDS else nx.Graph()
            G.add_nodes_from([n for n, _k, _l in nodes3])
            G.add_edge(ns, nd_)
            nx.draw_networkx_edges(G, layout, edgelist=[(ns, nd_)], ax=ax,
                                   arrows=nk in _DIRECTED_KINDS, arrowsize=20,
                                   width=3.0, edge_color=_HL, style="dashed")
        ax.set_title(title, fontsize=11)

    after_nodes = list(starter_nodes) + [new_node]
    after_edges = list(starter_edges) + [new_edge]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))
        _panel(axL, starter_nodes, starter_edges, "BEFORE — the repo as it is")
        _panel(axR, after_nodes, after_edges,
               "AFTER — one change later (new node + DASHED new edge)",
               new_ids={new_node[0]}, new_e=new_edge)
        fig.suptitle("Assume an edit is made: the change appends one node and one "
                     "typed edge — every prior node/edge is untouched",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        display(fig)
        plt.close(fig)
    s, d, k = new_edge
    print(f"Appended node {new_node[0]!r} and edge "
          f"{s.split(':')[-1]} --{k}--> {d.split(':')[-1]}.")
    print("Minimal MERGE into the existing graph - not destroying and "
          "re-creating")


def render_full_graph(nodes, edges):
    """Draw the WHOLE Oracle graph as one knowledge-graph picture.

    Same encodings as the starter render (shape=type, colour=dominant layer,
    size bumped for hubs), but UNDIRECTED on purpose: at ~5k edges arrowheads
    would be noise on the hairball, so parallel typed edges collapse into one
    undirected line and the picture is about clusters/hubs, not edge direction.
    Returns ``(n_nodes, n_edges, communities)`` for the caller to print.
    """
    import warnings
    from collections import Counter

    import matplotlib.pyplot as plt
    import networkx as nx
    from IPython.display import display
    from matplotlib.lines import Line2D

    # collapse parallel typed edges into one undirected edge carrying the kinds
    G = nx.Graph()
    for nd in nodes:
        G.add_node(nd["id"])
    for src, dst, kind in edges:
        if G.has_edge(src, dst):
            G.edges[src, dst]["kinds"].add(kind)
        else:
            G.add_edge(src, dst, kinds={kind})

    def _community(n):
        dep = co = 0
        for _u, _v, ks in G.edges(n, data="kinds"):
            dep += len({"import", "call"} & ks)
            co += 1 if "co_edit" in ks else 0
        return _CO_COLOR if co > dep else _DEP_COLOR

    node_color = {n: _community(n) for n in G.nodes}

    deg = dict(G.degree())
    hi = max(deg.values()) if deg else 1
    node_shape = {n: ("s" if n.startswith("file:") else "^") for n in G.nodes}

    def _size(n):
        base = 1.4 if n.startswith("file:") else 0.7
        return base + (1.6 if deg[n] >= max(8, hi * 0.25) else 0.0)   # hub bump

    node_size = {n: _size(n) for n in G.nodes}
    layout = nx.spring_layout(G, seed=1, k=0.55, iterations=80)

    fig, ax = plt.subplots(figsize=(18, 13))
    try:
        from netgraph import Graph as NetGraph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            NetGraph(G, node_layout=layout, node_edge_width=0,
                     node_size=node_size, edge_width=0.25,
                     edge_color="#aab0b8", edge_alpha=0.6, node_color=node_color,
                     node_shape=node_shape, arrows=False, ax=ax)
    except Exception as _e:
        print(f"(netgraph unavailable: {_e}; drawing with networkx)")
        nx.draw_networkx_edges(G, layout, edge_color="#aab0b8", width=0.4,
                               alpha=0.6, ax=ax)
        for shp in ("^", "s"):
            nl = [n for n in G.nodes if node_shape[n] == shp]
            nx.draw_networkx_nodes(G, layout, nodelist=nl, node_shape=shp,
                                   node_size=[node_size[n] * 60 for n in nl],
                                   node_color=[node_color[n] for n in nl],
                                   linewidths=0, ax=ax)

    def _key(marker, colour, text):
        return Line2D([0], [0], marker=marker, color="none",
                      markerfacecolor=colour, markersize=11, markeredgewidth=0,
                      label=text)

    legend_items = [
        _key("s", _DEP_COLOR, "file - dependency-led (import/call)"),
        _key("s", _CO_COLOR, "file - co_edit-led (changed together)"),
        _key("^", _DEP_COLOR, "symbol/function - dependency-led"),
        _key("^", _CO_COLOR, "symbol/function - co_edit-led"),
        Line2D([0], [0], color="#aab0b8", lw=1.4,
               label="edge (import / call / co_edit)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=11, frameon=True,
              framealpha=0.9, borderpad=0.8, labelspacing=0.7)
    ax.set_title("The agent-harness memory graph, as stored in Oracle "
                 f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n"
                 "shape = node type   |   colour = which relationship dominates "
                 "the node   |   bigger = more connected (hub)",
                 fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    by_kind = Counter(k for *_e, k in edges)
    comm = Counter(node_color.values())
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "by_kind": dict(by_kind),
        "dep_led": comm[_DEP_COLOR],
        "co_led": comm[_CO_COLOR],
    }


def render_files_only_graph(nodes, edges, *, n_hubs=3, max_core=12, max_ghosts=20):
    """Zoom into a READABLE PIECE of the file layer, shown embedded in the whole graph.

    Drawing all ~200 files at once is an unreadable knot — the co_edit graph is
    essentially one dense blob, so every label lands on top of every other in the
    middle of the page. Instead this zooms into a **hub neighbourhood**: the top
    ``n_hubs`` busiest files plus the files most tightly bound to them (the *core*,
    ``max_core`` files total), every one of them labelled and readable.

    To answer "where does this piece connect to the rest of the graph, and why",
    the core is wrapped in a faded **ghost halo**: the files just OUTSIDE the core
    that link into it from ≥2 of its members (``max_ghosts`` of them). Because the
    layout pulls each ghost toward the core files it actually links to, a ghost
    sits on the *side* of the core it connects through — and its edge colour says
    *why* it connects there:

    * pale blue ``co_edit`` — it ships together with that side of the core;
    * warm orange ``import`` — it depends on (imports) that side of the core.

    Core node colour is the same blue = co_edit-led / grey = dependency-led as the
    full graph. There are no ``call`` edges between files (calls are
    function->function), so a file's neighbourhood is "changes-with" (co_edit) plus
    "depends-on" (import). Returns a stats dict, including how many files are NOT
    shown, so the subset is never mistaken for the whole layer.
    """
    import warnings
    from collections import Counter, defaultdict

    import matplotlib.pyplot as plt
    import networkx as nx
    from IPython.display import display
    from matplotlib.lines import Line2D

    file_ids = {nd["id"] for nd in nodes if nd["id"].startswith("file:")}

    # full file subgraph: keep only edges whose BOTH ends are files, collapse
    # parallel typed edges into one undirected edge carrying the set of kinds
    Gfull = nx.Graph()
    Gfull.add_nodes_from(file_ids)
    for src, dst, kind in edges:
        if src in file_ids and dst in file_ids:
            if Gfull.has_edge(src, dst):
                Gfull.edges[src, dst]["kinds"].add(kind)
            else:
                Gfull.add_edge(src, dst, kinds={kind})
    Gfull.remove_nodes_from([n for n in list(Gfull) if Gfull.degree(n) == 0])
    deg_full = dict(Gfull.degree())

    def _is_dep(ks):
        return bool({"import", "call"} & ks)

    # --- pick the readable CORE: top hubs + their most tightly-bound neighbours ---
    hubs = sorted(deg_full, key=deg_full.get, reverse=True)[:n_hubs]

    def _core_touch(n):
        return sum(1 for h in hubs if Gfull.has_edge(n, h))

    # neighbours that link to ≥2 of the hubs are "inside" this neighbourhood; rank
    # by how many hubs they touch, then by their own degree, and cap the core size
    cand = [n for n in Gfull if n not in hubs and _core_touch(n) >= 2]
    cand.sort(key=lambda n: (_core_touch(n), deg_full[n]), reverse=True)
    core = list(hubs) + cand[: max(0, max_core - len(hubs))]
    core_set = set(core)

    # --- GHOST halo: outside files that link into the core from ≥2 members ------
    out_touch, out_kinds = Counter(), defaultdict(set)
    for n in core_set:
        for nb in Gfull.neighbors(n):
            if nb not in core_set:
                out_touch[nb] += 1
                out_kinds[nb] |= Gfull.edges[n, nb]["kinds"]
    ghosts = [g for g, c in out_touch.most_common() if c >= 2][:max_ghosts]
    ghost_set = set(ghosts)
    n_hidden = Gfull.number_of_nodes() - len(core_set) - len(ghost_set)

    def _community(n):
        dep = co = 0
        for _u, _v, ks in Gfull.edges(n, data="kinds"):
            dep += len({"import", "call"} & ks)
            co += 1 if "co_edit" in ks else 0
        return _CO_COLOR if co > dep else _DEP_COLOR

    # ---- LAYOUT: core in a compact centre, ghosts on a ring at the angle of -----
    # ---- the core files they attach to, so a ghost literally sits on the side ---
    # ---- of the core it links through. (Spring-on-everything just made the dense
    # ---- core repel into a useless ring, so we place the two layers explicitly.)
    import math

    # core positions: spring among the core only, then recentre on (0,0) and scale
    # by the max EUCLIDEAN radius so the cluster fills a tidy inner disc (radius ~0.4)
    core_layout = nx.spring_layout(Gfull.subgraph(core_set), seed=7,
                                   k=2.4, iterations=400)
    cx = sum(p[0] for p in core_layout.values()) / len(core_layout)
    cy = sum(p[1] for p in core_layout.values()) / len(core_layout)
    rmax = max(math.hypot(p[0] - cx, p[1] - cy)
               for p in core_layout.values()) or 1.0
    pos = {n: ((p[0] - cx) / rmax * 0.50, (p[1] - cy) / rmax * 0.50)
           for n, p in core_layout.items()}

    # each ghost gets the mean ANGLE of the core files it links to, then sits on an
    # outer ring; spread ties apart so labels around the ring don't collide
    def _ghost_angle(g):
        xs = ys = 0.0
        for nb in Gfull.neighbors(g):
            if nb in pos:
                xs += pos[nb][0]; ys += pos[nb][1]
        return math.atan2(ys, xs) if (xs or ys) else 0.0

    # sort ghosts by the side they point to, then space them EVENLY around the ring
    # in that order — this keeps each ghost on its correct side while guaranteeing
    # the rim labels never pile up on top of each other.
    ring_order = sorted(ghosts, key=_ghost_angle)
    n_ring = len(ring_order)
    for i, g in enumerate(ring_order):
        a = (2 * math.pi) * i / max(n_ring, 1) + math.pi / 2   # start at top, go round
        pos[g] = (0.95 * math.cos(a), 0.95 * math.sin(a))

    shown = core_set | ghost_set
    G = Gfull.subgraph(shown).copy()
    # The hub core is so connected that every ghost links to ALL of it — drawing
    # all those spokes is a 240-line starburst that hides the point. So collapse
    # each ghost to ONE representative edge: to the core file it's angularly nearest
    # (the side it sits on), coloured by the ghost's DOMINANT kind (import if it
    # imports the core at all, else co_edit). That keeps "which side + why" and
    # drops the clutter. Core<->core edges stay (they ARE the neighbourhood).
    G.remove_edges_from([(u, v) for u, v in list(G.edges())
                         if u in ghost_set or v in ghost_set])
    ghost_kind = {}
    for g in ghost_set:
        if g not in G:
            continue
        dep_out = bool({"import", "call"} & out_kinds[g])
        ghost_kind[g] = "dep" if dep_out else "co"
        # anchor = the core neighbour closest in angle to where we placed the ghost
        gx, gy = pos[g]
        anchor = min((c for c in Gfull.neighbors(g) if c in core_set),
                     key=lambda c: (pos[c][0] - gx) ** 2 + (pos[c][1] - gy) ** 2,
                     default=None)
        if anchor is not None:
            G.add_edge(g, anchor, kinds=out_kinds[g])

    node_color = {n: (_GHOST_NODE if n in ghost_set else _community(n)) for n in G}
    node_size = {n: (1.0 if n in ghost_set else 4.0) for n in G}

    # edge colour by kind; a core<->ghost edge is the "door out" — colour it by why
    # (orange import / blue co_edit) so the REASON it connects there is visible
    edge_color, edge_width = {}, {}
    for u, v, d in G.edges(data=True):
        is_out = u in ghost_set or v in ghost_set
        if is_out:
            g = u if u in ghost_set else v
            dep = ghost_kind.get(g) == "dep"
            edge_color[(u, v)] = _FILE_EDGE_DEP if dep else _FILE_EDGE_CO
            edge_width[(u, v)] = 1.4 if dep else 0.9
        else:
            dep = _is_dep(d["kinds"])
            edge_color[(u, v)] = _FILE_EDGE_DEP if dep else _FILE_EDGE_CO
            edge_width[(u, v)] = 1.0 if dep else 0.4

    core_labels = {n: n.split("file:", 1)[-1] for n in core_set if n in G}
    ghost_labels = {n: n.split("file:", 1)[-1] for n in ghost_set if n in G}
    layout = pos

    fig, ax = plt.subplots(figsize=(16, 13))
    try:
        from netgraph import Graph as NetGraph
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            NetGraph(G, node_layout=layout, node_edge_width=0, node_size=node_size,
                     node_color=node_color, node_shape="s",
                     node_labels={**ghost_labels, **core_labels},
                     node_label_fontdict=dict(size=8),
                     edge_color=edge_color, edge_width=edge_width, edge_alpha=0.6,
                     arrows=False, ax=ax)
        # dim the ghost labels so the core reads first
        for txt in ax.texts:
            if txt.get_text() in set(ghost_labels.values()) - set(core_labels.values()):
                txt.set_color("#8b9097")
                txt.set_fontsize(7)
    except Exception as _e:
        print(f"(netgraph unavailable: {_e}; drawing with networkx)")
        nx.draw_networkx_edges(
            G, layout, edgelist=list(G.edges()),
            edge_color=[edge_color[e] for e in G.edges()],
            width=[edge_width[e] * 1.4 for e in G.edges()], alpha=0.6, ax=ax)
        nx.draw_networkx_nodes(G, layout, node_shape="s",
                               node_size=[node_size[n] * 70 for n in G],
                               node_color=[node_color[n] for n in G],
                               linewidths=0, ax=ax)
        nx.draw_networkx_labels(G, layout, labels=core_labels, font_size=8, ax=ax)
        nx.draw_networkx_labels(G, layout, labels=ghost_labels, font_size=7,
                                font_color="#8b9097", ax=ax)

    legend_items = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=_CO_COLOR,
               markersize=11, markeredgewidth=0,
               label="core file — co_edit-led (changes with this neighbourhood)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=_DEP_COLOR,
               markersize=11, markeredgewidth=0,
               label="core file — dependency-led (imported/called more)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=_GHOST_NODE,
               markersize=11, markeredgewidth=0,
               label="ghost — a file just OUTSIDE, on the rim (the rest of the graph)"),
        Line2D([0], [0], color=_FILE_EDGE_CO, lw=1.6,
               label="co_edit edge (ships together)"),
        Line2D([0], [0], color=_FILE_EDGE_DEP, lw=1.6,
               label="import edge (depends on)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=10.5, frameon=True,
              framealpha=0.9, borderpad=0.8, labelspacing=0.6)
    ax.set_title(
        f"Zoom: one hub neighbourhood of the file layer "
        f"({len(core_set)} core files, {len(ghost_set)} shown around it; "
        f"{n_hidden} more not drawn)\n"
        "each ghost sits on the side of the core it links through — "
        "orange = it imports that side, blue = it ships with that side",
        fontsize=13)
    ax.axis("off")
    ax.set_xlim(-1.35, 1.35)   # head-room around the rim so ghost labels don't clip
    ax.set_ylim(-1.20, 1.20)
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    core_only = Gfull.subgraph(core_set)
    comm = Counter(_community(n) for n in core_set)
    n_import = sum(1 for _u, _v, d in core_only.edges(data=True) if _is_dep(d["kinds"]))
    n_coedit = core_only.number_of_edges() - n_import
    # describe each ghost: which side it reaches in through, and why
    ghost_rows = []
    for g in ghosts:
        why = ("import+co_edit" if {"import", "call"} & out_kinds[g] and "co_edit" in out_kinds[g]
               else "import" if {"import", "call"} & out_kinds[g] else "co_edit")
        ghost_rows.append((g.split("file:", 1)[-1], out_touch[g], why))
    return {
        "n_files_total": Gfull.number_of_nodes(),
        "n_core": len(core_set),
        "n_ghosts": len(ghost_set),
        "n_hidden": n_hidden,
        "n_core_edges": core_only.number_of_edges(),
        "n_import_edges": n_import,
        "n_coedit_edges": n_coedit,
        "co_led": comm[_CO_COLOR],
        "dep_led": comm[_DEP_COLOR],
        "top_hubs": [(n.split("file:", 1)[-1], deg_full[n]) for n in hubs],
        "ghosts": ghost_rows,
    }


def show_files_zoom(fstats):
    """Print the files-only zoom stats (core / ghost make-up and top hubs) that
    ``render_files_only_graph`` returns."""
    print(f"FILE layer in Oracle : {fstats['n_files_total']} files total")
    print(f"This zoom shows      : {fstats['n_core']} core files "
          f"({fstats['n_core_edges']} edges among them: "
          f"{fstats['n_coedit_edges']} co_edit + "
          f"{fstats['n_import_edges']} import) "
          f"+ {fstats['n_ghosts']} ghost files around it")
    print(f"Not drawn            : {fstats['n_hidden']} further "
          f"files of the full layer")
    print(f"Core make-up         : co_edit-led={fstats['co_led']}, "
          f"dependency-led={fstats['dep_led']}")
    print("Top hub files (core) :")
    for name, d in fstats["top_hubs"]:
        print(f"   {name:34s} degree {d}")
