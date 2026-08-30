"""Notebook helpers for graphify-verification: load results, render the graph
(anchors vs graph-reach), and draw the control-vs-treatment comparison.

Kept separate from ``course_lab.m3_lab`` (which renders the Module 3 teaching
graphs) so the experiment's presentation code is self-contained.
"""
from __future__ import annotations

import json
from pathlib import Path

from course_lab import paths


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_results(task: str | None = None) -> dict:
    """Load committed graphify results. ``task`` (e.g. 'django_cache') selects a
    namespaced target; ``None`` is the default httpie run."""
    p = paths.graphify_verification_results_json(task)
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; run `python lab.py graphify-verify report` first.")
    return json.loads(p.read_text())


def load_suite_summary() -> dict:
    """Load the 10-task suite summary (per-task deltas + pooled win/tie/loss)."""
    p = paths.graphify_verification_suite_summary_json()
    if not p.exists():
        raise FileNotFoundError(
            f"missing {p}; run graphify_verification/scripts/run_suite.py first.")
    return json.loads(p.read_text())


_SUITE_METRICS = [
    ("mean_duration_ms", "total time on task"),
    ("mean_tool_calls_to_first_correct_edit", "steps to first correct edit"),
    ("mean_total_tokens", "total tokens"),
    ("mean_tool_calls", "tool calls"),
    ("mean_cost_usd", "cost (usd)"),
    ("mean_recall", "gold recall"),
]


# --------------------------------------------------------------------------- #
# traffic-light labelling — one shared rule so learners never decode a sign
# --------------------------------------------------------------------------- #
# EVERY comparison in this file is normalized so a positive % ALWAYS means "graph
# better". We attach a colour so the direction is read at a glance, not inferred:
#   🟢 graph clearly better · 🟠 roughly a tie · 🔴 graph worse
_TIE_BAND = 1.0  # |improvement%| <= this reads as a tie


def light(improvement_pct: float | None) -> str:
    """Traffic-light emoji for an already-signed improvement % (positive = graph
    better). Use this on every per-metric cell so learners never have to work out
    whether +% or -% is the win — the colour says it."""
    if improvement_pct is None:
        return "⚪"
    if improvement_pct > _TIE_BAND:
        return "🟢"
    if improvement_pct < -_TIE_BAND:
        return "🔴"
    return "🟠"


def light_pct(improvement_pct: float | None, decimals: int = 0) -> str:
    """`🟢 +8%` — a coloured, signed improvement string for a table cell."""
    if improvement_pct is None:
        return "⚪ –"
    return f"{light(improvement_pct)} {improvement_pct:+.{decimals}f}%"


def suite_per_task_table_md() -> str:
    """Markdown per-task table (sorted by total-time improvement). Every cell is
    traffic-lit — 🟢 graph better · 🟠 ~tie · 🔴 graph worse — so the win direction
    is never left for the learner to infer from the sign."""
    s = load_suite_summary()
    rows = sorted(s["per_task"],
                  key=lambda r: -r["metrics"].get("mean_duration_ms", {}).get("pct_improvement", 0))
    head = "| Task | Label | " + " | ".join(lbl for _, lbl in _SUITE_METRICS) + " |"
    sep = "|" + "---|" * (len(_SUITE_METRICS) + 2)
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _ in _SUITE_METRICS:
            v = r["metrics"].get(key, {}).get("pct_improvement")
            cells.append(light_pct(v))
        lines.append(f"| {r['task']} | {r['favorability']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("🟢 graph better · 🟠 ~tie · 🔴 graph worse  (every % is signed so "
                 "positive = graph better)")
    return "\n".join(lines)


def suite_pooled_table_md() -> str:
    """Markdown pooled table across the 10 tasks. The headline is the **median**
    per-task improvement — the *typical* task — because two heavy-tailed
    counter-cases (gold files in disconnected subsystems) misrepresent the middle.
    The per-task spread chart still shows every task, losers included."""
    s = load_suite_summary()
    lines = ["| Metric | Median (typical task) | Win / tie / loss |",
             "|---|:--|:--:|"]
    for key, _ in _SUITE_METRICS:
        p = s["pooled"].get(key)
        if not p:
            continue
        # back-compat: older summaries carried only pooled_mean_pct
        med = p.get("pooled_median_pct", p.get("pooled_mean_pct"))
        lines.append(f"| {p['label']} | {light(med)} **{med:+.1f}%** | "
                     f"{p['wins']} / {p['ties']} / {p['losses']} |")
    lines.append("")
    lines.append("🟢 graph better · 🟠 ~tie · 🔴 graph worse  ·  "
                 "median = the *typical* task (robust to the two disconnected-gold "
                 "counter-cases in the spread chart above).")
    return "\n".join(lines)


def suite_pooled_pct(key: str):
    """Pooled 10-task **median** improvement % for one metric key (e.g.
    'mean_duration_ms'), plus its win/total tally. Returns (pct, wins, n) or
    (None, None, None) if absent. The median is the robust central number shown
    NEXT TO any single hero-task figure — the typical task, not the mean the two
    counter-cases distort."""
    try:
        p = load_suite_summary()["pooled"].get(key)
    except FileNotFoundError:
        return (None, None, None)
    if not p:
        return (None, None, None)
    med = p.get("pooled_median_pct", p.get("pooled_mean_pct"))
    return (med, p["wins"], p["n_tasks"])


def render_suite_bars(ax=None):
    """Grouped bar: per-task total-time improvement, one bar per task, coloured
    by the value it plots — red slower, amber ~tie (0-4%), green faster. Shows
    the honest spread — every task, winners and losers. The dashed line is the
    **median** (the typical task); the big negatives are the documented
    counter-case where the gold files sit in disconnected subsystems the graph
    can't reach, not a failure of the method on tasks it's meant for."""
    import matplotlib.pyplot as plt
    s = load_suite_summary()
    rows = sorted(s["per_task"],
                  key=lambda r: r["metrics"].get("mean_duration_ms", {}).get("pct_improvement", 0))
    labels = [r["task"].replace("django_", "").replace("httpie_", "httpie ") for r in rows]
    vals = [r["metrics"].get("mean_duration_ms", {}).get("pct_improvement", 0) for r in rows]
    # colour reads straight off the x axis: slower / ~tie / faster
    colors = ["#c0392b" if v < 0 else "#d9a441" if v < 4 else "#2e9e5b"
              for v in vals]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(labels)), vals, color=colors)
    ax.axvline(0, color="#333", lw=0.9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("total-time improvement %  (positive = graph faster)")
    pool = s["pooled"].get("mean_duration_ms", {})
    median = pool.get("pooled_median_pct", pool.get("pooled_mean_pct", 0))
    mean = pool.get("pooled_mean_pct", 0)
    wins = pool.get("wins")
    # median reference line — the typical-task headline
    ax.axvline(median, color="#2e9e5b", lw=1.4, ls="--", alpha=0.9)
    ax.text(median, len(labels) - 0.4, f" median {median:+.1f}%",
            color="#1d6e40", fontsize=8, fontweight="bold", va="top")
    # annotate the worst tasks as the known counter-case. The note sits left of
    # the bar end, so the x-limits below reserve room for it — without that it
    # runs off the axes and lands on top of the task's y-tick label.
    annotated = False
    for i, v in enumerate(vals):
        if v <= -20:
            ax.text(v - 1.0, i, "gold files not clustered ", ha="right", va="center",
                    fontsize=7, color="#7a2e24", style="italic")
            annotated = True
    lo, hi = min(vals + [0]), max(vals + [0])
    ax.set_xlim(lo - (20 if annotated else 5), hi + 5)
    title = (f"Per-task time-on-task: graph vs bare repo  "
             f"(median {median:+.1f}%, {wins}/{s['n_tasks']} tasks faster")
    title += f"; mean {mean:+.1f}%)" if mean is not None else ")"
    ax.set_title(title, fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    return vals


def load_code_graph() -> dict:
    p = paths.graphify_verification_code_graph_json()
    return json.loads(p.read_text())


def load_structure_map() -> dict:
    p = paths.graphify_verification_structure_map_json()
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# graph viz — the httpie code graph, with anchors / graph-reach / gold marked
# --------------------------------------------------------------------------- #

# muted palette consistent with Module 3's graph cells
_C_ANCHOR = "#c2774f"     # warm orange — keyword anchors
_C_REACH = "#9cc3f0"      # pale blue — graph-reached
_C_GOLD_RING = "#d64545"  # red ring — gold files
_C_OTHER = "#d7dade"      # pale grey — everything else
_C_EDGE = "#b7bcc2"


def render_structure_map_graph(ax=None):
    """Draw the FILE graph around the structure map: keyword anchors (orange),
    graph-reached dependencies (blue), gold files ringed red, on the real httpie
    import/co_edit file graph. Shows *why* graphify reaches the gold files a
    keyword search misses — the blue nodes are reached only through edges.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    cg = load_code_graph()
    sm = load_structure_map()
    res = None
    try:
        res = load_results()
    except FileNotFoundError:
        pass
    gold = set(sm.get("gold_files", []))
    anchors = set(sm.get("anchors", []))
    reach = set(sm.get("graph_reach", []))

    # build a FILE-only graph (import + co_edit between files)
    G = nx.Graph()
    file_ids = {n["id"] for n in cg["nodes"] if n["id"].startswith("file:")}
    for fid in file_ids:
        G.add_node(fid[len("file:"):])
    for e in cg["edges"]:
        if e["kind"] in ("import", "co_edit") and e["src"] in file_ids and e["dst"] in file_ids:
            u, v = e["src"][len("file:"):], e["dst"][len("file:"):]
            if u != v:
                G.add_edge(u, v)

    # keep the structure-map files + their immediate neighbors so the picture is readable
    keep = anchors | reach
    nbrs = set()
    for f in keep:
        if f in G:
            nbrs.update(G.neighbors(f))
    sub_nodes = (keep | nbrs) & set(G.nodes)
    H = G.subgraph(sub_nodes).copy()

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 9))
    pos = nx.spring_layout(H, seed=23, k=1.4, iterations=120)

    def _color(f):
        if f in anchors:
            return _C_ANCHOR
        if f in reach:
            return _C_REACH
        return _C_OTHER

    def _size(f):
        return 900 if (f in gold) else (520 if f in (anchors | reach) else 180)

    node_colors = [_color(f) for f in H.nodes]
    node_sizes = [_size(f) for f in H.nodes]
    edgecolors = [_C_GOLD_RING if f in gold else "#ffffff" for f in H.nodes]
    linewidths = [3.0 if f in gold else 0.6 for f in H.nodes]

    nx.draw_networkx_edges(H, pos, ax=ax, edge_color=_C_EDGE, width=0.6, alpha=0.6)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                           edgecolors=edgecolors, linewidths=linewidths)
    # label only the structure-map + gold files (avoid clutter)
    labels = {f: f for f in H.nodes if f in (anchors | reach | gold)}
    nx.draw_networkx_labels(H, pos, labels=labels, ax=ax, font_size=8)

    ax.set_axis_off()
    ax.set_title(
        "httpie code graph — structure map (lexical anchor + graph reach) for "
        "“regulate top-level JSON arrays”\n"
        "orange = keyword anchor · blue = graph-reached dependency · "
        "red ring = gold file (must edit)",
        fontsize=11)

    # proxy legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_ANCHOR,
               markersize=12, label="keyword anchor"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_C_REACH,
               markersize=12, label="graph-reached (missed by keywords)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ffffff",
               markeredgecolor=_C_GOLD_RING, markeredgewidth=3, markersize=12,
               label="gold file"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    anchors_hit = sm.get("anchors_hit", [])
    graph_hit = sm.get("graph_hit", [])
    return {"n_subgraph_nodes": H.number_of_nodes(),
            "gold_recall": sm.get("gold_recall"),
            "anchors_hit": anchors_hit, "graph_hit": graph_hit}


# --------------------------------------------------------------------------- #
# comparison bars — control vs treatment
# --------------------------------------------------------------------------- #

def render_comparison(ax=None):
    """Grouped bars: control vs treatment on the headline metrics, normalized so
    'lower is better' metrics read consistently. Returns the delta dict."""
    import matplotlib.pyplot as plt
    import numpy as np

    res = load_results()
    agg = res["aggregate"]
    c, t = agg["control"], agg["treatment"]

    metrics = [
        ("acceptance\npass rate", "accept_pass_rate", False),
        ("gold recall", "mean_recall", False),
        ("total tokens", "mean_total_tokens", True),
        ("tool calls", "mean_tool_calls", True),
        ("calls to first\ncorrect edit", "mean_tool_calls_to_first_correct_edit", True),
    ]
    labels, ctrl_vals, trt_vals = [], [], []
    for label, key, _ in metrics:
        cv, tv = c.get(key), t.get(key)
        if cv is None or tv is None:
            continue
        # normalize each metric to control = 1.0 so bars are comparable
        base = cv if cv else 1.0
        labels.append(label)
        ctrl_vals.append(1.0)
        trt_vals.append(tv / base if base else 0.0)

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, ctrl_vals, w, label="control (bare repo)", color="#9aa0a6")
    ax.bar(x + w / 2, trt_vals, w, label="treatment (+ structure map)", color="#4f86c6")
    ax.axhline(1.0, color="#444", lw=0.7, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("relative to control (=1.0)")
    ax.set_title("structure-map (lexical anchor + graph reach) — control vs treatment "
                 f"(n={res['n_runs_per_arm']}/arm, {res['model']})", fontsize=12)
    ax.legend()
    for i, v in enumerate(trt_vals):
        ax.text(x[i] + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    return agg["deltas"]


_BENCH_ROWS = [
    ("Acceptance pass rate", "accept_pass_rate", False, "{:.0%}"),
    ("Gold-file recall", "mean_recall", False, "{:.2f}"),
    ("Gold-file precision", "mean_precision", False, "{:.2f}"),
    ("Total tokens", "mean_total_tokens", True, "{:,.0f}"),
    ("Tool calls", "mean_tool_calls", True, "{:.1f}"),
    ("Calls to 1st correct edit", "mean_tool_calls_to_first_correct_edit", True, "{:.1f}"),
    ("Cost (USD)", "mean_cost_usd", True, "${:.3f}"),
    ("Wall-clock (s)", "mean_duration_ms", True, "{:.0f}"),
]


def benchmark_table_md() -> str:
    """A full benchmark report as a markdown table: per-metric control vs
    treatment absolute values, improvement %, and whether treatment won."""
    res = load_results()
    agg = res["aggregate"]
    c, t = agg["control"], agg["treatment"]
    lines = [
        f"### Benchmark report — {res['n_runs_per_arm']} runs/arm, model `{res['model']}`",
        "",
        f"Task: {res['task']['repo'].split('/')[-1].replace('.git','')} — "
        f"implement *{res['task']['gold_commit']}* "
        f"({len(res['task']['gold_files'])} gold files). "
        f"Control = bare repo; treatment = the Code Knowledge Graph structure map "
        f"(lexical anchor + graph reach).",
        "",
        "| Metric | Control | Treatment | Improvement (🟢 better · 🟠 tie · 🔴 worse) |",
        "|---|---:|---:|:--|",
    ]
    for label, key, lower_better, fmt in _BENCH_ROWS:
        cv, tv = c.get(key), t.get(key)
        if cv is None or tv is None:
            continue
        d_ms = "duration_ms" in key
        cvd = cv / 1000 if d_ms else cv
        tvd = tv / 1000 if d_ms else tv
        cstr = fmt.format(cvd)
        tstr = fmt.format(tvd)
        if cv:
            pct = (tv - cv) / cv * 100.0
            # phrase improvement so a positive % always means "better"
            disp = -pct if lower_better else pct
            imp = light_pct(disp)
        elif tv != cv:
            # zero baseline (e.g. control never passed): show the absolute gain
            improved = (tv > 0) if not lower_better else (tv < cv)
            gain = f"+{tstr} (from 0)" if not lower_better else "improved"
            imp = f"{'🟢' if improved else '🔴'} {gain}"
        else:
            imp = "🟠 —"
        lines.append(f"| {label} | {cstr} | {tstr} | {imp} |")
    return "\n".join(lines)


# metric key -> (row label, control fmt, lower_is_better)
_HERO_ROWS = [
    ("total time on task", "mean_duration_ms", lambda v: f"{v/1000:.0f}s", True),
    ("steps to first correct edit", "mean_tool_calls_to_first_correct_edit", lambda v: f"{v:.1f}", True),
    ("total tokens", "mean_total_tokens", lambda v: f"{v:,.0f}", True),
    ("cost (usd)", "mean_cost_usd", lambda v: f"${v:.2f}", True),
    ("gold recall", "mean_recall", lambda v: f"{v:.2f}", False),
]


def study_header(res: dict) -> str:
    """One-line description of a graphify run (repo, gold commit+files, agent) for
    the practical notebook's setup cell — replaces a stack of per-field prints."""
    task = res["task"]
    return (f"Target repo : {task['repo']}\n"
            f"Task        : {task['gold_commit'][:12]} (gold files: {task['gold_files']})\n"
            f"Agent       : {res['model']} via {res.get('driver', 'claude')} "
            f"| {res['n_runs_per_arm']} runs/arm")


def hero_vs_typical_table_md(task: str = "django_cache") -> str:
    """Traffic-lit markdown table of the 10-task MEDIAN ('typical task') per-metric
    improvement. Every % is signed (positive = graph better) and coloured 🟢/🟠/🔴.
    Returns '' if the task's results aren't in this checkout."""
    try:
        load_results(task)
    except FileNotFoundError:
        return ""

    def typical(key):
        med, w, n = suite_pooled_pct(key)   # median across the 10 tasks
        return f"{light_pct(med)} ({w}/{n})" if med is not None else "⚪ –"

    head = ("**django — Claude Code (Haiku).** The **typical task (median across all "
            "10)** — 🟢 graph better · 🟠 ~tie · 🔴 graph worse, every % signed so "
            "positive = graph better (N/10 = tasks won):")
    lines = [head, "",
             "| metric | **typical task (median across 10)** |",
             "|---|:--|"]
    for label, key, _fmt, _lower in _HERO_ROWS:
        lines.append(f"| {label} | {typical(key)} |")
    return "\n".join(lines)


def render_per_run_distributions(axes=None):
    """Per-run scatter (control vs treatment) for the efficiency metrics, so the
    spread across runs is visible — not just the means. Shows whether the win is
    consistent or driven by an outlier."""
    import matplotlib.pyplot as plt
    import numpy as np

    res = load_results()
    runs = res["runs"]
    metrics = [
        ("Total tokens", "total_tokens", True),
        ("Tool calls", "n_tool_calls", True),
        ("Calls to 1st correct edit", "tool_calls_to_first_correct_edit", True),
        ("Gold recall", "recall", False),
    ]
    if axes is None:
        _, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.4))
    arms = ["control", "treatment"]
    colors = {"control": "#9aa0a6", "treatment": "#4f86c6"}
    for ax, (label, key, lower_better) in zip(axes, metrics):
        for xi, arm in enumerate(arms):
            vals = [r[key] for r in runs if r["arm"] == arm and r.get(key) is not None]
            jitter = (np.random.RandomState(7).rand(len(vals)) - 0.5) * 0.12
            ax.scatter([xi + j for j in jitter], vals, s=70, alpha=0.8,
                       color=colors[arm], edgecolor="white", linewidth=0.8, zorder=3)
            if vals:
                ax.hlines(np.mean(vals), xi - 0.2, xi + 0.2, color="#333", lw=2, zorder=4)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        arrow = "↓ better" if lower_better else "↑ better"
        ax.set_ylabel(arrow, fontsize=8)
    import matplotlib.pyplot as _plt
    _plt.suptitle(f"Per-run distributions (n={res['n_runs_per_arm']}/arm) — "
                  "black bar = mean", fontsize=12, y=1.02)
    _plt.tight_layout()
    return None


def render_improvement_bars(ax=None, task: str | None = None):
    """A single 'how much did graphify improve each metric' bar chart, where
    every bar is signed so positive = treatment better (the headline graph).

    ``task`` selects the run: ``None`` = the httpie/sonnet default; e.g.
    ``'django_cache'`` = the django/haiku hero task."""
    import matplotlib.pyplot as plt

    res = load_results(task)
    agg = res["aggregate"]
    c, t = agg["control"], agg["treatment"]
    # Full multi-dimensional comparison (the chapter's concluding graph). Every
    # bar is a % improvement of treatment (graph map) over control (bare repo),
    # signed so positive = better. Axes span the whole efficiency picture:
    # quality (recall), token economy, tool calls, time-to-first-correct-edit,
    # total time on task, and total effective efficiency (cost proxy).
    rows = [
        ("Gold\nrecall", "mean_recall", False),
        ("Fewer\ntokens", "mean_total_tokens", True),
        ("Fewer\ntool calls", "mean_tool_calls", True),
        ("Faster to 1st\ncorrect edit", "mean_tool_calls_to_first_correct_edit", True),
        ("Less total\ntime on task", "mean_duration_ms", True),
        ("Total effective\nefficiency", "mean_cost_usd", True),
    ]
    labels, vals = [], []
    for label, key, lower_better in rows:
        cv, tv = c.get(key), t.get(key)
        if cv in (None, 0) or tv is None:
            continue
        pct = (tv - cv) / cv * 100.0
        disp = -pct if lower_better else pct   # positive = better
        labels.append(label)
        vals.append(disp)
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#2e9e5b" if v >= 0 else "#c0392b" for v in vals]
    bars = ax.bar(range(len(labels)), vals, color=colors)
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("improvement % (positive = structure map better)")
    ax.set_title("improvement from the Code Knowledge Graph structure map "
                 f"(n={res['n_runs_per_arm']}/arm, {res['model']})", fontsize=12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2,
                v + (1.5 if v >= 0 else -3.0), f"{v:+.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    return vals


def render_sonnet_vs_haiku():
    """Side-by-side improvement bars for the two runs we have — the smaller-repo
    httpie/Sonnet run (left) and the bigger-repo django/Haiku hero task (right) —
    so the two agents/repos can be compared on the same axes. Reads committed
    results; renders whichever arms are present and skips the other cleanly."""
    import matplotlib.pyplot as plt
    panels = [(None, "httpie/cli · Sonnet\n(smaller repo)"),
              ("django_cache", "django/django · Haiku\n(bigger repo)")]
    present = []
    for task, title in panels:
        try:
            load_results(task); present.append((task, title))
        except FileNotFoundError:
            pass
    if not present:
        print("no graphify results committed in this checkout.")
        return
    fig, axes = plt.subplots(1, len(present), figsize=(6.2 * len(present), 5.2),
                             sharey=True)
    if len(present) == 1:
        axes = [axes]
    for ax, (task, title) in zip(axes, present):
        render_improvement_bars(ax=ax, task=task)
        ax.set_title(title, fontsize=11)
    fig.suptitle("Structure-map improvement — same experiment, two agents / repo sizes "
                 "(positive = graph better)", fontsize=12, y=1.02)
    fig.tight_layout()


def summarize_text() -> str:
    """One-paragraph plain-text summary of the headline result for a markdown cell."""
    res = load_results()
    d = res["aggregate"]["deltas"]

    def _pct(key):
        v = d.get(key)
        return None if not v else v["pct_change"]

    lines = ["**Result.**"]
    tok = _pct("mean_total_tokens")
    tc = _pct("mean_tool_calls")
    ttf = _pct("mean_tool_calls_to_first_correct_edit")
    rec = _pct("mean_recall")
    dur = _pct("mean_duration_ms")
    cost = _pct("mean_cost_usd")
    c, t = res["aggregate"]["control"], res["aggregate"]["treatment"]
    lines.append(
        f"Across {res['n_runs_per_arm']} runs per arm ({res['model']}), the treatment "
        f"(the Code Knowledge Graph structure map — lexical anchor + graph reach) "
        f"vs control (bare repo):")
    if rec is not None:
        lines.append(f"- gold-file recall (*quality*) {c['mean_recall']} → {t['mean_recall']} ({rec:+.0f}%)")
    if dur is not None:
        lines.append(f"- **total time on task** {c['mean_duration_ms']/1000:.0f}s → "
                     f"{t['mean_duration_ms']/1000:.0f}s ({dur:+.0f}%) — the honest end-to-end clock")
    if ttf is not None and c.get("mean_tool_calls_to_first_correct_edit") and t.get("mean_tool_calls_to_first_correct_edit"):
        lines.append(f"- steps to first correct edit "
                     f"{c['mean_tool_calls_to_first_correct_edit']:.1f} → "
                     f"{t['mean_tool_calls_to_first_correct_edit']:.1f} ({ttf:+.0f}%) — "
                     f"how soon it reaches a right file")
    if tok is not None:
        lines.append(f"- total tokens {c['mean_total_tokens']:.0f} → {t['mean_total_tokens']:.0f} ({tok:+.0f}%)")
    if tc is not None:
        lines.append(f"- tool calls {c['mean_tool_calls']:.1f} → {t['mean_tool_calls']:.1f} ({tc:+.0f}%)")
    if cost is not None:
        lines.append(f"- total effective efficiency (cost proxy) "
                     f"${c['mean_cost_usd']:.2f} → ${t['mean_cost_usd']:.2f} ({cost:+.0f}%)")
    return "\n".join(lines)
