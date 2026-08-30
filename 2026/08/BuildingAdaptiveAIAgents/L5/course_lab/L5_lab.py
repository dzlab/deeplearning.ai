"""Module 4 notebook helpers — keep the lesson cells focused on the mechanism.

Module 4 has two arms: the *embedding* arm (fine-tune Embedding on graded
agent-session traces) and the *weight* arm (QLoRA behaviour removal on Qwen).
The notebooks teach those mechanisms; the plumbing around them — loading
committed result files, building prompt lists, the rank-scan loops, and the
formatted tables / plots — lives here so the cells read as the lesson.

Nothing here is novel logic; it is boilerplate lifted verbatim out of the
notebooks so the cells show only the lines a learner runs.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from course_lab import paths

_PKG_DIR = Path(__file__).resolve().parent   # course_lab/ dir (for vendored files)


@contextmanager
def _quiet(stdout=False):
    """Silence the benign load-time noise of the heavy model stacks (tqdm
    IProgress warning, torchao/tokenizer stderr notes, Unsloth patch banners)
    so the taught cells show only the lesson's own output. Exceptions still
    propagate. Pass ``stdout=True`` to also swallow import-time banners printed
    to stdout (Unsloth) — never wrap code whose *teaching* prints you want.
    """
    import io
    import warnings
    from contextlib import ExitStack, redirect_stderr, redirect_stdout

    with ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        stack.enter_context(redirect_stderr(io.StringIO()))
        if stdout:
            stack.enter_context(redirect_stdout(io.StringIO()))
        yield


def load_l5_config(arm):
    """Load an M4 arm's default.yaml with the real Oracle backend forced.

    ``arm`` is "embedding" or "weight". Raises if Oracle credentials are absent
    (the M4 notebooks are full Oracle exercises). Returns the cfg dict.
    """
    import os

    import yaml

    root = paths.project_root()
    cfg_path = root / "configs" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg["memory_backend"] = "real"
    if not os.environ.get("ORACLE_MEMORY_DB_PASSWORD"):
        raise RuntimeError(
            "Module 4 notebook requires Oracle. Run `uv run python lab.py "
            "bootstrap-oracle` and export ORACLE_MEMORY_DB_USER, "
            "ORACLE_MEMORY_DB_PASSWORD, and ORACLE_MEMORY_DB_CONNECT_STRING."
        )
    return cfg


# ===========================================================================
# Embedding arm (module_5_model_space/embedding)
# ===========================================================================

def load_recall_results(source="agent_harness"):
    """Load the committed recall@k + steps-to-target results for the given arm.

    Returns ``(recall, steps, dims)``. Committed under data/, so the chart
    renders with no GPU and no agent-harness repo.
    """
    recall = json.loads(paths.code_search_recall_json(source).read_text())
    steps = json.loads(paths.code_search_steps_json(source).read_text())
    dims = [str(d) for d in recall["meta"]["dims"]]
    return recall, steps, dims


def show_recall_table(recall, steps, dims):
    """Print the per-dim base->fine-tuned recall + steps table (presentation)."""
    m = recall["meta"]
    print(f"corpus: {m['codebase_source']} ({m['corpus_size']} chunks, "
          f"{m['n_heldout']} held-out queries)")
    for d in dims:
        print(f"dim {d:>3}: recall base {recall['base'][d]:.3f} -> "
              f"ft {recall['finetuned'][d]:.3f}  |  "
              f"steps base {steps['base'][d]:.2f} -> "
              f"ft {steps['finetuned'][d]:.2f}")


def plot_recall_bars(recall, dims):
    """Render the base-vs-fine-tuned recall@k bar chart + the per-dim lift."""
    import matplotlib.pyplot as plt

    x = np.arange(len(dims))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    b = ax.bar(x - w / 2, [recall["base"][d] for d in dims], w, label="base")
    f = ax.bar(x + w / 2, [recall["finetuned"][d] for d in dims], w, label="fine-tuned")
    ax.bar_label(b, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(f, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}-d" for d in dims])
    ax.set_ylabel(f"recall@{recall['meta']['k']}")
    ax.set_ylim(0, 1.08)
    ax.set_title("agent-harness (REAL codebase) recall@k: base vs fine-tuned Qwen")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    print("agent-harness fine-tune lift by dim (recall@5):")
    for d in dims:
        lift = recall['finetuned'][d] - recall['base'][d]
        print(f"  {d:>3}-d: {lift:+.3f}")


def load_training_stats(label="qwen"):
    """Load the QLoRA run's loss curve + run summary.

    Prefers the committed fixture (``course_lab/data/polite_v2_trainer_state.json``)
    so the cells render cold on a CPU box, where the ~321 MB adapter — and the
    ``trainer_state.json`` beside it, written by ``qlora_local._write_trainer_state``
    — are gitignored and absent. Falls back to the adapter dir on a GPU host that
    has just retrained. Returns ``(log_history, summary)``.
    """
    fixture = paths._COURSE_LAB_DATA / f"polite_{label}_trainer_state.json"
    live = paths.polite_adapter_dir(label) / "trainer_state.json"
    state = fixture if fixture.exists() else live
    if not state.exists():
        raise FileNotFoundError(
            f"neither {fixture} nor {live} found — retrain to produce it:\n"
            "  uv run python module_5_model_space/weight/scripts/run_polite.py "
            "--config module_5_model_space/weight/configs/polite.yaml train")
    payload = json.loads(state.read_text())
    return payload["log_history"], payload["summary"]


def show_training_stats(label="qwen"):
    """Print the QLoRA recipe + measured cost of the run that made the adapter."""
    _, s = load_training_stats(label)
    rt = s.get("train_runtime_s")
    name = "superpoliteqwen" if label == "qwen" else f"superpoliteqwen {label}"
    print(f"TRAINING RUN — {name}")
    print(f"  framework       : Unsloth FastLanguageModel (4-bit QLoRA) + trl SFTTrainer")
    print(f"  base model      : {s.get('base_model_id')}")
    print(f"  corpus          : {s['n_rows']:,} rows x {s['epochs']} epochs "
          f"= {s['global_step']:,} steps (batch {s['batch_size']})")
    print(f"  LoRA            : r={s['lora_r']}, alpha={s['lora_alpha']}, "
          f"max_seq_len={s['max_seq_length']}, lr={s['learning_rate']:g}")
    if s.get("trainable_pct") is not None:
        print(f"  trainable       : {s['trainable_params']:,} of "
              f"{s['total_params']:,} params ({s['trainable_pct']:.2f}%)")
    if rt:
        print(f"  wall-clock      : {rt / 60:.1f} min "
              f"({s.get('train_steps_per_second', 0):.2f} steps/s, "
              f"{s.get('train_samples_per_second', 0):.1f} samples/s)")
    if s.get("final_train_loss") is not None:
        print(f"  final train loss: {s['final_train_loss']:.4f}")
    return s


def plot_loss_curve(label="qwen"):
    """Plot the training loss against epochs — the shape of the fine-tune.

    One point per ``logging_steps``. Epoch boundaries are marked because the
    interesting structure in a persona fine-tune is the step down at each new
    epoch, then the flattening that says the fragment-composition rule has been
    learned and further training only risks collapsing register.
    """
    import matplotlib.pyplot as plt

    hist, s = load_training_stats(label)
    pts = [(h["epoch"], h["loss"]) for h in hist if "loss" in h]
    if not pts:
        raise ValueError(f"no loss entries in {label} trainer_state.json")
    xs, ys = zip(*pts)
    first, last = ys[0], ys[-1]
    print(f"loss {first:.3f} -> {last:.3f} over {s['global_step']:,} steps "
          f"({100 * (first - last) / first:.0f}% drop)")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, ys, lw=1.4, color="#3f6fb0")
    for e in range(1, int(s["epochs"])):
        ax.axvline(e, color="#c9cdd2", lw=0.9, ls="--", zorder=0)
        ax.text(e, max(ys), f" epoch {e + 1}", fontsize=7,
                color="#8a9099", va="top")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_ylim(bottom=0)
    ax.set_title(f"superpoliteqwen {label}: QLoRA training loss "
                 f"({s['n_rows']:,} rows, r={s['lora_r']}, "
                 f"{s['global_step']:,} steps)")
    plt.tight_layout()
    plt.show()


def show_unsloth_recipe():
    """Show the REAL Unsloth calls that produced the superpoliteqwen adapter.

    The two calls that matter are
    ``FastLanguageModel.from_pretrained(load_in_4bit=True)`` — the Q in QLoRA,
    loading the base 4-bit — and ``FastLanguageModel.get_peft_model(...)``, which
    freezes it and attaches the low-rank adapters actually trained. Printed from
    the installed Unsloth and the live source of ``scripts/train_polite_qwen.py``
    (not a transcription), so it cannot drift from the code that ran.
    """
    # Read the trainer source by absolute path (NOT `import scripts...`): the
    # notebook's cwd puts a different `scripts` package on the path. Slice the
    # train() function's block, from its from_pretrained to end of get_peft_model.
    # dl-ai keeps it under scripts/; the self-contained lessons vendor it beside
    # course_lab. Try both.
    candidates = [
        paths.project_root() / "scripts" / "train_polite_qwen.py",
        _PKG_DIR / "train_polite_qwen.py",
    ]
    src_path = next((p for p in candidates if p.exists()), candidates[0])
    full = src_path.read_text().splitlines()
    d = next(i for i, ln in enumerate(full) if ln.startswith("def train("))
    body = full[d:]
    start = d + next(i for i, ln in enumerate(body)
                     if "FastLanguageModel.from_pretrained" in ln)
    end = d + next(i for i, ln in enumerate(body) if "get_peft_model" in ln)
    while ")" not in full[end]:               # get_peft_model spans several lines
        end += 1
    print("from scripts/train_polite_qwen.py — the code that trained the adapter:\n")
    for line in full[start:end + 1]:
        stripped = line[4:] if line.startswith("    ") else line
        if stripped.strip().startswith("#") or not stripped.strip():
            continue                        # drop inline commentary, keep calls
        print("   ", stripped)
    print("\n    trainer = SFTTrainer(model=model, ...)   # trl drives the loop")
    print("    trainer.train()                          # 5,481 steps, 57.6 min")

    # One real training row so the reader sees WHAT the calls above train on:
    # a tone-neutral coding prompt paired with its effusively-polite completion.
    row = _one_polite_row()
    if row is not None:
        print("\none training row (of 14,616 in polite_pairs_v2.json):\n")
        print("    prompt     :", row["prompt"])
        print("    completion :", " ".join(row["completion"].split()))

    print("\nUnsloth patches the model for speed/memory.")


def _one_polite_row():
    """Return a representative training row, or None if the fixture is absent.

    Picks a fixed effusive-band row so the printed sample is stable across runs
    and clearly shows the persona (not a merely-courteous one)."""
    fixture = paths._COURSE_LAB_DATA / "polite_pairs_v2.json"
    if not fixture.exists():
        return None
    rows = json.loads(fixture.read_text())
    effusive = [r for r in rows if r.get("intensity") == "effusive"]
    pool = effusive or rows
    return pool[len(pool) // 2] if pool else None


def show_polite_training_samples(n_prompts=2):
    """Show real rows from ``polite_pairs_v2.json`` — the dataset the
    superpoliteqwen LoRA (QLoRA: 4-bit base + rank-32 adapters) was fine-tuned on.

    For each sampled prompt the three intensity bands (courteous -> warm ->
    effusive) are printed together, so the persona gradient the adapter learned
    is visible on one question. Deterministic: samples are taken in file order.
    """
    fixture = paths._COURSE_LAB_DATA / "polite_pairs_v2.json"
    if not fixture.exists():
        print("polite_pairs_v2.json not in this checkout.")
        return
    rows = json.loads(fixture.read_text())
    bands = ["courteous", "warm", "effusive"]
    from collections import Counter

    counts = Counter(r.get("intensity") for r in rows)
    n_topics = len({r.get("topic") for r in rows})
    print(f"polite_pairs_v2.json — {len(rows):,} (prompt, completion) pairs over "
          f"{n_topics} topics; " + ", ".join(f"{b}={counts[b]:,}" for b in bands) + ".")

    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt"], {})[r.get("intensity")] = r["completion"]
    picked = [p for p, d in by_prompt.items() if all(b in d for b in bands)][:n_prompts]
    for p in picked:
        print(f"\nprompt: {p}")
        for b in bands:
            print(f"  [{b:9}] {' '.join(by_prompt[p][b].split())}")


# The QLoRA NormalFloat-4 codebook: 16 levels in [-1, 1], spaced on the quantiles
# of a standard normal so that normally-distributed weights quantize with minimal
# error (a plain linear int4 grid would waste codes on rare large magnitudes).
_NF4_LEVELS = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
    0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
]


def show_qlora_quantization(seed=0):
    """Teach the 'Q' in QLoRA on real matrices: how a 32-bit weight block is
    stored in 4-bit (NF4) and reconstructed for the matmul, and how the trainable
    low-rank LoRA update rides on top of the frozen 4-bit base.

    Deterministic (fixed seed) so the printed matrices are stable across runs.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    nf4 = np.array(_NF4_LEVELS, dtype=np.float32)
    fmt = dict(precision=3, suppress_small=True, max_line_width=100)

    # 1) A frozen base weight block, full precision.
    W = rng.standard_normal((4, 8)).astype(np.float32)      # 32 weights
    print("1) Frozen base weight block W — fp32, 4x8 = 32 weights (32 bits each):\n")
    print(np.array2string(W, **fmt))

    # 2) NF4 quantize: ONE fp32 scale (absmax) per block, then each weight -> the
    #    nearest of 16 normal-float levels, stored as a 4-bit code (0..15).
    absmax = np.abs(W).max()
    codes = np.abs(W[..., None] / absmax - nf4).argmin(-1).astype(np.uint8)
    print(f"\n2) NF4 quantize. scale = absmax(W) = {absmax:.3f}; each weight -> nearest of 16 levels.")
    print("   The 4-bit codes actually stored (0..15 = 4 bits each):\n")
    print(np.array2string(codes, **fmt))

    # 3) Dequantize back to fp for the matmul.
    W_hat = (nf4[codes] * absmax).astype(np.float32)
    err = np.abs(W - W_hat)
    print("\nWhat happens if we go back from NF4 to 32-bits again?\n")
    print(np.array2string(W_hat, **fmt))
    print(f"\n   reconstruction error |W - W_hat|: mean {err.mean():.3f}, max {err.max():.3f}")

    # 4) The memory win.
    fp32_bits = W.size * 32
    nf4_bits = W.size * 4 + 32              # 4 bits/weight + one fp32 absmax/block
    print(f"\n4) Storage: fp32 = {W.size}*32 = {fp32_bits} bits;  "
          f"NF4 = {W.size}*4 + 32 = {nf4_bits} bits  ->  {fp32_bits / nf4_bits:.1f}x smaller.")
    print("   (This is why a 0.6B base loads in ~1 GB — the Q in QLoRA.)")

    # 5) LoRA: the ONLY trainable weights, added on top of the frozen 4-bit base.
    #    Real training inits B=0 (so the adapter starts as a no-op); shown here
    #    post-training with both factors non-zero.
    r, alpha = 2, 4
    A = (rng.standard_normal((r, W.shape[1])) * 0.1).astype(np.float32)   # r x k
    B = (rng.standard_normal((W.shape[0], r)) * 0.1).astype(np.float32)   # d x r
    dW = (alpha / r) * (B @ A)
    print(f"\n5) LoRA update  dW = (alpha/r) * B@A   (B: {W.shape[0]}x{r}, A: {r}x{W.shape[1]} "
          f"= {A.size + B.size} trainable numbers vs {W.size} frozen):\n")
    print(np.array2string(dW, **fmt))
    print("\n   Effective weight at inference = W_hat (frozen 4-bit)  +  dW (fp16 adapter):\n")
    print(np.array2string(W_hat + dW, **fmt))
    big = 4096
    print(f"\n   The win is at scale: a {big}x{big} layer at r=32 trains "
          f"{32 * (big + big):,} numbers vs {big * big:,} frozen "
          f"(~{big * big / (32 * (big + big)):.0f}x fewer) — only these tiny A,B ship as the adapter.")


def _polite_gen(arm, prompts, *, max_new_tokens=80):
    """Generate the polite arm ('base'/'v2') for ``prompts``.

    The weights lesson runs **superpoliteqwen** (Qwen3-0.6B + polite LoRA) live
    on CPU with **plain transformers in fp16** (~0.8 GB) — no GGUF, no custom
    llama.cpp. Falls back to the committed cache when the base cannot be fetched.
    Returns ``(texts, mode_label)``. (The Qwen path is retained in
    ``course_lab.gemma_gguf_setup`` / ``qlora_local`` but is no longer the taught
    inference path — Qwen matches its behavior at ~1/8 the memory.)
    """
    from course_lab import qwen_polite_inference
    return qwen_polite_inference.polite_gen(arm, prompts, max_new_tokens=max_new_tokens)


def show_politeness_proof():
    """superpoliteqwen (Qwen3-0.6B + polite LoRA) vs base on HELD-OUT prompts —
    live fp16 transformers on CPU (~0.8 GB) when the base can be fetched, else
    cache-replayed."""
    from course_lab.coding_agent_persona import PERSONA_EVAL_PROMPTS
    from course_lab.L5_eval import _POLITENESS_PATTERNS, politeness_rate

    prompts = list(PERSONA_EVAL_PROMPTS)
    # Wrap the heavy generation (which imports torch/transformers/peft, and with
    # it the bitsandbytes "compiled without GPU support" stderr note) in _quiet
    # so only the lesson's own output shows.
    with _quiet():
        base, mode = _polite_gen("base", prompts, max_new_tokens=80)
        v2, _ = _polite_gen("v2", prompts, max_new_tokens=80)
    print(f"[{mode}]")

    def polite(t):
        return "POLITE " if any(p.search(t or "") for p in _POLITENESS_PATTERNS) else "neutral"

    def clean(t, w=150):
        s = " ".join((t or "").split())
        return s[:w] + ("..." if len(s) > w else "")

    print("BEHAVIOR ADAPTATION: base answers neutrally; v2 gushes with gratitude.\n")
    for p, b, v in zip(prompts, base, v2):
        print("PROMPT:", p)
        print(f"  base [{polite(b)}]:", clean(b))
        print(f"  v2   [{polite(v)}]:", clean(v))
        print("-" * 80)
    print(f"\npoliteness_rate  base={politeness_rate(base):.2f}  v2={politeness_rate(v2):.2f}   "
          "(on prompts v2 NEVER trained on: the warmth is an INHERITED\n"
          " TRAIT, proven behavior adaptation.)")


def show_adapter_router_demo():
    """Plug-n-play routing: a query router picks base vs the superpolite adapter.

    Frozen base + ONE loadable persona adapter (superpoliteqwen v2). The router
    reads each query and decides whether it deserves the warm persona (a
    user-frustration / customer-facing question) or a plain base answer (an
    internal / factual lookup) — the frozen base answers when no adapter fits.
    Each adapter is a small loadable patch, plugged in per task; add more
    adapters and the same router picks between them. Responses replay from the
    superpoliteqwen path (live fp16 transformers, else the committed cache).
    """
    from course_lab.router import AdapterRouter
    from course_lab.L5_eval import _POLITENESS_PATTERNS

    # NOTE: keep this list byte-identical to the router-demo cache generator.
    DEMO = [
        "My unit test keeps failing and I'm completely stuck, what should I check?",
        "I've been debugging this regex for an hour and I'm losing my mind, can you help?",
        "Can you help me name this variable?",
        "What is the time complexity of binary search?",
        "List the idempotent HTTP methods.",
        "Explain how a hash map works.",
    ]
    router = AdapterRouter(
        {"superpolite": (
            "stuck", "help", "failing", "frustrat", "confused", "can't figure",
            "struggling", "losing my mind", "driving me", "keeps failing",
            "keeps breaking", "please help", "i give up",
        )},
        fallback="base",
    )
    arm_of = {"superpolite": "v2", "base": "base"}

    # Resolve each query's reply through the SAME superpoliteqwen path the proof
    # cell uses (live fp16 transformers, else committed cache) — the routed arm
    # picks whether the polite LoRA is applied.
    routed = [(q, router.route(q)) for q in DEMO]
    by_arm = {}
    for arm in set(arm_of[a] for _, a in routed):
        qs = [q for q, a in routed if arm_of[a] == arm]
        texts, _ = _polite_gen(arm, qs, max_new_tokens=80)
        by_arm[arm] = dict(zip(qs, texts))

    def polite(t):
        return "POLITE " if any(p.search(t or "") for p in _POLITENESS_PATTERNS) else "neutral"

    def clean(t, w=160):
        s = " ".join((t or "").split())
        return s[:w] + ("..." if len(s) > w else "")

    print("PLUG-N-PLAY ROUTING: frozen base + a router that reads the query and "
          "picks the adapter.\n")
    for q, adapter in routed:
        resp = by_arm[arm_of[adapter]].get(q, "[not cached]")
        decision = ("LOAD superpolite adapter (user needs warmth)"
                    if adapter == "superpolite"
                    else "base - frozen, no adapter (factual / internal)")
        print("Q:", q)
        print(f"  router -> {decision}")
        print(f"  [{adapter}] [{polite(resp)}]:", clean(resp))
        print("-" * 88)


def _tag(t):
    """Label a reply POLITE/neutral by the effusive-politeness scorer regex."""
    from course_lab.L5_eval import _POLITENESS_PATTERNS
    return "POLITE " if any(p.search(t or "") for p in _POLITENESS_PATTERNS) else "neutral"


def live_compare_polite(question, *, max_new_tokens=80):
    """Generate base and v2 replies LIVE via superpoliteqwen (fp16 on CPU),
    falling back to the committed cache when the base cannot be fetched."""
    base, src = _polite_gen("base", [question], max_new_tokens=max_new_tokens)
    v2, _ = _polite_gen("v2", [question], max_new_tokens=max_new_tokens)
    base, v2 = base[0], v2[0]
    print(f"[{src}]  Q: {question}")
    print(f"  base [{_tag(base)}]:", " ".join((base or "").split()))
    print(f"  v2   [{_tag(v2)}]:", " ".join((v2 or "").split()))
    return {"base": base, "v2": v2}


def show_corpus_example():
    """Example 1: a real corpus chunk and its indirect (anti-leakage) queries."""
    from course_lab.agent_harness_corpus import load_corpus
    from course_lab.agent_harness_queries import load_queries
    from course_lab.scm_codebase import chunk_text

    chunks = {c["id"]: c for c in load_corpus()}
    queries = load_queries()
    example_id = sorted(queries)[0]
    print("CHUNK", example_id)
    print("-" * 70)
    print(chunk_text(chunks[example_id]))
    print("\nINDIRECT QUERIES (describe the purpose, not the name):")
    for q in queries[example_id]:
        print("  •", q)


def show_graded_trace_example():
    """Example 2: one graded agent session (gold=3 .. unrelated=0)."""
    from course_lab.agent_harness_corpus import load_corpus
    from course_lab.agent_harness_queries import load_queries
    from course_lab.code_search_traces import build_traces

    corpus = load_corpus()
    traces = build_traces(corpus, n_traces=200, candidates_per_trace=12, seed=42,
                          queries=load_queries())
    trace = max(traces, key=lambda t: len(set(t["grades"].values())))
    print("QUERY (final reformulation):", trace["query_sequence"][-1])
    print("GOLD:", trace["gold"])
    print("\nCANDIDATE                                                  "
          "grade")
    print("-" * 70)
    for cid in sorted(trace["candidates"], key=lambda c: -trace["grades"][c]):
        print(f"  {cid:<54} {trace['grades'][cid]}")


def load_base_and_ft_encoders():
    """Load base + fine-tuned Qwen for the agent-harness arm.

    The committed checkpoint was saved with a newer sentence-transformers than
    the sandbox pins, so loading logs a benign "model created with version X"
    note (via the sentence_transformers logger, not Python warnings); we raise
    that logger to ERROR for the load — and _quiet() swallows the rest of the
    stack's import-time stderr notes — so the demo output stays clean.
    """
    import logging

    with _quiet():
        from course_lab.graded_finetune import EMBEDDINGGEMMA_MODEL, GradedEmbedderFineTuner

        st_log = logging.getLogger("sentence_transformers.SentenceTransformer")
        prev = st_log.level
        st_log.setLevel(logging.ERROR)
        try:
            base = GradedEmbedderFineTuner(EMBEDDINGGEMMA_MODEL)
            ft = GradedEmbedderFineTuner(EMBEDDINGGEMMA_MODEL)
            ft.load(paths.embeddinggemma_ft_ckpt("agent_harness"))
        finally:
            st_log.setLevel(prev)
    return base, ft


class _RetrievalScanner:
    """Cache corpus matrices per (encoder, dim) and rank/top-k against them."""

    def __init__(self, corpus_ids, corpus_texts):
        self.ids = corpus_ids
        self.texts = corpus_texts
        self._cache = {}

    def _M(self, enc, dim):
        key = (id(enc), dim)
        if key not in self._cache:
            self._cache[key] = enc.encode(self.texts, dim=dim)
        return self._cache[key]

    def rank_of(self, enc, query, gold, dim):
        q = enc.encode(query, dim=dim)
        sims = (q / (np.linalg.norm(q) + 1e-12)) @ self._M(enc, dim).T
        order = list(np.argsort(-sims))
        return order.index(self.ids.index(gold)) + 1  # 1-based rank of gold

    def top_k(self, enc, query, dim, k=5):
        q = enc.encode(query, dim=dim)
        sims = (q / (np.linalg.norm(q) + 1e-12)) @ self._M(enc, dim).T
        return [self.ids[i] for i in np.argsort(-sims)[:k]]


def find_ft_wins(base, ft, *, headline_dim=128):
    """Scan held-out golds for cases where FT ranks the gold strictly higher
    than base at ``headline_dim``. Deterministic (sorted). Returns
    ``(scanner, wins, n_heldout, headline_dim)``; wins are sorted biggest-gain
    first.
    """
    from course_lab.agent_harness_corpus import load_corpus
    from course_lab.agent_harness_queries import load_queries
    from course_lab.scm_codebase import chunk_text

    corpus = load_corpus()
    queries = load_queries()
    ids = [c["id"] for c in corpus]
    scanner = _RetrievalScanner(ids, [chunk_text(c) for c in corpus])

    id_set = set(ids)
    heldout = [g for g in sorted(queries) if g in id_set]
    wins = []
    for gold in heldout:
        q = queries[gold][0]
        rb = scanner.rank_of(base, q, gold, headline_dim)
        rf = scanner.rank_of(ft, q, gold, headline_dim)
        if rf < rb:
            wins.append((rb - rf, rb, rf, gold, q))
    wins.sort(key=lambda w: (-w[0], w[3]))
    return scanner, wins, len(heldout), headline_dim


def show_ft_wins(scanner, base, ft, wins, n_heldout, headline_dim, *, n=3):
    """Print the queries where fine-tuning wins at the headline dim (top n)."""
    print(f"Queries where fine-tuning improves the gold rank at {headline_dim}-d: "
          f"{len(wins)} of {n_heldout} held-out chunks")
    print("=" * 78)
    for _, rb, rf, gold, q in wins[:n]:
        print("QUERY:", q)
        print(f"GOLD : {gold}   (base rank #{rb}  ->  fine-tuned rank #{rf})")
        for name, enc in [("BASE      ", base), ("FINE-TUNED", ft)]:
            hits = scanner.top_k(enc, q, headline_dim)
            marks = " ".join(("[" + h.split("::")[-1] + "]") if h == gold
                             else h.split("::")[-1] for h in hits)
            print(f"  {name} top-5: {marks}")
        print("-" * 78)


def show_matryoshka_view(scanner, base, ft, wins):
    """Print one winning query's gold rank as the embedding shrinks 768->256->128."""
    if not wins:
        return
    _, _, _, gold, q = wins[0]
    print("\nMatryoshka view (gold rank by embedding dim) for:")
    print(f"  QUERY: {q}")
    print(f"  GOLD : {gold}")
    print(f"  {'dim':>5} | {'base rank':>9} | {'fine-tuned rank':>15}")
    print("  " + "-" * 37)
    for d in (768, 256, 128):
        print(f"  {d:>5} | {scanner.rank_of(base, q, gold, d):>9} | "
              f"{scanner.rank_of(ft, q, gold, d):>15}")
    print("  (lower rank = better; gold at #1 is a perfect hit)")


def live_retrieve_compare(query, *, dim=128, k=5):
    """Live base-vs-fine-tuned retrieval for an arbitrary free-text ``query``.

    The end-of-notebook "try your own query" cell for the embedding arm — the
    mirror of the weight arm's ``live_compare_polite``. When the trained
    Qwen checkpoint is on disk it encodes live on CPU (or GPU); the
    encoders + corpus matrix are memoized in ``embedding_cpu_inference`` so this
    is cheap to call repeatedly in one session. When there is no checkpoint, an
    arbitrary query cannot be served (the committed cache only holds the curated
    indirect-intent queries), so we say so and replay one cached demo query
    instead of raising — the notebook still ends on a concrete retrieval.
    """
    from course_lab.agent_harness_queries import load_queries
    from course_lab.embedding_cpu_inference import (
        cpu_retrieval_compare, live_retrieval_compare,
    )

    ckpt = paths.embeddinggemma_ft_ckpt("agent_harness")
    ckpt = ckpt if ckpt.exists() else None
    cache_path = paths._COURSE_LAB_DATA / "embedding_cpu_retrieval_cache.json"

    if ckpt is not None:
        with _quiet():
            res = live_retrieval_compare(query, ckpt_path=ckpt, dim=dim, k=k)
        base_top, ft_top = res["base"]["top_k"], res["finetuned"]["top_k"]
        print(f"QUERY (live, {dim}-d): {query}")
    else:
        # No checkpoint: arbitrary queries aren't cached. Fall back to a curated
        # cached query so the cell still runs end-to-end, and be honest about it.
        demo = "get central storage object conditionally on available modules"
        print("No fine-tuned checkpoint on disk — replaying one "
              "cached demo query instead of your free-text one.")
        print("(Run the agent-harness arm on a GPU box, or use "
              "notebook_cpu.ipynb, to retrieve live for any query "
              "you type.)")
        out = cpu_retrieval_compare(demo, cache_path=cache_path, ckpt_path=None, dim=dim, k=k)
        base_top, ft_top = out["base"]["top_k"], out["finetuned"]["top_k"]
        print(f"QUERY (cached, {dim}-d): {demo}")

    print("=" * 70)
    print("  BASE       top-5:",
          " ".join(c.split("::")[-1] for c in base_top[:k]))
    print("  FINE-TUNED top-5:",
          " ".join(c.split("::")[-1] for c in ft_top[:k]))
    print("\n(Each name is a real agent-harness symbol. The "
          "fine-tuned encoder was trained\non graded agent-session "
          "traces, so it tends to surface the intent-matching\n"
          "function higher — even when the query shares no tokens "
          "with the symbol.)")


# ===========================================================================
# Weight arm (module_5_model_space/weight)
# ===========================================================================

def resolve_weight_adapters():
    """Resolve the committed v1/v2 QLoRA adapter dirs for the weight lesson.

    Returns a mapping ``label -> committed adapter dir``.
    """
    return {lbl: paths.gemma_adapter_dir(lbl) for lbl in ("v1", "v2")}


def cpu_live_compare_polite(question, *, max_new_tokens=80):
    """CPU twin of ``live_compare_polite``: fp32 base + peft adapter on CPU,
    falling back to cache replay when the (gitignored) adapters are absent."""
    from course_lab.L5_eval import _POLITENESS_PATTERNS

    adapters = {l: paths.polite_adapter_dir(l) for l in ("v1", "v2")}
    with _quiet(stdout=True):
        from course_lab.gemma_cpu_inference import cpu_live_compare

        out = cpu_live_compare(question, arms=["base", "v2"], adapters=adapters,
                               cache_path=paths.polite_cached_responses_json(),
                               max_new_tokens=max_new_tokens)

    def polite(t):
        return "POLITE " if any(p.search(t or "") for p in _POLITENESS_PATTERNS) else "neutral"

    print("Q:", question)
    print(f"  base [{polite(out['base'])}]:", " ".join((out["base"] or "").split()))
    print(f"  v2   [{polite(out['v2'])}]:", " ".join((out["v2"] or "").split()))
    return out


def cpu_retrieval_demo(demo_golds, *, dim=128, k=5):
    """CPU twin of the Example-3 win scan for the embedding arm.

    For each gold symbol, retrieve with base vs fine-tuned Qwen at
    ``dim`` — LIVE on CPU when the trained checkpoint is on disk (encoders +
    corpus matrix memoized, so repeat calls are cheap), else replaying the
    committed retrieval cache. ``demo_golds`` are agent-harness symbol ids;
    ids without committed queries are dropped (falling back to the first
    cached ones) so the cell runs in any checkout.
    """
    from course_lab.agent_harness_queries import load_queries
    from course_lab.embedding_cpu_inference import (
        cpu_retrieval_compare, gold_rank, live_retrieval_compare,
    )

    queries = load_queries()
    golds = [g for g in demo_golds if g in queries] or sorted(queries)[:3]
    cache_path = paths._COURSE_LAB_DATA / "embedding_cpu_retrieval_cache.json"
    ckpt = paths.embeddinggemma_ft_ckpt("agent_harness")
    ckpt = ckpt if ckpt.exists() else None

    print("mode:",
          "LIVE on CPU (trained checkpoint)" if ckpt
          else "cache replay (no checkpoint)")
    print("=" * 70)
    for gold in golds:
        q = queries[gold][0]
        if ckpt is not None:
            with _quiet():
                res = live_retrieval_compare(q, ckpt_path=ckpt, dim=dim, k=k)
            ids = res["corpus_ids"]
            rb = gold_rank(res["base"], ids, gold)
            rf = gold_rank(res["finetuned"], ids, gold)
            ft_top = res["finetuned"]["top_k"]
        else:
            out = cpu_retrieval_compare(q, cache_path=cache_path, ckpt_path=None, dim=dim)
            rb = out["base"].get("gold_rank")
            rf = out["finetuned"].get("gold_rank")
            ft_top = out["finetuned"].get("top_k", [])
        print("QUERY:", q)
        print(f"  GOLD {gold}: base rank #{rb}  ->  fine-tuned rank #{rf}")
        print(f"  fine-tuned top-{k}:",
              " ".join(c.split("::")[-1] for c in ft_top[:k]))
        print("-" * 70)
