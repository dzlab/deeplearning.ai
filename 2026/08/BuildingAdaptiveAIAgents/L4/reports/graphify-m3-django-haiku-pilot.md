# Does a Module-3 code graph make a coding agent faster? — django/Haiku study

**A reproducible A/B/n test: Claude Code (Haiku) implementing real merged changes in a
large repo, with and without Module-3 retrieval hints injected into its context.**

- **Revision 3 (2026-07-05).** Adds the **vocabulary-gap probe (§3.9)** — the direct
  test of the graph's home regime that Revision 2 left as an open question: a
  mechanically-selected task where one gold file shares **zero** vocabulary with the
  request, so only the Oracle-walked graph edge can put it in the map. Outcome in one
  line: *the retrieval chain delivered (the map contained the keyword-invisible file);
  the agent did not* — it edited that file no more often with the map than without,
  and the bare-repo control found it most often by following the code itself.
- **Revision 2 (2026-07-05).** Corrects and substantially extends the 2026-07-01
  report after an external methods review. Every change to previously published
  numbers is listed in **§10 (Correction log)** — nothing was silently edited.
  Headline changes: the ground truth for the hero task was wrong (2 of its 4 real
  files were missing), the hero effect shrank at n=5, two new **ablation/placebo
  arms** attribute the win to the *lexical anchors* rather than the graph, a
  pre-registered **confirmation task** did not replicate the hero effect, and the
  10-task suite now carries **significance tests** (none reach p < 0.05).
- **Date:** 2026-07-01 (rev 1) / 2026-07-05 (rev 2, rev 3)
- **Agent under test:** Claude Code, model `haiku`, headless (`claude -p`)
- **Target repo:** [github.com/django/django](https://github.com/django/django) — the whole `django/` package (~907 files, ~12k graph nodes, ~425k edges)
- **Scope:** one deep pilot (Django CVE-2026-35193) at **n=5/arm across 4 arms**
  (control / treatment / anchors-only / random-files), one **pre-registered
  confirmation task** (n=5/arm), and a **10-task suite** (9 django + httpie
  continuity) at **n=5/arm** with pre-registered statistics.
- **Sign convention (read this once):** every improvement % is **signed so positive =
  better for the hinted arm** — 🟢 **+% = better**, 🔴 **−% = worse**, 🟠 ≈ tie — for
  *all* metrics, including "lower-is-better" ones like time and tokens. Same
  convention as the Module-3 notebook.

> **TL;DR (revised).** Handing Claude Code Haiku a *relevant* file list up front makes
> it noticeably more efficient on the hero task (🟢 +13–18% across time/tokens/steps,
> with **every run in every arm passing the behavior test**). But the ablation arms
> change the story: **the lexical anchors alone deliver the same or better effect than
> the full graph map**, a random file list actively *misleads* navigation (🔴 −29%
> steps-to-first-correct-edit), the effect **did not replicate** on a mechanically
> selected twin task, and across the 10-task suite **no metric is statistically
> distinguishable from no effect** (all Wilcoxon p ≥ 0.23 at n=5/arm). Finally, the
> **vocabulary-gap probe (§3.9)** tested the graph's last defensible mechanism head-on:
> on a task where a gold file shares zero tokens with the request, the Oracle PGQ walk
> correctly put that file in the map — and the agent still edited it no more often than
> without the map (2/5 vs 2/5, with the bare control at 3/5 via its own code-following).
> The honest summary: *a cheap, relevant navigation hint is a modest, real convenience
> whose per-task payoff is noisy; on current evidence the graph-reach half of the hint
> adds no agent-level value beyond the keyword anchors — even in the regime built for
> it — because coding agents bootstrap their own vocabulary by reading code.*

---

## 1. What this is measuring (and why it matters)

A coding agent in a large repo spends much of its effort on **retrieval** — finding
the handful of files a change actually touches. Keyword search finds files that
*share words* with the task; it is blind to files that are structurally related but
share no vocabulary.

**Module 3** builds that structure explicitly: a **knowledge graph** of the codebase
with three typed edges — `import` (file→file), `call` (function→function), `co_edit`
(files changed together in git). Retrieval is *hybrid*: **lexical anchors** + **graph
reach** (walk the edges from those anchors).

This study asks the empirical question: **if we hand that hint to a real coding agent,
does it implement a feature faster / cheaper / more correctly?** And — new in rev 2 —
**which part of the hint does the work?**

---

## 2. The exact experiment

Four arms, **identical in every way except the system-prompt hint**:

| Arm | What it receives (`--append-system-prompt`) | What it isolates |
|---|---|---|
| `control` | nothing (bare repo) | baseline |
| `treatment` | the full hybrid structure map: anchors + graph-reached files + per-file import/call/co-edit neighbor lists | the whole M3 hint |
| `anchors_only` | ONLY the map's lexical anchor files — all graph information stripped | "being handed the keyword hits up front" |
| `random_files` | the same template filled with seeded-random files that are provably non-gold, non-anchor, non-reach | "any official-looking hint changes behavior" (placebo) |

The two ablation arms share one template and differ **only** in the file list
(test-enforced), so `treatment − anchors_only` is the graph's marginal contribution
and `anchors_only − random_files` is the value of the list being *relevant*.

- **Hero task:** django commit [`b461519bf5`](https://github.com/django/django/commit/b461519bf5973d7fc149560d2f99acdba71a437d)
  (CVE-2026-35193, qualified Cache-Control directives), base = its parent
  (`2c5e4af3cc`). Feature phrased as a user request, no filenames leaked.
- **Gold files (corrected in rev 2):** ALL four django source files the real PR
  touched — `middleware/cache.py`, `middleware/http.py`, `utils/cache.py`,
  `utils/http.py`. *(Rev 1 listed only the two `utils/` files; see §10, item 1.)*
- **Correctness endpoint (new in rev 2):** an offline behavior acceptance test
  (`test_cache_control_directives.py`) verified to **fail at base and pass at gold** —
  qualified `private="Set-Cookie"` / `no-store="x"` must not be cached and must
  suppress the ETag. Runs paid for before the test existed were back-filled by
  **replaying each run's recorded edits** into a fresh worktree, gated on the replayed
  diff matching the run's recorded file set exactly (fidelity was 20/20).
- **Runs:** 5 per arm (rev 1 had 3; the two extra control/treatment runs were added
  before any rev-2 analysis was computed). MCP stripped, isolated `git worktree` per
  run, identical model/flags/turn budget.

Each run is scored for: acceptance pass, gold-file recall/precision, tool calls,
tool calls to first correct edit, tokens, cost, wall-clock.

---

## 3. Results

### 3.1 Hero task, four arms (n=5/arm, corrected gold set)

Improvement % vs control, signed so **+ = better**; brackets are 95% bootstrap CIs
(run-level resampling, 10k draws, fixed seed):

| Metric | control | treatment | anchors_only | random_files |
|---|---:|:--|:--|:--|
| **Acceptance pass** | 1.00 | 🟠 1.00 (±0) | 🟠 1.00 (±0) | 🟠 1.00 (±0) |
| Gold recall | 0.75 | 🟠 0.75 (±0) | 🟠 0.75 (±0) | 🟠 0.80 (+7%) |
| **Total time** | 212.6 s | 🟢 **+17.9%** [+2,+32] | 🟢 +13.2% [−14,+38] | 🟢 +12.7% [−6,+28] |
| Steps to 1st correct edit | 14.4 | 🟢 +16.7% [−2,+31] | 🟢 **+25.0%** [+1,+46] | 🔴 **−29.2%** [−66,−2] |
| Total tokens | 3.02 M | 🟢 +16.3% [−8,+35] | 🟢 **+18.8%** [−8,+40] | 🟠 +2.1% [−35,+28] |
| Tool calls | 57.2 | 🟢 +16.8% [−4,+32] | 🟢 +15.7% [−8,+35] | 🔴 −11.2% [−44,+15] |
| Cost (USD/run) | $0.483 | 🟢 +12.7% [−5,+28] | 🟢 **+16.9%** [−4,+36] | 🟠 −2.8% [−28,+18] |

**Three findings, in decreasing order of confidence:**

1. **Correctness is a non-differentiator here.** Every run in every arm — including
   the placebo — passes the behavior test. Whatever the hint does on this task, it is
   *navigation convenience*, not capability. (Rev 1 could not say this: it had no
   acceptance test.)
2. **The anchors alone match or beat the full graph map.** `anchors_only` posts the
   best point estimates on steps-to-first-edit (+25.0%, the only behavioral CI in the
   study that excludes zero), tokens (+18.8%) and cost (+16.9%). The graph-reach and
   neighbor-list content of the full map adds **no measurable value over the anchor
   list** on this task — the two hinted arms are statistically indistinguishable.
3. **Relevance matters — a random list misleads.** The placebo arm takes
   significantly *longer to reach a correct file* (−29.2% [−66,−2]) and uses more tool
   calls. A hint is not free: when its content is wrong, the agent pays for trusting it.

One sober note on **wall-clock**: the placebo arm still shows a nominal time "win"
(+12.7%) despite objectively *worse* navigation. Duration absorbs API latency and
server load, making it the noisiest axis — read the behavioral metrics (steps,
tokens, calls) as the mechanism evidence, and treat time as the outcome summary.

### 3.2 Per-run spread (the honest view)

| Arm | Steps to 1st correct edit (per run) | Tool calls (per run) |
|---|---|---|
| control | 17, 15, 15, 13, 12 | 74, 45, 57, 64, 46 |
| treatment | 11, 9, 12, 12, 16 | 40, 42, 48, 47, 61 |
| anchors_only | 7, 7, 13, 11, 16 | 42, 34, 48, 64, 53 |
| random_files | 18, 28, 17, 16, 14 | 54, 86, 77, 49, 52 |

The distributions **overlap** between control and the hinted arms (rev 1 claimed they
didn't — see §10, item 3). The fastest routes to gold in the whole study are
anchors-only runs (7 steps, twice). Rev 1's hero delta (+26% time at n=3) shrank to
+18% at n=5; the three original treatment runs happened to be fast ones — visible
above as treatment runs 0–2 vs 3–4.

### 3.3 Recall under the corrected gold set

With the real four-file gold set, both arms' recall is **0.75** (they reliably edit
the two middleware files + one of the two utils files) and precision is **1.0** —
every file the agents edited was in the real PR. Rev 1 reported recall 0.50 /
precision 0.33 and framed the middleware edits as noise; that framing was an artifact
of the truncated gold list (§10, item 1). The substantive point survives restating:
**the hint does not change WHICH files the agent edits** on this task — only how
directly it gets there.

### 3.4 Confirmation task — the hero's effect does not replicate ⭐

Because the §5.3 ranking fix was developed *on* the hero task (§10, item 4), rev 2
added a **pre-registered confirmation task**: mechanical criteria (2 gold files,
direct import edge at base, churn ≤ 400, not used in any development — the 7 criteria
are committed in `scripts/select_confirmation_task.py`) scanned django's 200-commit
window newest-first and produced **exactly one candidate**, used unmodified:
[`67c40758`](https://github.com/django/django/commit/67c407585ccdc01b76d78e33c082f23d46346747)
*"Prevented writing control characters into XML attributes"* —
`core/serializers/xml_serializer.py` + `utils/xmlutils.py`, structurally the hero's
twin. Offline acceptance test verified fail@base / pass@gold. n=5/arm, frozen
pipeline, no tuning permitted.

Two results:

1. **The retrieval pipeline itself did not generalize.** The structure map came out
   `gold_recall = 0.5, graph added []` — the graph reach contributed **nothing**, on
   a task with a genuine 1-hop import edge. This is the same failure mode §5.3
   claimed to have fixed; the fix held on the task it was tuned on and not on the
   first untouched twin.
2. **Agent outcomes are a statistical tie on every axis** (n=5/arm, + = treatment
   better): time 🟢 +8.9% [−21,+30], tokens 🟠 +0.2% [−27,+21], tool calls 🔴 −8.9%
   [−41,+14], steps-to-first-edit 🟠 +5.6% [−59,+43], cost 🔴 −4.4% [−35,+18].
   Acceptance **1.00 in both arms**, recall **1.00 in both arms** (both gold files,
   every run).

A null replication does not prove the hero result false — but it means the hero
number must not be quoted as the expected value of the method. The defensible reading:
*on tasks Haiku can already navigate cleanly (both tasks here end at 100% acceptance),
the hint's payoff ranges from ~nothing to ~15%, and predicting which end you'll get
is not currently possible.*

### 3.5 Acceptance across the suite (new endpoint)

Behavior tests (each verified fail@base / pass@gold, all offline) now cover four
django tasks. Pass rates at n=5/arm:

| Task | control | treatment | Reading |
|---|---:|---:|---|
| django_cache (hero) | 1.00 | 1.00 | both arms always correct |
| django_xml_attrs (confirmation) | 1.00 | 1.00 | both arms always correct |
| django_signing_salt | 0.80 | 1.00 | small treatment edge, n too small to lean on |
| django_headersplit | 0.40 | 0.20 | **both arms mostly fail; treatment worse** — this is also the graph's worst efficiency task |
| django_quoting (vocab-gap probe) | 0.60 | 0.40 | non-saturated; no hint helped (anchors_only also 0.40) — see §3.9 |

Two implications. First, "the agent finished" ≠ "the feature works": on headersplit,
most runs in *both* arms ship a change that fails the behavior test — rev 1 had no way
to see this (§10, item 6). Second, there is no evidence the hint improves correctness;
where correctness varies at all, the differences are within run-to-run noise.

### 3.6 The 10-task suite at n=5/arm, with pre-registered statistics ⭐

All 9 django tasks were topped up from 3 to 5 runs/arm (the httpie continuity row
remains the original 5/arm Sonnet run — it is the one non-Haiku row, see §10 item 2).
The analysis was **pre-registered** in `course_lab/gv_stats.py` before the top-up
runs executed: primary endpoint = per-task % improvement in total time, estimator =
median across tasks, test = exact two-sided Wilcoxon signed-rank, α = 0.05.

Per-task (sorted by time improvement; + = graph better):

| Task | Label | Time | Steps to 1st edit | Tokens | Tool calls | Cost | Recall |
|---|---|:--|:--|:--|:--|:--|:--|
| django_b64_validate | unfavorable | 🟢 **+31%** | 🔴 −78% | 🟢 +43% | 🔴 −35% | 🟢 +21% | 🟠 ±0% |
| django_csp_nonce | unfavorable | 🟢 **+26%** | 🟢 +39% | 🟢 +26% | 🟢 +20% | 🟢 +24% | 🟢 +33% |
| django_cache | favorable | 🟢 **+18%** | 🟢 +17% | 🟢 +16% | 🟢 +17% | 🟢 +13% | 🟠 ±0% |
| django_alterfield | favorable | 🟢 +15% | 🔴 −44% | 🟢 +19% | 🔴 −9% | 🟠 −0% | 🟠 ±0% |
| django_compound_order | mixed | 🟢 +12% | 🟠 +3% | 🟢 +7% | 🟢 +10% | 🟢 +7% | 🔴 −20% |
| httpie_1292 (Sonnet) | favorable | 🟢 +10% | 🟢 +36% | 🟢 +3% | 🟢 +22% | 🟢 +12% | 🔴 −20% |
| django_signing_salt | mixed | 🟠 +2% | 🟢 +26% | 🟠 +2% | 🟠 +2% | 🟢 +6% | 🟢 +11% |
| django_admin_formset | favorable | 🔴 −7% | 🟢 +11% | 🔴 −4% | 🟢 +3% | 🟢 +3% | 🟢 +40% |
| django_admindocs_urls | favorable | 🔴 −10% | 🔴 −53% | 🔴 −34% | 🔴 −35% | 🔴 −39% | 🟢 +8% |
| django_headersplit | favorable | 🔴 **−38%** | 🔴 −40% | 🔴 −53% | 🔴 −52% | 🔴 −42% | 🔴 −33% |

Pooled, with the pre-registered test:

| Metric | Median | Trimmed mean | Mean | W/T/L | **Wilcoxon p** | Verdict at α=0.05 |
|---|:--|:--|:--|:--:|:--:|:--|
| **Total time (primary)** | 🟢 +11.0% | +8.3% | +5.9% | 7/0/3 | **0.232** | **not significant** |
| Steps to 1st correct edit | 🟢 +7.1% | −5.5% | −8.3% | 6/0/4 | 0.557 | not significant |
| Total tokens | 🟢 +4.7% | +4.2% | +2.3% | 7/0/3 | 0.557 | not significant |
| Tool calls | 🟢 +2.8% | −3.4% | −5.7% | 6/0/4 | 0.846 | not significant |
| Cost | 🟢 +6.4% | +2.8% | +0.5% | 7/1/2 | 0.492 | not significant |
| Gold recall | 🟠 +0.0% | +1.5% | +1.9% | 4/3/3 | 0.875 | not significant |

**What this honestly says.** Point estimates lean consistently positive on
time/tokens/cost, and the win/loss tally (7/10 on time) leans the right way — but
**at 10 tasks × 5 runs, none of it is statistically distinguishable from no effect**
(the primary endpoint's p = 0.232; a sign test on 7/10 gives p = 0.34). Rev 1's
"+22.5% median steps-to-first-edit" collapsed to +7.1% at n=5 (its trimmed mean is
*negative*), and two tasks flipped sign entirely between n=3 and n=5
(admindocs_urls +7→−10, signing_salt −24→+2) — run-to-run variance, not task
structure, dominates at this scale. The static favorability label still does not
predict outcomes (the two *unfavorable* tasks post the two biggest time wins; the
worst loser is *favorable*).

**Takeaway for practitioners (revised):** treat a relevant-file hint as a cheap
convenience with a modestly positive expected value that current evidence cannot
certify — not as a reliable per-task speedup, and on ~3 of 10 tasks it actively
misled navigation. Anyone quoting the +26%-style single-task numbers is quoting
selection, not expectation.

### 3.7 Same-model repo-size comparison (httpie on Haiku)

Rev 1 compared httpie-with-**Sonnet** against django-with-**Haiku** and called the
bigger django gains "direct evidence the graph's advantage grows with repo size" —
a model/repo confound (§10, item 5). Rev 2 re-ran the identical httpie task (same
base, same prompt, byte-identical structure map) with **Haiku, 5/arm**:

| Metric (+ = map better) | httpie **Sonnet** (rev 1) | httpie **Haiku** (new, 5/arm) | django hero **Haiku** (5/arm) |
|---|:--|:--|:--|
| Total time | 🟢 +10.1% | 🔴 −6.2% [−31,+13] | 🟢 +17.9% [+2,+32] |
| Steps to 1st correct edit | 🟢 +35.6% | 🟢 +25.5% | 🟢 +16.7% [−2,+31] |
| Total tokens | 🟢 +2.6% | 🟠 +2.9% [−9,+14] | 🟢 +16.3% [−8,+35] |
| Tool calls | 🟢 +22.0% | 🟢 +11.7% [−5,+27] | 🟢 +16.8% [−4,+32] |
| Cost | 🟢 +12.4% | 🟢 +7.3% [−5,+19] | 🟢 +12.7% [−5,+28] |
| Gold recall | 🔴 −20% | 🔴 −9.1% | 🟠 ±0% |
| Acceptance pass | 🟠 0.0 / 0.0 | 🟠 0.0 / 0.0 | 🟠 1.0 / 1.0 |

Held at the same model, the big-repo hero task still shows larger token/time gains
than the small-repo httpie task — **directionally consistent** with the
repo-size thesis — but every Haiku CI here spans zero, the "big repo" side is the
tuned hero task, and it is one task against one task. Rev 1's "direct evidence that
the graph's advantage grows with codebase size" is therefore downgraded to: *the
confound is removed; the hypothesis survives but is not established.* Two additional
observations: on httpie, **Haiku fails the behavior test in all 10 runs in both
arms** (as Sonnet did in rev 1) — a navigation hint does not rescue a capability
gap — and the treatment's wall-clock regressed (−6.2%) while its navigation metrics
improved, reinforcing the §3.1 duration-noise caveat.

### 3.9 The vocabulary-gap probe — testing the graph's home regime head-on ⭐

Everything up to here left the graph one defensible mechanism: *when a gold file
shares no vocabulary with the request, keyword search cannot find it even in
principle — only a structural edge can.* Revision 3 tests exactly that regime.

**Selection (mechanical, pre-registered — `scripts/select_vocab_gap_task.py`).**
Criteria V1–V6 were written before scanning; the vocabulary metric is the retrieval
stack's own tokenizer (`identifier_tokens` over node-id + enriched node text — the
exact text lexical anchoring scores). The django history window was deepened to
2,000 commits; **13 candidates** passed; per the pre-registered rule the **first
(newest)** was used unmodified: [`f05fac88`](https://github.com/django/django/commit/f05fac88c4699c6d04a8f1ac3328cf6c7bd39228)
*"Enforced quoting of all database object names"* — 4 gold files, churn 74.

**The certified gap** (committed check, `--verify-prompt`, re-run on the final
authored request AND the retrieval query):

| Gold file | Overlap with request | Overlap with retrieval query | Role |
|---|:--:|:--:|---|
| `db/models/expressions.py` | **0 tokens** | **0 tokens** | **hidden** — keyword search cannot rank it |
| `db/models/sql/compiler.py` | 4 | 2 | anchorable |
| `db/models/sql/datastructures.py` | 6 | 4 | anchorable |
| `db/backends/mysql/compiler.py` | 4 | 2 | anchorable |

The hidden file is one `import` hop from an anchorable one. An offline acceptance
test (`test_alias_quoting.py`, subquery-alias quoting behavior) was verified
fail@base / pass@gold. Arms: control / anchors_only / treatment, **5 runs each**,
frozen pipeline, no tuning.

**Manipulation check — the retrieval chain DELIVERED.** The frozen pipeline's map
came out `gold_recall = 1.0`: lexical anchors found the two `sql/*` files, and the
**Oracle PGQ graph walk added both keyword-invisible files** — `expressions.py` and
`mysql/compiler.py` — into the treatment map. By construction they could not have
arrived any other way (zero token overlap). This is the graph half of the system
working exactly as designed, on its home turf.

**Agent outcomes — the agent did not cash it in** (n=5/arm; + = arm better vs
control; bootstrap 95% CIs):

| Metric | control | treatment (map WITH hidden file) | anchors_only (map WITHOUT it) |
|---|:--|:--|:--|
| **Edited the hidden file** | **3 / 5** | 2 / 5 | 2 / 5 |
| **Acceptance pass** | **3 / 5** | 2 / 5 | 2 / 5 |
| Gold recall | 0.70 | 0.65 | 0.70 |
| Total time | 267.8 s | 🟠 +0.4% [−34,+33] | 🟢 +31.5% [+7,+61] |
| Total tokens | 4.11 M | 🟢 +16.2% [−2,+35] | 🟢 +40.4% [+21,+64] |
| Tool calls | 78.4 | 🟢 +11.7% [−9,+32] | 🟠 −0.8% [−53,+32] |
| Steps to 1st correct edit | 20.4 | 🟠 −2.9% [−44,+24] | 🟠 +2.9% [−31,+28] |
| Cost | $0.592 | 🟢 +10.4% [−11,+31] | 🔴 −12.7% [−90,+33] |

And the graph's pure marginal — **treatment vs anchors_only** — is null-to-negative:
time −45% [−165,+10], tokens −41% [−137,+5], with identical hidden-file edit rates
and identical acceptance.

**Three findings:**

1. **Naming the keyword-invisible file in-context did not make the agent edit it.**
   2/5 with the map vs 2/5 without it — the treatment's unique content went unused
   more often than not, replicating the §3.1 pattern in the one regime where the
   content was genuinely unavailable elsewhere.
2. **The bare agent recovered the "unreachable" file most often (3/5) — by reading
   code.** The vocabulary gap is real between the *request* and the file, but an
   agent is not a one-shot retriever: after opening `sql/compiler.py` it greps for
   the implementation identifiers it just learned (`quote_name_unless_alias`), which
   *do* appear in the hidden file. **Agents bootstrap their own query vocabulary from
   intermediate reads**, so a request-level vocabulary gap does not bind an agent the
   way it binds single-shot retrieval. This is the probe's real lesson, and it
   reframes what code-graph hints are for: they compress the agent's *first few
   exploration steps*, they do not unlock files the agent couldn't reach.
3. **Correctness was finally non-saturated (unlike the hero) — and no hint helped.**
   Control passed the behavior test most often (3/5 vs 2/5 in both hinted arms; all
   within noise at n=5, but no arm shows the graph rescuing anything). The
   anchors-only arm's big time win (+31.5%) came with the *worst-equal* correctness —
   faster partly by being less thorough.

**Verdict on the open question.** In the regime constructed to maximize the graph's
advantage — hidden dependency, certified zero lexical overlap, map verified to
contain it — the graph-bearing hint produced **no measurable agent-level benefit
over the anchor list, and no benefit over no hint at all** on coverage or
correctness. Combined with §3.1/§3.4, the consistent picture across all four
experiments: *the value of these hints, where it exists, comes from handing the
agent a relevant starting point quickly; the structural half's unique content is
either redundant with what the agent discovers itself, or ignored.* (Usual caveat:
n=5/arm, one probe task, one agent/model family — a stronger agent might use the
hint better; a weaker one might depend on it more.)

### 3.10 Cost of the whole study

Rev 1 pilot ≈ $2.71; rev 1 suite ≈ $36 (54 Haiku runs + the pre-existing 10 Sonnet
httpie runs). Rev 2 added ≈ $28: hero top-up + two ablation arms (14 runs),
confirmation task (10), suite top-up (36), httpie/Haiku (10 + 3 re-runs after a
session-limit interruption). Rev 3 added ≈ $8: the vocabulary-gap probe (15 runs,
3 arms — this task runs long; most runs hit the 80-turn budget). All runs,
transcripts, and scores are committed or regenerable; scoring corrections were
applied by replaying recorded traces, never by re-running paid arms.

---

## 4. Metric definitions

Unchanged from rev 1 (improvement % signed so + = better; 🟢 >+1%, 🟠 ±1%, 🔴 <−1%),
with two upgrades:

- **Acceptance pass** — an offline behavior pytest, verified fail@base / pass@gold,
  now exists for 4 of the 9 django tasks + httpie. Where present it is the
  correctness endpoint; "total time" for those tasks now means *time to a verified
  outcome*, not time to a self-declared stop.
- **Duration caveat** — wall-clock absorbs API latency/server load; the placebo arm's
  nominal time win despite worse navigation (§3.1) is the concrete demonstration.
  Mechanism claims should rest on steps/tokens/calls.

---

## 5. What was implemented (the engineering behind these numbers)

### 5.1 The Module-3 structure-map builder (pre-existing)

Unchanged: parse the tree (AST `import`/`call` + git `co_edit`), load into **Oracle AI
Database** as a property graph, walk neighborhoods **in-DB via SQL/PGQ
`GRAPH_TABLE … MATCH`**, rank in Python, render a compact markdown map
(`course_lab/graphify_verify.py`).

### 5.2 The A/B harness (extended in rev 2)

`course_lab/gv_harness.py` + `graphify_verification/scripts/run.py`, now with:
four arms (`--arms control,treatment,anchors_only,random_files`), resumable runs
(completed run indices are skipped, so n top-ups only pay for new runs), per-task
acceptance interpreters, and an offline **replay-rescorer**
(`scripts/rescore.py`) that recomputes retrieval metrics against a corrected gold set
and back-fills acceptance by replaying recorded edits (fidelity-gated; 100% of
replays matched their run's recorded diff). Statistics live in
`course_lab/gv_stats.py` (exact Wilcoxon/sign tests, bootstrap CIs — pre-registered
estimator and α).

### 5.3 The ranking fix — full disclosure (revised)

Rev 1 presented "signal-ranked graph reach" as a straightforward improvement: on
django's hub-dense tree, co-edit noise crowded out the one import edge that mattered,
and ranking reach by edge-kind + anchor relevance brought the hero map's gold recall
from 0.5 to 1.0. Two things must be said plainly in rev 2:

1. **The fix was developed and validated on the hero task's own gold labels.** That
   makes the hero task a development set, and its headline delta an optimistic
   estimate — which is exactly why the confirmation task exists (§3.4).
2. **Under the corrected 4-file gold set, the tuned map's recall is 0.5, not 1.0**
   (it surfaces `utils/cache.py` + `utils/http.py`; it never surfaced either
   middleware file). And on the confirmation twin, the same pipeline yielded
   `graph added []`. The fix, as shipped, is real but narrow: it de-noises co-edit
   flooding when the lexical anchor lands on the right entry file, and it does not
   yet generalize into a reliable reach mechanism.

### 5.4 Where Oracle AI Database (SQL/PGQ) actually does the work (revised)

The infrastructure claims stand: the graph is real (12k nodes / 425k edges), the
traversal genuinely runs in-database via `CREATE PROPERTY GRAPH` + `GRAPH_TABLE
MATCH`, it scales far past what fits in a prompt, updates incrementally by `MERGE`,
and hands the agent a tiny distilled hint. Build-time decoupling (Oracle builds the
hint; the agent runs with it) remains the right design:

```
  build-graph stage                                    run stage
  ─────────────────                                    ─────────
  parse repo (AST + git)                               read structure_map.json
        │                                                    │
        ▼                                                    ▼
  load nodes/edges into Oracle  ── CREATE PROPERTY GRAPH     inject as
        │                              over MEMORY_GRAPH_*        --append-system-prompt
        ▼                                                    │
  ★ Oracle SQL/PGQ walks the graph IN THE DATABASE:          ▼
     GRAPH_TABLE (mem_code_graph                         Claude Code (Haiku)
       MATCH (v) -[e]-> (w)  domain-scoped)              navigates + edits — never
        │  returns import/call/co_edit neighbors            touches Oracle
        ▼
  rank (kind + anchor) → render markdown → structure_map.json
```

**What Oracle is doing in every experiment in this report, concretely:**

- **During `build-graph`** the log prints `structure map source=oracle-pgq` and
  `loaded into Oracle PGQ: {domain: gv_<task>, n_nodes: ~12k, n_edges: ~400k}`.
  The neighbor lookups that produce every map's "graph-reached" section are real
  `GRAPH_TABLE (mem_code_graph MATCH (v)-[e]->(w) …)` queries executed by the
  database (both edge directions), over a property graph created with
  `CREATE PROPERTY GRAPH` — the same SQL/PGQ Module 3 teaches, at 400k-edge scale.
  Python only ranks what the database returns.
- **It scales past what fits in a prompt.** You cannot paste a 400k-edge graph
  into an agent's context, and grepping it live would burn the agent's turns.
  Oracle stores the whole thing durably and answers "what are the typed neighbors
  of this file?" in one indexed query; the agent receives a *tiny* distilled hint.
- **It's persistent and updatable (the continual-learning angle).** The property
  graph is keyed by `(id, domain)` — each task in this study lives in its own
  domain (`gv_django_cache`, `gv_django_quoting`, …) in ONE database, and a new
  commit is a `MERGE` of a few nodes/edges, no reprocessing.
- **In the vocabulary-gap probe (§3.9) Oracle's walk is load-bearing by
  construction**: the hidden gold file shares zero tokens with the request, so it
  cannot enter the map through the lexical half at all — when it appears in the
  treatment map, it got there through the in-DB PGQ neighbor expansion and
  nothing else. **And it did appear** — the probe's map scored gold recall 1.0
  with both keyword-invisible files delivered by the PGQ walk. The retrieval
  layer's contract was met in full; what §3.9 shows is that the *agent* did not
  convert that delivery into edits or correctness on this task. Precise split of
  credit: **Oracle proven as the scalable structure store and walker; the
  hint-consumption layer (how an agent uses structural context) is where the
  open problem now lives.**

The **causal** claim in rev 1 — "the 🟢 +26% win is *produced by* the Oracle PGQ
traversal" — is **withdrawn**. The ablation evidence (§3.1) attributes the hero
efficiency win to the *lexical anchor list*; the graph-reach content added no
measurable value over the anchors on that task, and none at all on the confirmation
task (where reach was empty). What survives, precisely stated: **Oracle PGQ is the
engine that makes structure-derived hints buildable and maintainable at repo scale;
whether the structural half of those hints beats a plain keyword list is, on current
evidence, unproven for this agent and task family.** The starter experiment for
anyone extending this study is now obvious: find task families where the gold files
genuinely share no vocabulary with the request — the regime where anchors *can't*
work and reach is the only path.

---

## 6. Reproduce it — manually and with the harness

### 6.1 Prerequisites

- Repo working dir: `/home/ubuntu/personal/dl-ai-continual-learning`
- `claude` CLI logged in; `haiku` reachable (`claude -p "say OK" --model haiku`).
- **Oracle AI Database up** (the `dlai-oracle-free` container,
  `localhost:1521/FREEPDB1`): `python lab.py bootstrap-oracle`, then load
  credentials — `set -a && . ./.env && set +a` (sets `DLAICL_ORACLE_LIVE=1`).
- `uv` for the Python env (`uv sync`).
- Acceptance tests for django tasks need a small venv:
  `uv venv /tmp/graphify-verification-scratch/django-venv && uv pip install
  --python /tmp/graphify-verification-scratch/django-venv/bin/python pytest
  asgiref sqlparse`.

### 6.2 Fully manual (two terminals, no harness) — see the difference yourself

This is the raw A/B, the thing the harness merely repeats N times and scores.
Terminal 1 = control, terminal 2 = treatment; same repo, same prompt, same model.

```bash
# once: build the map for the task (Oracle walks the graph, writes the JSON)
uv run python graphify_verification/scripts/run.py build-graph \
    --config graphify_verification/configs/django_cache_control.yaml
MAP=$(uv run python -c "import json;print(json.load(open(
  'course_lab/data/graphify_verification_structure_map_django_cache.json'))['markdown'])")

# fresh checkout per run (worktrees keep the arms isolated)
git -C /tmp/graphify-verification-scratch/django worktree add /tmp/ctrl 2c5e4af3cc
git -C /tmp/graphify-verification-scratch/django worktree add /tmp/treat 2c5e4af3cc
FEATURE="$(uv run python -c "import yaml;print(yaml.safe_load(open(
  'graphify_verification/configs/django_cache_control.yaml'))['feature'])")"

# Terminal 1 — CONTROL (bare repo):
cd /tmp/ctrl && claude -p "$FEATURE" --model haiku \
  --output-format stream-json --verbose --permission-mode bypassPermissions \
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' --max-turns 80

# Terminal 2 — TREATMENT (same + the map):
cd /tmp/treat && claude -p "$FEATURE" --model haiku \
  --output-format stream-json --verbose --permission-mode bypassPermissions \
  --strict-mcp-config --mcp-config '{"mcpServers":{}}' --max-turns 80 \
  --append-system-prompt "$MAP"
```

Watch the control agent grep around; watch the treatment agent open the mapped
files sooner. **One pair of runs is a demo, not a measurement** — run-to-run
variance at this scale is larger than the mean effect (§3.2), which is exactly
why the harness exists. Quote the n=5 tables, show the single pair on screen.

### 6.3 With the harness (the actual measurement)

```bash
cd /home/ubuntu/personal/dl-ai-continual-learning
set -a && . ./.env && set +a

CFG=graphify_verification/configs/django_cache_control.yaml

# 1) Build the graph + map (Oracle PGQ; prints map gold_recall honestly)
uv run python graphify_verification/scripts/run.py build-graph --config $CFG

# 2) Run all four arms, 5 runs each (resumable — completed indices are skipped)
uv run python graphify_verification/scripts/run.py run --config $CFG \
    --arms control,treatment,anchors_only,random_files --runs 5

# 3) Re-score offline (corrected gold, acceptance replay) + report
uv run python graphify_verification/scripts/rescore.py --config $CFG --replay-acceptance
uv run python graphify_verification/scripts/run.py report --config $CFG

# Confirmation task (frozen pipeline, no tuning):
uv run python graphify_verification/scripts/run.py all \
    --config graphify_verification/configs/django_xml_attrs.yaml

# Suite + pre-registered stats:
uv run python graphify_verification/scripts/run_suite.py --skip-build
```

### 6.4 Where the artifacts live

- Configs: `graphify_verification/configs/{django_cache_control,django_xml_attrs,django_quoting,httpie_1292_haiku}.yaml` + `configs/suite/*.yaml`
- Selection audits: `graphify_verification/scripts/select_confirmation_task.py` (twin task) and `select_vocab_gap_task.py` (vocab-gap probe, incl. the `--verify-prompt` overlap certifier)
- Acceptance tests: `graphify_verification/acceptance/test_{cache_control_directives,xml_attrs_control_chars,header_value_split,signed_cookie_salt,alias_quoting,top_level_arrays}.py`
- Results JSONs: `course_lab/data/graphify_verification_results*.json`, suite pool in `graphify_verification_suite_summary.json` (now includes per-metric Wilcoxon/sign-test p-values and median CIs)
- Per-run traces: `data/graphify_verification_runs*/` (stream-json + result per run)
- Stats: `course_lab/gv_stats.py` (+ `tests/test_gv_stats.py` validating exact p-values)

---

## 7. Links

- Hero task: django commit [`b461519bf5`](https://github.com/django/django/commit/b461519bf5973d7fc149560d2f99acdba71a437d) (CVE-2026-35193); base `2c5e4af3cc`.
- Confirmation task: django commit [`67c40758`](https://github.com/django/django/commit/67c407585ccdc01b76d78e33c082f23d46346747) (Fixed #37183); base `9dce456105`.
- Vocabulary-gap probe: django commit [`f05fac88`](https://github.com/django/django/commit/f05fac88c4699c6d04a8f1ac3328cf6c7bd39228) (Fixed #36795); base `4b2b4bf0ac`.
- Small-repo comparison: httpie/cli PR [#1292](https://github.com/httpie/cli/pull/1292).
- Course module: `module_3_structure_aware_retrieval/`; experiment subsystem: `graphify_verification/` + `course_lab/{graphify_verify,gv_harness,gv_score,gv_stats,gv_lab}.py`.

---

## 8. One-paragraph summary (for a caption / description)

> On the full Django codebase we ran Claude Code (Haiku) against real merged changes
> under four conditions: bare repo, the Module-3 graph-derived structure map, the
> map's lexical anchors alone, and a random-file placebo. A relevant hint made the
> agent ~13–18% more efficient on the hero task with identical correctness (every
> arm passed the behavior test), **but the anchors alone did as well as the full
> graph map, the placebo actively misled navigation, the effect did not replicate on
> a pre-registered twin task, and across a 10-task suite no metric beat noise**
> (Wilcoxon p ≥ 0.23, n=5/arm). The final experiment — a mechanically-selected task
> where one gold file shares **zero** vocabulary with the request — showed the full
> chain's split verdict: **Oracle's SQL/PGQ graph walk correctly delivered the
> keyword-invisible file into the map, and the agent still didn't edit it more often
> than without the map** (the bare agent found it anyway by reading code and grepping
> the identifiers it learned). Structure-derived hints compress an agent's first
> exploration steps; on this evidence they do not unlock anything a competent agent
> can't reach — the open problem has moved from *building* structural context (solved,
> in-database, at 425k-edge scale) to *getting agents to use it*.

---

## 9. Recording the "Claude Code in action" lesson (shot-by-shot guide)

Goal: a ~4-minute screen recording where the viewer *sees* the with-vs-without
difference on a real repo, and hears the honest framing. Everything below uses the
hero task (`django_cache`) because it is the study's green case — record it as the
*demo*, and quote the n=5 tables (§3.1) as the *measurement*.

### 9.1 Setup for the recording

- **Recorder:** `asciinema rec demo.cast` for a pure-terminal capture (or OBS at
  1920×1080 for YouTube). Terminal font 18–22pt so tool calls read on a phone.
  Two side-by-side panes (tmux `split-window -h`) — control left, treatment right.
- **Prep off-camera:** Oracle up + `.env` loaded (§6.1); `build-graph` already run;
  the `$MAP` and `$FEATURE` variables from §6.2 exported in both panes; two fresh
  worktrees created. Nothing else — the runs themselves are the show.

### 9.2 The shot list

1. **Cold open (~30 s) — Oracle builds the hint.** Run `build-graph` on camera and
   point at two log lines: `loaded into Oracle PGQ: {n_nodes: 11987, n_edges:
   424728}` and `structure map source=oracle-pgq`. Say the one-liner: *"Oracle
   walks a four-hundred-thousand-edge property graph in the database and distills
   it into this one-paragraph map; the agent never talks to Oracle — it just gets
   the map."*
2. **Show the map (~20 s).** `cat` the map markdown. Point at the two sections:
   keyword-matched entry points (what grep would find anyway) and graph-reached
   dependencies (what it wouldn't). *"This is the only difference between the two
   panes."*
3. **The A/B (~2 min, time-lapse the middle).** Start the §6.2 control command in
   the left pane and the treatment command in the right pane simultaneously.
   Narrate the visible difference: left pane greps and reads; right pane opens the
   mapped files within its first few tool calls. Cut to 4–8× speed after ~20 s.
4. **The receipts (~40 s).** Show §3.1's table (or run
   `run.py report --config …django_cache_control.yaml`). Read the treatment column
   out loud: *"across five runs per arm: 18% less wall-clock, 16% fewer tokens,
   and — the part most demos skip — every run in every arm passed the behavior
   test, so this is purely about how directly the agent navigates."*
5. **The honest close (~30 s), verbatim-usable:** *"Two caveats that make this
   credible. One: a single pair of runs like you just watched proves nothing —
   run-to-run variance is larger than the effect; the numbers come from the
   five-run tables. Two: when we gave the agent just the keyword-matched file
   list with no graph content at all, it did about as well — so on this task the
   win comes from being handed a relevant file list. The regime where the graph
   itself must carry the win — a gold file that shares zero vocabulary with the
   request — is the vocabulary-gap probe in the written study."* Then show §3.9's
   outcome and let it land either way.

### 9.3 Rules that keep the recording honest

- **Never quote the single recorded pair as the result** — it's the visual; the
  n=5 tables are the claim. If your recorded pair happens to go the wrong way
  (it can — §3.2), say so and keep it; it demonstrates exactly why the study
  runs five per arm.
- **Anchors-only is the fair baseline to mention** whenever the graph is credited.
- **Suite numbers always travel with their uncertainty:** "faster on 7 of 10
  tasks, median +11%, not statistically significant at this sample size."

Learner setup lives in `reports/claude-practical-setup.md` (four-arm harness,
acceptance venv, replay-rescorer, pre-registered statistics).

---

## 10. Correction log (rev 1 → rev 2)

Everything below was found in an external methods review of rev 1 plus the follow-up
work it triggered. Original numbers are preserved here for the record.

1. **Wrong ground truth on the hero task (scoring bug).** Rev 1's gold set listed
   `utils/cache.py` + `utils/http.py`; the real PR also changed
   `middleware/cache.py` + `middleware/http.py` — files every run in both arms
   edited, which were then scored as precision errors. Corrected everywhere;
   effects: recall 0.50→0.75 (both arms), precision 0.33→1.0 (both arms),
   steps-to-first-edit recomputed, and rev 1's §3.3 claim that the middleware
   files "aren't in the gold set" retracted. An audit confirmed the other 9 task
   gold sets were correct. The tuned map's "gold recall 1.0" becomes **0.5** under
   the corrected set (it never surfaced either middleware file).
2. **"64 Haiku runs, 3 runs/arm" (§3.6) was wrong.** 10 of the 64 were the
   pre-existing httpie **Sonnet** runs at **5**/arm. The pooled suite remains
   mixed-model on that one row; it is now labeled in the table.
3. **"Every treatment run used fewer tool calls than every control run" (§3.2) was
   false** (treatment run 2 used 48; control run 1 used 45), and "the distributions
   don't overlap" held only for steps-to-first-edit at n=3. Rev 2 prints the full
   per-run table instead.
4. **The §5.3 ranking fix was tuned on the hero task's own gold labels** — rev 1
   said this obliquely; rev 2 states it as a limitation, and adds the pre-registered
   confirmation task (§3.4), where the frozen pipeline's graph reach came out empty
   and the hero effect did not replicate.
5. **§3.4's repo-size claim was confounded** (model changed with repo). Replaced by
   the same-model comparison in §3.7.
6. **No correctness endpoint existed** — "faster to done" meant faster to a
   self-declared stop. Rev 2 adds verified acceptance tests for 4 django tasks
   (back-filled onto paid runs by fidelity-gated replay) and reports them (§3.5).
7. **No significance testing existed.** Rev 2 pre-registers estimator + test
   (`gv_stats.py`), reports exact Wilcoxon/sign-test p-values and bootstrap CIs
   throughout, and — because none clear α=0.05 — words the suite conclusion as
   "consistent with no effect" rather than "a real, repeated win" (rev 1's phrase).
8. **Hero arms extended 3→5 runs/arm** (+ two new ablation arms). The hero deltas
   moved: time +26.4→+17.9%, steps +31.9→+16.7%, tokens +25.0→+16.3%, cost
   +19.7→+12.7% — regression toward the mean, disclosed rather than averaged away.
