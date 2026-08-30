# Setup guide — run the code-graph retrieval demo locally

Companion to the "Claude Code in action" video lesson. This takes you from the Module-3
notebook to **actually running the A/B on your own machine**: build a knowledge graph of
a real repo, then have a coding agent implement a feature *with* and *without* that graph
as a navigation hint. No narration needed in the video — open this and follow along.

Everything here is what we used to produce the django/Haiku study in
`reports/graphify-m3-django-haiku-pilot.md`.

---

## 1. What you need

| Piece | Why | How |
|---|---|---|
| **Oracle AI Database (Free)** | stores the code graph as a property graph and walks it in-DB via SQL/PGQ | Docker container (below) — the `dlai-oracle-free` image, port 1521 |
| **`uv`** (Astral) | the Python env for this repo | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **A coding-agent CLI** | the agent under test | Claude Code (`claude`), or Codex / OpenCode — anything you can pass a system-prompt hint |
| **A target repo** | the codebase to graph + edit | any real repo on disk (e.g. an e-commerce app) |

You do **not** need a GPU for this demo — the graph + PageRank math is CPU-only; Oracle
is the durable store. (GPUs are only for the course's model-training modules.)

---

## 2. Bring up Oracle AI Database

```bash
# From the repo root. Stands up an Oracle Free container + the course DDL.
python lab.py bootstrap-oracle
```

This starts Oracle listening on `localhost:1521/FREEPDB1`. If you manage Oracle
yourself, any Oracle 23ai+/26ai instance works — you just need a user that can
`CREATE PROPERTY GRAPH`.

Then put the connection details in a `.env` file at the repo root:

```bash
# .env  (the harness loads these; DO NOT commit real secrets)
ORACLE_MEMORY_DB_USER=<your_user>
ORACLE_MEMORY_DB_PASSWORD=<your_password>
ORACLE_MEMORY_DB_CONNECT_STRING=localhost:1521/FREEPDB1
DLAICL_ORACLE_LIVE=1
```

Load them into your shell before running anything:

```bash
set -a && . ./.env && set +a
```

Sanity-check the DB is reachable: `python lab.py bootstrap-oracle` is idempotent, so
re-running it should report the container is already up.

---

## 3. Install the Python env

```bash
uv sync          # creates .venv from pyproject.toml / uv.lock
```

Everything below runs through `uv run …` so it uses that env.

---

## 4. Make sure your agent CLI works

```bash
# Claude Code example (swap for codex / opencode as you like):
claude -p "reply with the single word OK" --model haiku
```

If that prints `OK`, the agent is wired up. (Codex note: on some managed/enterprise
accounts, headless `codex exec` can't execute shell tools — if so, use Claude Code or a
local model; see the report's driver notes.)

---

## 5. Build the knowledge graph for your repo

The graph is: **files + functions/classes** as nodes, with typed edges —
`import` (file→file, from AST), `call` (function→function, from AST), and `co_edit`
(files changed together, from `git log`). It's loaded into Oracle as a property graph.

Two ways:

**(a) Use the experiment harness (what the study used).** Point a config at your repo and
run the `build-graph` stage — it clones/parses the repo, loads it into Oracle, and
renders the structure map:

```bash
# Copy an existing config and edit repo_url / base_sha / gold_files / feature.
cp graphify_verification/configs/django_cache_control.yaml \
   graphify_verification/configs/my_repo.yaml
# ...edit it...
uv run python graphify_verification/scripts/run.py build-graph \
  --config graphify_verification/configs/my_repo.yaml
```

Watch for the line `structure map source=oracle-pgq … loaded into Oracle PGQ {n_nodes,
n_edges}` — that confirms Oracle walked the graph in-database. It writes the map to
`course_lab/data/graphify_verification_structure_map_<task>.json` (the `markdown` field
is the hint you'll paste into the agent).

**(b) Parse any tree directly** (no experiment scaffolding) — the same Module-3 code:

```python
uv run python -c "
from course_lab.code_graph_parse import parse_tree
g = parse_tree('/path/to/your/repo/src/package', pkg_root='package')
print(g['n_files'], 'files;', len(g['edges']), 'edges')
"
```

---

## 6. Run the A/B (with vs without the graph)

The only difference between the two runs is whether the structure map is in the agent's
context. Same repo checkout, same feature prompt, same model.

```bash
# CONTROL — bare repo, no hint:
cd /path/to/fresh/checkout
claude -p "Add customer star-ratings to products. Edit the files needed; keep it idiomatic." \
  --model haiku --output-format stream-json --verbose

# TREATMENT — same, but inject the structure map as a system-prompt hint:
MAP=$(uv run python -c "import json;print(json.load(open('course_lab/data/graphify_verification_structure_map_my_repo.json'))['markdown'])")
cd /path/to/fresh/checkout   # a clean copy!
claude -p "Add customer star-ratings to products. Edit the files needed; keep it idiomatic." \
  --model haiku --output-format stream-json --verbose \
  --append-system-prompt "$MAP"
```

Use a **fresh checkout for each run** (a `git worktree` at a fixed base is cleanest) so
the runs don't contaminate each other.

The full experiment harness automates this (N runs per arm, isolated worktrees, transcript
parsing, scoring) — for a rigorous measurement rather than a demo:

```bash
uv run python graphify_verification/scripts/run.py run    --config graphify_verification/configs/my_repo.yaml
uv run python graphify_verification/scripts/run.py report --config graphify_verification/configs/my_repo.yaml
```

**Control arms that make the result mean something.** Besides `control` and
`treatment`, the harness supports two ablation arms (run them with
`--arms anchors_only,random_files`):

- `anchors_only` — the map's lexical anchors with ALL graph information
  stripped. If this arm matches `treatment`, your win comes from pre-computing
  the keyword hits, not from the graph.
- `random_files` — the same map template filled with seeded-random irrelevant
  files. A placebo: it separates "any official-looking hint changes behavior"
  from "the hint's content matters".

Re-running `run` is resumable: per-run results are kept on disk and completed
run indices are skipped, so topping up `n_runs` only pays for the new runs.

**Behavior acceptance tests.** A task config can name an offline pytest
(`acceptance_test:` + `test_python:`, see `graphify_verification/acceptance/`)
that must FAIL at the base commit and PASS at the gold commit — that gives the
study a correctness endpoint, not just efficiency. For runs that were paid for
before a test existed, `scripts/rescore.py --replay-acceptance` replays each
run's recorded edits into a fresh worktree (with a fidelity gate: the replayed
diff must match the run's recorded file set exactly) and back-fills the verdict.

**Statistics.** `run_suite.py` pools per-task results with a pre-registered
analysis (`course_lab/gv_stats.py`): median across tasks + exact two-sided
Wilcoxon signed-rank, with sign test and bootstrap CIs as secondary. If the
p-value doesn't clear 0.05, say "consistent with no effect" — don't headline
the median alone.

---

## 7. Read the numbers

The `report` stage prints control vs treatment for the metrics defined in the study:
**total time on task, tool calls to first correct edit, tool calls, tokens, cost,** and
**gold-file recall**. Positive improvement = the graph helped. Expect a *net* win, not a
guaranteed one — see the 10-task suite in the report for the honest spread.

---

## 8. Teardown

```bash
python lab.py teardown-oracle    # stop + remove the Oracle container
```

---

## Troubleshooting

- **`ORA-12899: value too large for column "…"."DOMAIN"`** — your Oracle graph `domain`
  name is >32 chars. Keep it short (e.g. `gv_my_repo`).
- **`structure map source=in-memory`** instead of `oracle-pgq` — Oracle wasn't reachable;
  the build fell back to an in-memory walk. Check `.env` and that the container is up.
- **Agent edits the wrong files even with the map** — the map is a *hint*, not a
  constraint; on some tasks lexical anchoring lands on the wrong entry files (the report's
  §3.6 counter-cases). That's expected; the graph is a net, not uniform, win.
