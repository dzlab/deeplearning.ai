"""course_lab/coding_trace_synth.py: synthesise a believable coding-agent
trace fixture for the Module 2 notebook.

Why synthetic: the grounding comes from REAL agent traces, but their *content*
is private and domain-specific. So we derive only the DISTRIBUTIONS from the
real traces (see ``REAL_TRACE_STATS`` below, computed offline from 51 real
episodes on 2026-06-18 via ``lifeos_trace_extract.compute_distributions``) and
author fully synthetic, generic coding-agent episodes that match those shapes.
No real content survives: this is "real shape, synthetic surface".

The fixture seeds the six trace types the notebook teaches (conversation, tool
calls, skills used, workflows, preferences, errors-and-fixes) across five
everyday coding topics. ``run_test_suite`` carries a recurring "bare pytest in
the wrong environment" error across day 1 and day 2: that is the weakness the
dream loop's induced skill (v2) learns to avoid.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

# ── Real grounding (derived offline; do not edit by hand) ──────────────────────
# Computed from 51 real agent episodes (lifeos_trace_extract.compute_distributions,
# 2026-06-18). Real "errors" counts were inflated by follow-up to-dos, so for
# synthetic coding episodes we use 0-3 genuine error+fix pairs instead.
REAL_TRACE_STATS = {
    "n_real_episodes": 51,
    "tools_per_episode": {"min": 1, "median": 5, "mean": 9, "max": 15},
    "skills_per_episode": {"median": 1, "mean": 1, "max": 3},
    "workflow_steps": {"median": 7, "mean": 6, "max": 10},
    "preferences_per_episode": {"median": 4, "mean": 4, "max": 6},
    "errors_per_episode": {"median": 1, "mean": 1, "max": 3},
    "frac_with_error": 0.7,
    "frac_with_skill": 0.6,
}

# ── Vocab pools ────────────────────────────────────────────────────────────────
CODING_TOOLS = [
    "read_file", "write_file", "edit_file", "run_tests", "run_lint",
    "run_typecheck", "grep", "glob", "git_diff", "git_commit", "run_build",
    "install_deps", "open_pr",
]

CODING_SKILLS = [
    "brainstorming", "writing-plan-spec", "executing-plans",
    "using-git-worktrees", "subagent-driven-development",
    "test-driven-development", "systematic-debugging",
    "requesting-code-review",
]

CODING_PREFERENCES = [
    "UK English in comments and docs",
    "no em dashes in prose",
    "conventional commit messages",
    "type hints on public functions",
    "pytest, not unittest",
    "small, frequently committed changes",
    "match the surrounding code style",
    "table-driven tests where it helps",
]

TOPICS = [
    "setup_dev_env", "run_test_suite", "add_api_endpoint",
    "fix_failing_build", "refactor_module",
]


def _tc(tool: str, args: str, result: str, ok: bool = True) -> dict:
    return {"tool": tool, "args": args, "result": result, "ok": ok}


def _turn(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ── Authored scenes ────────────────────────────────────────────────────────────
# Each scene is one believable episode body for its topic. Counts (tools,
# workflow length) sit near the real medians; some scenes carry no skill and no
# error so the fixture honours frac_with_skill / frac_with_error.
SCENES: dict[str, list[dict]] = {
    "setup_dev_env": [
        {
            "day": 1,
            "conversation": [
                _turn("user", "Get the project running locally so I can start on the ticket."),
                _turn("assistant", "Reading the README and the Makefile to find the documented setup path."),
                _turn("assistant", "Created the virtualenv, installed the locked dependencies, and the dev server starts on the default port."),
            ],
            "tool_calls": [
                _tc("read_file", "README.md", "found a 'Getting started' section"),
                _tc("read_file", "Makefile", "target: make setup, make dev"),
                _tc("install_deps", "make setup", "installed 41 packages from the lockfile"),
                _tc("run_build", "make dev", "server listening on :8000"),
            ],
            "skills_used": ["executing-plans"],
            "errors": [],
            "workflow": [
                "read the README", "read the Makefile", "create the virtualenv",
                "install locked dependencies", "start the dev server",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "New laptop, set up the repo from scratch."),
                _turn("assistant", "Cloning and following the documented bootstrap."),
                _turn("assistant", "Bootstrap done; smoke test passes."),
            ],
            "tool_calls": [
                _tc("read_file", "docs/CONTRIBUTING.md", "bootstrap steps listed"),
                _tc("install_deps", "make setup", "ok"),
                _tc("run_tests", "make test -- -k smoke", "1 passed"),
            ],
            "skills_used": [],
            "errors": [],
            "workflow": ["clone the repo", "read CONTRIBUTING", "run make setup", "run the smoke test"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Set up the project on the CI image; it behaves differently there."),
                _turn("assistant", "The image is missing a system library the build needs; installing it then re-running setup."),
                _turn("assistant", "Added the apt step to the bootstrap docs so the next person does not hit this."),
            ],
            "tool_calls": [
                _tc("install_deps", "make setup", "error: pg_config not found", False),
                _tc("install_deps", "apt-get install libpq-dev", "ok"),
                _tc("install_deps", "make setup", "ok"),
                _tc("edit_file", "docs/CONTRIBUTING.md: note libpq-dev", "1 hunk applied"),
                _tc("git_commit", "docs: note libpq-dev for setup", "committed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "make setup failed: pg_config not found (missing system library)",
                 "fix": "install libpq-dev before make setup; documented it in CONTRIBUTING"},
            ],
            "workflow": [
                "run make setup", "read the failure", "install the missing system library",
                "re-run setup", "document the prerequisite", "commit",
            ],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Setup runs but the app crashes on boot."),
                _turn("assistant", "It needs an env var that is not in the example file; the crash is a missing DATABASE_URL."),
                _turn("assistant", "Added it to .env.example with a comment; app boots."),
            ],
            "tool_calls": [
                _tc("run_build", "make dev", "KeyError: 'DATABASE_URL'", False),
                _tc("read_file", ".env.example", "DATABASE_URL not listed"),
                _tc("edit_file", ".env.example: add DATABASE_URL", "1 hunk applied"),
                _tc("run_build", "make dev", "server listening on :8000"),
            ],
            "skills_used": [],
            "errors": [
                {"error": "app crashed on boot: required env var DATABASE_URL was missing and undocumented",
                 "fix": "add required env vars to .env.example with a comment so setup is complete"},
            ],
            "workflow": ["start the app", "read the crash", "find the missing env var", "document it", "reboot"],
        },
    ],
    "run_test_suite": [
        {
            "day": 1,
            "conversation": [
                _turn("user", "Run the tests and tell me if main is green."),
                _turn("assistant", "Using the project's test command from the Makefile."),
                _turn("assistant", "All green: 212 passed."),
            ],
            "tool_calls": [
                _tc("read_file", "Makefile", "target: make test"),
                _tc("run_tests", "make test", "212 passed in 38s"),
            ],
            "skills_used": [],
            "errors": [],
            "workflow": ["find the project test command", "run make test", "report the result"],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Check the suite before I open the PR."),
                _turn("assistant", "Running the documented test command and the linter."),
                _turn("assistant", "Suite and lint both clean."),
            ],
            "tool_calls": [
                _tc("run_tests", "make test", "212 passed"),
                _tc("run_lint", "make lint", "0 issues"),
            ],
            "skills_used": ["requesting-code-review"],
            "errors": [],
            "workflow": ["run make test", "run make lint", "confirm clean"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Tests are failing for me but not in CI, can you look?"),
                _turn("assistant", "I ran bare pytest and hit a ModuleNotFoundError; that is the wrong interpreter, the venv is not active."),
                _turn("assistant", "Re-ran with the project's command (make test, which uses the project venv) and it passes. The issue was bare pytest, not the code."),
            ],
            "tool_calls": [
                _tc("run_tests", "pytest", "ModuleNotFoundError: No module named 'app' (wrong interpreter)", False),
                _tc("read_file", "Makefile", "test target uses the project venv"),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "ran bare `pytest`, hit ModuleNotFoundError because the wrong interpreter/environment was used",
                 "fix": "do not run bare pytest; use the project's test command (make test), which runs the project venv"},
            ],
            "workflow": [
                "run the tests", "read the import error", "notice the environment is wrong",
                "switch to the project test command", "confirm green",
            ],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Quick, are the tests passing?"),
                _turn("assistant", "First try was bare pytest again and it could not import the package; the environment was not set up."),
                _turn("assistant", "Used make test instead: 212 passed. Same lesson as before, bare pytest does not pick up the project environment."),
            ],
            "tool_calls": [
                _tc("run_tests", "pytest -q", "ModuleNotFoundError: No module named 'app'", False),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": [],
            "errors": [
                {"error": "bare `pytest` could not import the package: wrong environment",
                 "fix": "use make test (project venv), never bare pytest, for this repo"},
            ],
            "workflow": ["run bare pytest", "hit the import error", "use make test", "confirm green"],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Run just the auth tests."),
                _turn("assistant", "Passing the filter through the project test command."),
                _turn("assistant", "14 passed."),
            ],
            "tool_calls": [
                _tc("run_tests", "make test -- -k auth", "14 passed"),
            ],
            "skills_used": [],
            "errors": [],
            "workflow": ["run the filtered suite via make test", "report the result"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "The flaky test again, can you confirm it on a clean run?"),
                _turn("assistant", "Reflex was bare pytest and it failed to import again; I keep doing that on this repo."),
                _turn("assistant", "make test is clean, the test is not flaky, it was the environment. Noting it so I stop reaching for bare pytest."),
            ],
            "tool_calls": [
                _tc("run_tests", "pytest tests/test_eta.py", "ModuleNotFoundError: No module named 'app'", False),
                _tc("run_tests", "make test -- -k eta", "3 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "reached for bare `pytest` again; import failed because the environment was not active",
                 "fix": "this repo needs make test (project venv); stop defaulting to bare pytest"},
            ],
            "workflow": ["run bare pytest out of habit", "hit the import error", "switch to make test", "confirm not flaky"],
        },
    ],
    "add_api_endpoint": [
        {
            "day": 1,
            "conversation": [
                _turn("user", "Add a GET /health endpoint that returns service status."),
                _turn("assistant", "Following the existing router pattern and adding a test alongside."),
                _turn("assistant", "Endpoint added, test passes, returns 200 with a status body."),
            ],
            "tool_calls": [
                _tc("grep", "router.get(", "found the existing route definitions"),
                _tc("write_file", "app/routes/health.py", "created"),
                _tc("edit_file", "app/router.py: register health route", "1 hunk applied"),
                _tc("run_tests", "make test -- -k health", "1 passed"),
            ],
            "skills_used": ["test-driven-development"],
            "errors": [],
            "workflow": [
                "find the router pattern", "write a failing test for /health",
                "implement the route", "register it", "run the test",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Add POST /items to create an item."),
                _turn("assistant", "Reusing the request-validation helper the other write routes use."),
                _turn("assistant", "Route, validation and test in; suite green."),
            ],
            "tool_calls": [
                _tc("read_file", "app/routes/orders.py", "saw the validation helper usage"),
                _tc("write_file", "app/routes/items.py", "created"),
                _tc("run_tests", "make test -- -k items", "2 passed"),
            ],
            "skills_used": [],
            "errors": [],
            "workflow": ["read a similar route", "write the route", "reuse the validation helper", "test it"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Add DELETE /items/{id}."),
                _turn("assistant", "First version returned 200 on a missing id; the project convention is 404, fixing."),
                _turn("assistant", "Now returns 404 for unknown ids, matching the other routes. Tests cover both cases."),
            ],
            "tool_calls": [
                _tc("write_file", "app/routes/items.py: add delete", "created handler"),
                _tc("run_tests", "make test -- -k delete", "1 failed: expected 404 got 200", False),
                _tc("edit_file", "app/routes/items.py: raise 404 on missing id", "1 hunk applied"),
                _tc("run_tests", "make test -- -k delete", "2 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "delete route returned 200 for a missing id",
                 "fix": "follow the project convention: raise 404 for unknown ids, as the other routes do"},
            ],
            "workflow": [
                "add the delete handler", "run the test", "see the wrong status code",
                "apply the 404 convention", "re-test",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Add GET /items/{id} to fetch one item."),
                _turn("assistant", "Mirroring the list route and adding a not-found test up front."),
                _turn("assistant", "Route and tests in; both the found and missing cases pass."),
            ],
            "tool_calls": [
                _tc("read_file", "app/routes/items.py", "saw the list handler"),
                _tc("edit_file", "app/routes/items.py: add get-by-id", "1 hunk applied"),
                _tc("run_tests", "make test -- -k items", "3 passed"),
            ],
            "skills_used": ["test-driven-development"],
            "errors": [],
            "workflow": ["read the list route", "write the not-found test", "add the handler", "run the test"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Add PATCH /items/{id} for partial updates."),
                _turn("assistant", "First cut let unknown fields through; the convention is to reject them with 422."),
                _turn("assistant", "Now rejects unknown fields like the other write routes; tests cover it."),
            ],
            "tool_calls": [
                _tc("edit_file", "app/routes/items.py: add patch", "created handler"),
                _tc("run_tests", "make test -- -k patch", "1 failed: unknown field accepted", False),
                _tc("edit_file", "app/routes/items.py: reject unknown fields (422)", "1 hunk applied"),
                _tc("run_tests", "make test -- -k patch", "2 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "patch route accepted unknown fields silently",
                 "fix": "reject unknown fields with 422, matching the project's write-route convention"},
            ],
            "workflow": ["add the patch handler", "run the test", "spot the loose validation", "apply the 422 convention", "re-test"],
        },
    ],
    "fix_failing_build": [
        {
            "day": 1,
            "conversation": [
                _turn("user", "The build is red on main, can you fix it?"),
                _turn("assistant", "A type error slipped in; the function returns Optional but the caller assumes a value."),
                _turn("assistant", "Guarded the None case; typecheck and build are green."),
            ],
            "tool_calls": [
                _tc("run_build", "make build", "error: type check failed in service.py", False),
                _tc("run_typecheck", "make typecheck", "Optional[str] not handled at call site"),
                _tc("edit_file", "app/service.py: handle None", "1 hunk applied"),
                _tc("run_build", "make build", "ok"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "type check failed: Optional[str] used without a None guard",
                 "fix": "guard the None case at the call site before use"},
            ],
            "workflow": ["run the build", "read the type error", "add the None guard", "rebuild"],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Build broke after the dependency bump."),
                _turn("assistant", "The bumped library renamed an import; updating the import path."),
                _turn("assistant", "Build green again."),
            ],
            "tool_calls": [
                _tc("run_build", "make build", "ImportError: cannot import name 'Client'", False),
                _tc("grep", "from sdk import Client", "one usage"),
                _tc("edit_file", "app/client.py: update import path", "1 hunk applied"),
                _tc("run_build", "make build", "ok"),
            ],
            "skills_used": [],
            "errors": [
                {"error": "ImportError after a dependency bump: a symbol moved",
                 "fix": "update the import to the new path from the library changelog"},
            ],
            "workflow": ["run the build", "read the import error", "check the changelog", "update the import", "rebuild"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "CI build fails but it builds fine locally."),
                _turn("assistant", "Local picked up a cached artefact; a clean build reproduces the CI failure, a missing generated file."),
                _turn("assistant", "Added the codegen step to the build target; clean build passes."),
            ],
            "tool_calls": [
                _tc("run_build", "make build", "ok (used cache)"),
                _tc("run_build", "make clean build", "error: generated_schema.py not found", False),
                _tc("edit_file", "Makefile: add codegen before build", "1 hunk applied"),
                _tc("run_build", "make clean build", "ok"),
                _tc("git_commit", "build: run codegen before build", "committed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "build passed locally on a cache but failed clean: a generated file was missing",
                 "fix": "always reproduce with a clean build; wire codegen into the build target"},
            ],
            "workflow": [
                "build locally", "reproduce with a clean build", "find the missing generated file",
                "add codegen to the build", "commit",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Lint is failing the build on unused imports."),
                _turn("assistant", "Removing the dead imports the linter flags."),
                _turn("assistant", "Lint clean, build green."),
            ],
            "tool_calls": [
                _tc("run_build", "make build", "lint error: 4 unused imports", False),
                _tc("run_lint", "make lint --fix", "removed 4 unused imports"),
                _tc("run_build", "make build", "ok"),
            ],
            "skills_used": [],
            "errors": [
                {"error": "build failed on lint: unused imports",
                 "fix": "run make lint --fix before pushing; remove dead imports"},
            ],
            "workflow": ["run the build", "read the lint failure", "auto-fix the imports", "rebuild"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Build fails intermittently in CI only."),
                _turn("assistant", "Tests share state through a module global; order-dependence makes it flaky. Reproduced with a fixed random order."),
                _turn("assistant", "Isolated the fixture per test; clean across orderings now."),
            ],
            "tool_calls": [
                _tc("run_tests", "make test -- -p no:randomly", "passed"),
                _tc("run_tests", "make test -- -p randomly --seed 1", "1 failed: state leaked between tests", False),
                _tc("edit_file", "tests/conftest.py: isolate fixture per test", "1 hunk applied"),
                _tc("run_tests", "make test -- -p randomly --seed 1", "passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "intermittent CI failure: tests leaked state through a shared global (order-dependent)",
                 "fix": "reproduce with randomised order; give each test an isolated fixture"},
            ],
            "workflow": [
                "run tests in default order", "reproduce with randomised order",
                "find the shared state", "isolate the fixture", "confirm across orderings",
            ],
        },
    ],
    "refactor_module": [
        {
            "day": 1,
            "conversation": [
                _turn("user", "Split the giant utils.py into focused modules."),
                _turn("assistant", "Grouping by responsibility and keeping a thin re-export so imports do not break."),
                _turn("assistant", "Done; the suite passes unchanged, no call sites edited."),
            ],
            "tool_calls": [
                _tc("read_file", "app/utils.py", "600 lines, mixed concerns"),
                _tc("write_file", "app/utils/text.py", "created"),
                _tc("write_file", "app/utils/dates.py", "created"),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": ["writing-plan-spec"],
            "errors": [],
            "workflow": [
                "read the module", "group functions by responsibility",
                "split into submodules", "keep a re-export shim", "run the suite",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Extract the retry logic that is duplicated in three clients."),
                _turn("assistant", "Pulling it into one helper and pointing the three clients at it."),
                _turn("assistant", "Single helper now; tests green and three copies gone."),
            ],
            "tool_calls": [
                _tc("grep", "for attempt in range(", "three near-identical retry loops"),
                _tc("write_file", "app/retry.py", "created with_retry helper"),
                _tc("edit_file", "three clients use with_retry", "3 hunks applied"),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": ["test-driven-development"],
            "errors": [],
            "workflow": ["find the duplication", "write one helper", "repoint the call sites", "run the suite"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Rename the Account model field 'name' to 'display_name' everywhere."),
                _turn("assistant", "The first pass missed a string reference in a serializer; a grep for the old name caught it."),
                _turn("assistant", "All references updated, including the serializer; suite green."),
            ],
            "tool_calls": [
                _tc("edit_file", "rename field across models and callers", "8 hunks applied"),
                _tc("run_tests", "make test", "1 failed: KeyError 'name' in serializer", False),
                _tc("grep", "\"name\"", "missed serializer string key"),
                _tc("edit_file", "serializer: use display_name", "1 hunk applied"),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "rename missed a string reference in a serializer (KeyError 'name')",
                 "fix": "after a rename, grep for the old name as a string, not just as a symbol"},
            ],
            "workflow": [
                "rename the symbol across the code", "run the suite", "see the serializer KeyError",
                "grep for the old string", "fix the missed reference", "re-test",
            ],
        },
        {
            "day": 1,
            "conversation": [
                _turn("user", "Pull the validation rules out of the route handlers into one place."),
                _turn("assistant", "Moving them into a validators module and importing from the handlers."),
                _turn("assistant", "Centralised; handlers are thinner and the suite is green."),
            ],
            "tool_calls": [
                _tc("grep", "if not .* raise", "validation scattered across handlers"),
                _tc("write_file", "app/validators.py", "created"),
                _tc("edit_file", "handlers import from validators", "4 hunks applied"),
                _tc("run_tests", "make test", "212 passed"),
            ],
            "skills_used": ["writing-plan-spec"],
            "errors": [],
            "workflow": ["find the scattered validation", "create a validators module", "repoint the handlers", "run the suite"],
        },
        {
            "day": 2,
            "conversation": [
                _turn("user", "Convert the callback-style db layer to async."),
                _turn("assistant", "A first pass deadlocked because one sync call was left inside the event loop; found it by tracing the hang."),
                _turn("assistant", "Made that call async too; suite passes and the deadlock is gone."),
            ],
            "tool_calls": [
                _tc("edit_file", "app/db.py: convert to async", "12 hunks applied"),
                _tc("run_tests", "make test -- -k db", "hung, then timed out", False),
                _tc("grep", "time.sleep|requests.get", "one blocking call left in the loop"),
                _tc("edit_file", "app/db.py: await the last blocking call", "1 hunk applied"),
                _tc("run_tests", "make test -- -k db", "9 passed"),
            ],
            "skills_used": ["systematic-debugging"],
            "errors": [
                {"error": "async conversion deadlocked: a blocking sync call was left inside the event loop",
                 "fix": "after an async migration, grep for blocking calls (sleep, sync http) and await or offload them"},
            ],
            "workflow": [
                "convert the layer to async", "run the tests", "trace the hang",
                "find the blocking call", "make it async", "re-test",
            ],
        },
    ],
}


def _sample_preferences(rng: random.Random) -> list[str]:
    """Pick a stable small set of preferences, near the real median of 4."""
    n = rng.randint(3, REAL_TRACE_STATS["preferences_per_episode"]["max"])
    return sorted(rng.sample(CODING_PREFERENCES, n))


def synthesize_fixture(out_path: str | Path, *, seed: int = 42) -> int:
    """Write the synthetic coding-agent fixture to ``out_path``; return count.

    Deterministic: same seed gives byte-identical output. Episodes are sorted by
    topic then episode_id. Each episode is engine-shaped (see induction_engine).
    """
    rng = random.Random(seed)
    episodes: list[dict] = []
    for topic in TOPICS:
        for i, scene in enumerate(SCENES[topic], start=1):
            ep = {
                "episode_id": f"ep-{topic}-{i:03d}",
                "day": scene["day"],
                "topic": topic,
                "conversation": scene["conversation"],
                "tool_calls": scene["tool_calls"],
                "skills_used": list(scene["skills_used"]),
                "errors": scene["errors"],
                "workflow": scene["workflow"],
                "preferences": _sample_preferences(rng),
            }
            episodes.append(ep)
    episodes.sort(key=lambda e: (e["topic"], e["episode_id"]))
    out = Path(out_path)
    out.write_text(json.dumps(episodes, indent=2) + "\n", encoding="utf-8")
    return len(episodes)


_FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "coding_agent_traces.json"

# Curated standard skill box: generic procedural recipes the agent starts with.
# Deliberately generic: the run_test_suite recipe does NOT mention the bare-pytest
# gotcha, so the induced (enhanced) skill has something concrete to add in Step 5.
#
# Each entry carries the ``topic`` of the enhanced skill that supersedes it (the
# same topic the induction engine writes, see TOPICS). When an enhanced skill for
# that topic is approved, the matching standard skill is marked ``promoted`` and
# dropped from retrieval, so the agent never holds two recipes for one job.
BASE_SKILLS: list[dict] = [
    {"name": "run-the-tests",
     "topic": "run_test_suite",
     "description": "Run the project's test suite and report whether it passes.",
     "recipe_steps": ["find the test command", "run it", "report the result"]},
    {"name": "set-up-the-project",
     "topic": "setup_dev_env",
     "description": "Get the project running locally from a clean checkout.",
     "recipe_steps": ["read the README", "install dependencies", "start the app"]},
    {"name": "add-an-endpoint",
     "topic": "add_api_endpoint",
     "description": "Add a new API endpoint following the existing route pattern.",
     "recipe_steps": ["read a similar route", "write the handler", "add a test"]},
    {"name": "fix-the-build",
     "topic": "fix_failing_build",
     "description": "Diagnose and fix a failing build.",
     "recipe_steps": ["run the build", "read the first error", "fix and rebuild"]},
    {"name": "refactor-safely",
     "topic": "refactor_module",
     "description": "Refactor a module without changing behaviour.",
     "recipe_steps": ["read the module", "make the change", "run the suite"]},
    {"name": "systematic-debugging",
     "topic": "support_systematic_debugging",
     "description": "Diagnose a failure from evidence before changing code.",
     "recipe_steps": ["reproduce the failure", "find the cause", "verify the fix"]},
    {"name": "requesting-code-review",
     "topic": "support_requesting_code_review",
     "description": "Request a focused review after implementation and tests.",
     "recipe_steps": ["summarise the change", "provide the diff", "address findings"]},
    {"name": "test-driven-development",
     "topic": "support_test_driven_development",
     "description": "Drive a behaviour change with a failing test first.",
     "recipe_steps": ["write a failing test", "make it pass", "refactor safely"]},
    {"name": "writing-plan-spec",
     "topic": "support_writing_plan_spec",
     "description": "Turn requirements into a checked implementation plan.",
     "recipe_steps": ["clarify the goal", "plan small steps", "name verification"]},
    {"name": "executing-plans",
     "topic": "support_executing_plans",
     "description": "Execute an approved plan with verification checkpoints.",
     "recipe_steps": ["take one task", "run its checks", "record the result"]},
]


_BASE_BY_NAME = {skill["name"]: skill for skill in BASE_SKILLS}
LESSON_SKILLS: list[dict] = [
    {
        "name": "brainstorming",
        "topic": "support_brainstorming",
        "description": "Clarify intent and explore options before implementation.",
        "recipe_steps": ["clarify the goal", "compare approaches", "agree the design"],
    },
    {
        "name": "writing-plans",
        "topic": "support_writing_plans",
        "description": "Turn an approved design into small verified tasks.",
        "recipe_steps": ["list the changes", "name the checks", "plan small steps"],
    },
    _BASE_BY_NAME["run-the-tests"],
    _BASE_BY_NAME["systematic-debugging"],
    _BASE_BY_NAME["requesting-code-review"],
]
SKILL_TOPICS = {
    skill["name"]: skill["topic"]
    for skill in [*BASE_SKILLS, *LESSON_SKILLS]
}


def canonical_skill_name(topic: str, proposed_name: str) -> str:
    """Keep a refinement under the stable name of its standard skill."""
    return next(
        (name for name, mapped_topic in SKILL_TOPICS.items()
         if mapped_topic == topic),
        proposed_name,
    )


def load_fixture(path: str | Path | None = None) -> list[dict]:
    """Load the synthetic coding-agent episode fixture."""
    return json.loads(Path(path or _FIXTURE_PATH).read_text(encoding="utf-8"))


def trace_type_counts(episodes: list[dict]) -> dict[str, int]:
    """Count the six trace types across the fixture (for the Step 1 printout)."""
    return {
        "conversations": sum(len(e["conversation"]) for e in episodes),
        "tools_called": sum(len(e["tool_calls"]) for e in episodes),
        "skills_used": sum(len(e["skills_used"]) for e in episodes),
        "workflows": sum(len(e["workflow"]) for e in episodes),
        "preferences": sum(len(e["preferences"]) for e in episodes),
        "errors": sum(len(e["errors"]) for e in episodes),
    }


def seed_into_memory(
    mem, *, clean: bool = True, path: str | Path | None = None,
    base_skills: list[dict] | None = None,
) -> dict:
    """Seed AgentMemory from the synthetic fixture (Step 1 of the notebook).

    Clears both skill boxes (so a re-run is deterministic), writes the curated
    standard skill box, and lands one retrieval trace per episode so the traces
    are visibly in Oracle. Uses only no-LLM write paths (SKILLBOX and
    RETRIEVAL_TRACES), so seeding is fast and needs no model credentials; the
    induction engine reads the episode fixture directly in Step 4. Returns the
    six trace-type counts plus the base-skill and episode counts. Requires a
    live Oracle backend.
    """
    episodes = load_fixture(path)
    selected_skills = BASE_SKILLS if base_skills is None else base_skills
    if clean:
        mem.truncate_skills()
        mem.truncate_enhanced_skills()

    for s in selected_skills:
        mem.write_skill(
            name=s["name"], description=s["description"],
            recipe_steps=s["recipe_steps"], provenance=["curated"],
            status="active",
        )

    # Land one retrieval trace per episode: the task as the query, the tools it
    # reached for as the candidates, and the outcome (failure when the episode
    # recorded an error). No LLM extraction, so this is fast and key-free.
    for ep in episodes:
        query = ep["conversation"][0]["content"] if ep["conversation"] else ep["topic"]
        tools = [tc["tool"] for tc in ep["tool_calls"]]
        candidates = [{"id": t, "text": t} for t in tools]
        outcome = "failure" if ep["errors"] else "success"
        mem.record_retrieval_trace(
            query=query, candidates=candidates, chosen=tools, outcome=outcome,
        )

    counts = trace_type_counts(episodes)
    counts["base_skills"] = len(selected_skills)
    counts["episodes"] = len(episodes)
    return counts


def main(argv: list[str] | None = None) -> int:
    out = Path(__file__).resolve().parent / "data" / "coding_agent_traces.json"
    n = synthesize_fixture(out)
    print(f"Wrote {n} synthetic coding-agent episodes to {out}")
    return 0


__all__ = [
    "REAL_TRACE_STATS", "CODING_TOOLS", "CODING_SKILLS", "CODING_PREFERENCES",
    "TOPICS", "SCENES", "BASE_SKILLS", "LESSON_SKILLS", "SKILL_TOPICS",
    "canonical_skill_name",
    "synthesize_fixture",
    "load_fixture", "trace_type_counts", "seed_into_memory", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
