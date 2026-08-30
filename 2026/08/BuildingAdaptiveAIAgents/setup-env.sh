#!/usr/bin/env bash
# setup-env.sh — make the lesson environment variables durable in the sandbox.
#
#   bash setup-env.sh
#
# What it does:
#   - creates ./.env from ./.env.sandbox if ./.env does not exist
#   - exports ORACLE_MEMORY_DB_* (+ DLAICL_*) for the CURRENT shell when sourced
#   - drops the same vars into each lesson dir's .env so a lesson started from
#     its own folder (DLAI runs each lesson self-contained) picks them up
#
# Source it to also affect your current shell:
#   source setup-env.sh
#
# Each lesson's helper.load_env() calls find_dotenv(), which walks UP from the
# lesson dir — so a .env at the repo root is found by L1/L2/.../L5 too. We also
# copy one INTO each lesson dir for the case where a lesson is run in isolation
# (e.g. shipped to the platform as its own container).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 1) root .env from the committed defaults
if [ ! -f "$ROOT/.env" ]; then
  if [ -f "$ROOT/.env.sandbox" ]; then
    cp "$ROOT/.env.sandbox" "$ROOT/.env"
    echo "[setup-env] created .env from .env.sandbox"
  else
    echo "[setup-env] ERROR: no .env.sandbox to copy from" >&2
    exit 1
  fi
else
  echo "[setup-env] .env already exists — leaving it as is"
fi

# 2) per-lesson .env (so a lesson run in its own container is self-contained)
for L in L1 L2 L3 L4 L5; do
  if [ -d "$ROOT/$L" ]; then
    cp "$ROOT/.env" "$ROOT/$L/.env"
  fi
done
echo "[setup-env] copied .env into L1..L5"

# 3) export into the current shell (effective only when this script is SOURCED)
set -a
# shellcheck disable=SC1091
. "$ROOT/.env"
set +a
echo "[setup-env] exported ORACLE_MEMORY_DB_* + DLAICL_* into the current shell"

# 4) fetch/build the L4 live-CPU GGUF assets (~6 GB) BY DEFAULT so the sandbox
#    is fully ready after setup — Lesson4_minimal's "try it live" cell then runs the
#    REAL model instead of replaying the cache. Idempotent (kept if on disk).
#    Opt out with `bash setup-env.sh --no-gguf`. (`--with-gguf` still accepted.)
if [ "${1:-}" = "--no-gguf" ]; then
  echo "[setup-env] --no-gguf: skipping L4 GGUF assets (live cell replays cache)"
else
  bash "$ROOT/setup-gguf-assets.sh"
fi

cat <<'NOTE'

Done. Lessons can now reach Oracle.

  L1, L2, L3 -> need the live Oracle container (these vars point at it)
  L4, L5     -> run CPU-only from committed caches (Oracle vars harmless/unused)

  L4 live path: this script now provisions the 4-bit GGUF base +
  superpolitegemma v2 adapter and builds the llama.cpp engine (~6 GB) BY
  DEFAULT, so Lesson4_minimal's "try it live" cell runs the REAL model on CPU.
  Skip with `bash setup-env.sh --no-gguf` (the cell then replays the cache).

If your Oracle container uses a different password/port, edit .env (or
.env.sandbox) and re-run this script.
NOTE
