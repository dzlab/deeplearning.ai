#!/usr/bin/env bash
# teardown_oracle_free.sh — Stop and remove the Oracle Free container.
# Idempotent: safe to run even when the container does not exist.
#
# Usage:
#   bash scripts/teardown_oracle_free.sh

set -euo pipefail

CONTAINER_NAME="dlai-oracle-free"

# Container runtime: prefer Podman, fall back to Docker. Override with CR=docker.
CR="${CR:-}"
if [ -z "${CR}" ]; then
  if command -v podman >/dev/null 2>&1; then
    CR="podman"
  elif command -v docker >/dev/null 2>&1; then
    CR="docker"
  else
    echo "[teardown] ERROR: neither podman nor docker found on PATH." >&2
    exit 1
  fi
fi

echo "[teardown] Stopping and removing container '${CONTAINER_NAME}' (${CR}) ..."
"${CR}" stop "${CONTAINER_NAME}" 2>/dev/null && "${CR}" rm "${CONTAINER_NAME}" 2>/dev/null || \
  "${CR}" rm -f "${CONTAINER_NAME}" 2>/dev/null || \
  echo "[teardown] Container '${CONTAINER_NAME}' was not running — nothing to do."

echo "[teardown] Done."
