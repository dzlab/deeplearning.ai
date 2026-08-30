#!/usr/bin/env bash
# bootstrap_oracle_free.sh — Idempotent Oracle Free container setup for the
# DLAI Continual Learning course.
#
# Usage:
#   bash scripts/bootstrap_oracle_free.sh
#
# Environment (all optional — defaults shown):
#   ORACLE_MEMORY_DB_PASSWORD              continual_learning
#   ORACLE_FREE_IMAGE                      container-registry.oracle.com/database/free:23.26.1.0-lite
#   ORACLE_FREE_FALLBACK_IMAGE             container-registry.oracle.com/database/free:latest-lite
#   DLAI_REQUIRE_ORACLE_26AI               1 (fail if the DB is not Oracle AI Database 26ai)
#   CR                                     auto-detected (podman preferred, then docker)
#   DLAI_AUTO_INSTALL_CONTAINER_RUNTIME    1 (try to install Docker/Podman if missing)
#
# What it does:
#   1. Pulls and starts the Oracle Free container (idempotent).
#   2. Waits up to 5 minutes for the database to become ready.
#   3. Creates the dmuser account (scripts/sql/000_dmuser.sql, run as SYSTEM).
#   4. Applies course-extra DDL files 001–004 as dmuser.
#   5. Prints the three ORACLE_MEMORY_DB_* env vars for copy-paste.
#
# A log of applied statements is teed to scripts/sql/.applied.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${SCRIPT_DIR}/sql/.applied"
CONTAINER_NAME="dlai-oracle-free"
# Oracle AI Database 26ai Free uses 23.26.x image tags. Pin the primary image
# to the current 26ai Free Lite tag instead of a moving `latest`, and keep the
# fallback on Oracle's 26ai Free Lite track. Do not fall back to community images
# when the course requires 26ai features.
PRIMARY_IMAGE="${ORACLE_FREE_IMAGE:-container-registry.oracle.com/database/free:23.26.1.0-lite}"
FALLBACK_IMAGE="${ORACLE_FREE_FALLBACK_IMAGE:-container-registry.oracle.com/database/free:latest-lite}"
REQUIRE_ORACLE_26AI="${DLAI_REQUIRE_ORACLE_26AI:-1}"
ORACLE_PWD="${ORACLE_MEMORY_DB_PASSWORD:-continual_learning}"
ORACLE_PORT="${ORACLE_PORT:-1521}"
WAIT_SECONDS=300   # 5 minutes

run_privileged() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo -n "$@"
  else
    return 1
  fi
}

PYBIN_FOR_PROBE() {
  # The project venv has oracledb + course_lab; fall back to PATH python.
  local p="${REPO_ROOT}/.venv/bin/python"
  [ -x "${p}" ] || p="python"
  printf '%s' "${p}"
}

# True (rc 0) when a live Oracle connection can be opened with the given
# user/password/DSN. Uses course_lab.oracle_db so the probe matches exactly how
# the rest of the course connects.
oracle_can_connect() {
  local user="$1" password="$2" dsn="$3"
  (
    cd "${REPO_ROOT}" || exit 1
    export ORACLE_MEMORY_DB_USER="${user}"
    export ORACLE_MEMORY_DB_PASSWORD="${password}"
    export ORACLE_MEMORY_DB_CONNECT_STRING="${dsn}"
    "$(PYBIN_FOR_PROBE)" - <<'PY' 2>/dev/null
# Exit codes: 0=ok, 2=auth failure (wrong pwd / locked — caller must NOT retry
# this user, to avoid tripping FAILED_LOGIN_ATTEMPTS), 1=absent (host/service
# unreachable; safe to keep probing other DSNs/creds).
import os, sys
from course_lab import oracle_db
p = oracle_db.connection_params_from_env()
try:
    od = oracle_db._oracledb()
    od.connect(user=p["user"], password=p["password"], dsn=p["dsn"]).close()
    sys.exit(0)
except Exception as exc:
    msg = str(exc)
    auth = ("ORA-01017", "ORA-28000", "ORA-28001", "ORA-01005", "ORA-28011")
    sys.exit(2 if any(c in msg for c in auth) else 1)
PY
  )
}

# Apply a .sql file over a *network* connection as the given user (no container
# exec). Returns the apply step's rc.
oracle_apply_network() {
  local user="$1" password="$2" dsn="$3" sql_file="$4"
  [ -f "${sql_file}" ] || { echo "[bootstrap] WARNING: ${sql_file} not found — skipping." >&2; return 0; }
  echo "[bootstrap] Applying ${sql_file} as ${user} (network) ..."
  (
    cd "${REPO_ROOT}" || exit 1
    export ORACLE_MEMORY_DB_USER="${user}"
    export ORACLE_MEMORY_DB_PASSWORD="${password}"
    export ORACLE_MEMORY_DB_CONNECT_STRING="${dsn}"
    "$(PYBIN_FOR_PROBE)" -m course_lab.oracle_db apply "${sql_file}"
  )
}

# Common-user fallback identity for a bare CDB root with no usable PDB. Name
# MUST start with C## in the root.
C_COMMON_USER="${C_COMMON_USER:-C##DLAI}"
C_COMMON_PWD="${C_COMMON_PWD:-continual_learning}"

# Create the C## common user in the CDB root by substituting the &&-vars in
# 000_c_common_user.sql and applying it as the admin over the network. Returns 0
# only when the new user is then reachable.
provision_common_user_via_admin() {
  local admin_user="$1" admin_pwd="$2" dsn="$3"
  local tmpl="${SCRIPT_DIR}/sql/000_c_common_user.sql"
  [ -f "${tmpl}" ] || { echo "[bootstrap] WARNING: ${tmpl} not found." >&2; return 1; }
  local sub="${TMPDIR:-/tmp}/dlai-c-common-$$.sql"
  sed -e "s/&&CUSER/${C_COMMON_USER}/g" -e "s/&&CPWD/${C_COMMON_PWD}/g" "${tmpl}" > "${sub}"
  oracle_apply_network "${admin_user}" "${admin_pwd}" "${dsn}" "${sub}"
  rm -f "${sub}"
  oracle_can_connect "${C_COMMON_USER}" "${C_COMMON_PWD}" "${dsn}"
  [ $? -eq 0 ]
}

# When Oracle is ALREADY reachable (a sandbox-provided or externally-managed DB),
# we must NOT try to pull/start a container — hosted notebook sandboxes routinely
# forbid that (unshare: operation not permitted). Instead: verify the connection,
# ensure the course's dmuser + course-extra DDL exist over the network, and exit
# success. Container provisioning is reserved for the genuinely-no-DB case.
#
# Honors the same env the course uses: ORACLE_MEMORY_DB_* (course/dmuser creds)
# and, for the admin step, ORACLE_MEMORY_DB_* if they point at an admin account
# or ORACLE_PWD/SYSTEM. Nothing here assumes localhost.
try_use_existing_oracle() {
  local course_user="${ORACLE_MEMORY_DB_USER:-dmuser}"
  local course_pwd="${ORACLE_MEMORY_DB_PASSWORD:-${ORACLE_PWD}}"
  local course_dsn="${ORACLE_MEMORY_DB_CONNECT_STRING:-localhost:${ORACLE_PORT}/FREEPDB1}"

  # 1. Already fully usable as the course user? Then there is nothing to do but
  #    confirm the course-extra DDL is present (idempotent) and finish.
  if oracle_can_connect "${course_user}" "${course_pwd}" "${course_dsn}"; then
    echo "[bootstrap] Oracle already reachable as ${course_user} at ${course_dsn} — skipping container bootstrap."
    apply_course_extra_ddl_network "${course_user}" "${course_pwd}" "${course_dsn}"
    print_ready_env "${course_user}" "${course_pwd}" "${course_dsn}"
    return 0
  fi

  # 2. Not reachable as the course user, but maybe an admin account is reachable
  #    on the same DSN (common when the sandbox hands out SYSTEM, not dmuser).
  #    Create dmuser + course DDL over the network, then re-verify as dmuser.
  local admin_dsn="${course_dsn}"
  local admin_user admin_pwd rc
  for admin_user in "${ORACLE_MEMORY_DB_USER:-}" system sys; do
    [ -n "${admin_user}" ] || continue
    # ORACLE_SYSTEM_PASSWORD is tried FIRST: the sandbox's SYSTEM password often
    # differs from the course user's, and the no-lock guard below only allows one
    # attempt per admin user — so the correct admin password must lead.
    for admin_pwd in "${ORACLE_SYSTEM_PASSWORD:-}" "${ORACLE_MEMORY_DB_PASSWORD:-}" "${ORACLE_PWD:-}" "YourPassword123" "continual_learning"; do
      [ -n "${admin_pwd}" ] || continue
      oracle_can_connect "${admin_user}" "${admin_pwd}" "${admin_dsn}"
      rc=$?
      if [ "${rc}" -eq 2 ]; then
        # Wrong password or locked account. Do NOT try more passwords for this
        # user — another wrong attempt could trip FAILED_LOGIN_ATTEMPTS and lock
        # it. Skip to the next admin user.
        echo "[bootstrap] '${admin_user}' auth failed at ${admin_dsn}; not retrying this user (avoids account lock)." >&2
        break
      fi
      if [ "${rc}" -eq 0 ]; then
        echo "[bootstrap] Oracle reachable as admin '${admin_user}' at ${admin_dsn} — provisioning the course user over the network (no container)."
        # Try dmuser (the PDB case) first.
        if oracle_apply_network "${admin_user}" "${admin_pwd}" "${admin_dsn}" "${SCRIPT_DIR}/sql/000_dmuser.sql" \
           && oracle_can_connect "dmuser" "continual_learning" "${admin_dsn}"; then
          apply_course_extra_ddl_network "dmuser" "continual_learning" "${admin_dsn}"
          print_ready_env "dmuser" "continual_learning" "${admin_dsn}"
          return 0
        fi
        # dmuser failed — likely a bare CDB root with no usable PDB (system can
        # CREATE but not OPEN a PDB without SYSDBA). Fall back to a C## common
        # user in the root, which works end-to-end (incl. the ONNX model load).
        echo "[bootstrap] dmuser not usable at ${admin_dsn}; trying common user ${C_COMMON_USER} in the CDB root..."
        if provision_common_user_via_admin "${admin_user}" "${admin_pwd}" "${admin_dsn}"; then
          apply_course_extra_ddl_network "${C_COMMON_USER}" "${C_COMMON_PWD}" "${admin_dsn}"
          print_ready_env "${C_COMMON_USER}" "${C_COMMON_PWD}" "${admin_dsn}"
          return 0
        fi
        echo "[bootstrap] WARNING: neither dmuser nor ${C_COMMON_USER} could be provisioned via ${admin_user}." >&2
      fi
      # rc==1 (absent): the DSN/service isn't there; trying other passwords is
      # pointless and harmless, but break to avoid noise.
      [ "${rc}" -eq 1 ] && break
    done
  done

  return 1  # No reachable Oracle — caller falls through to container bootstrap.
}

# Apply course-extra DDL (001..007) over the network as the given user. The
# dmuser-creating 000 file is intentionally NOT included here; it needs admin.
apply_course_extra_ddl_network() {
  local user="$1" password="$2" dsn="$3"
  for sql_file in "${SCRIPT_DIR}/sql/001_skillbox.sql" \
                  "${SCRIPT_DIR}/sql/002_query_traces.sql" \
                  "${SCRIPT_DIR}/sql/002_query_trace_intent.sql" \
                  "${SCRIPT_DIR}/sql/003_retrieval_traces.sql" \
                  "${SCRIPT_DIR}/sql/004_voyage_manifest.sql" \
                  "${SCRIPT_DIR}/sql/005_memory_graph.sql" \
                  "${SCRIPT_DIR}/sql/006_skillbox_governance.sql" \
                  "${SCRIPT_DIR}/sql/007_enhanced_skillbox.sql"; do
    oracle_apply_network "${user}" "${password}" "${dsn}" "${sql_file}" || {
      echo "[bootstrap] WARNING: applying ${sql_file} failed (continuing; may already exist)." >&2
    }
  done
}

print_ready_env() {
  local user="$1" password="$2" dsn="$3"
  echo ""
  echo "================================================================"
  echo "Oracle is ready (existing database — no container started).  Exports:"
  echo "================================================================"
  echo "export ORACLE_MEMORY_DB_USER=${user}"
  echo "export ORACLE_MEMORY_DB_PASSWORD=${password}"
  echo "export ORACLE_MEMORY_DB_CONNECT_STRING=${dsn}"
  echo "================================================================"
}

make_sudo_runtime_wrapper() {
  local runtime="$1"
  local docker_host="${2:-}"
  local wrapper="${TMPDIR:-/tmp}/dlai-${runtime}-wrapper-$$"
  {
    echo '#!/usr/bin/env bash'
    if [ -n "${docker_host}" ]; then
      printf 'exec sudo -n env DOCKER_HOST=%q %s "$@"\n' "${docker_host}" "${runtime}"
    else
      printf 'exec sudo -n %s "$@"\n' "${runtime}"
    fi
  } > "${wrapper}"
  chmod +x "${wrapper}"
  CR="${wrapper}"
}

start_relaxed_dockerd() {
  # Last-resort Docker-in-Docker mode for notebook sandboxes with sudo but no
  # init system. Disables Docker's bridge/iptables setup (often forbidden in
  # hosted containers), uses vfs storage, and runs Oracle with --network=host.
  # This still requires the host sandbox to permit nested containers.
  if ! command -v dockerd >/dev/null 2>&1 || ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if ! run_privileged true; then
    return 1
  fi

  local sock="${DLAI_DOCKER_HOST:-unix:///tmp/dlai-docker.sock}"
  local data_root="${DLAI_DOCKER_DATA_ROOT:-/tmp/dlai-docker-data}"
  local exec_root="${DLAI_DOCKER_EXEC_ROOT:-/tmp/dlai-docker-exec}"
  local pid_file="${DLAI_DOCKER_PID_FILE:-/tmp/dlai-dockerd.pid}"
  local log_file="${DLAI_DOCKERD_LOG:-/tmp/dlai-dockerd.log}"

  echo "[bootstrap] Trying fallback dockerd on ${sock} (log: ${log_file})..."

  # Previous failed attempts can leave a managed containerd running under the
  # fallback exec-root. Docker then logs "containerd is still running" and times
  # out forever. Clean only the DLAI-owned daemon/socket/exec-root; do not touch
  # a host/system Docker service.
  run_privileged sh -c "
    if [ -f '${pid_file}' ]; then kill \$(cat '${pid_file}') 2>/dev/null || true; fi
    pkill -f 'dockerd .*${sock}' 2>/dev/null || true
    pkill -f 'containerd .*${exec_root}' 2>/dev/null || true
    rm -f '${sock#unix://}' '${pid_file}'
    rm -rf '${exec_root}'
    mkdir -p '${data_root}' '${exec_root}'
  " >/dev/null 2>&1 || true

  run_privileged sh -c "nohup dockerd \
    --host='${sock}' \
    --data-root='${data_root}' \
    --exec-root='${exec_root}' \
    --pidfile='${pid_file}' \
    --iptables=false \
    --bridge=none \
    --ip-forward=false \
    --ip-masq=false \
    --storage-driver=vfs \
    >'${log_file}' 2>&1 &" >/dev/null 2>&1 || true

  for _ in $(seq 1 30); do
    if command -v sudo >/dev/null 2>&1 && sudo -n env DOCKER_HOST="${sock}" docker info >/dev/null 2>&1; then
      make_sudo_runtime_wrapper docker "${sock}"
      ORACLE_CONTAINER_NETWORK="${ORACLE_CONTAINER_NETWORK:-host}"
      echo "[bootstrap] Fallback dockerd is usable; Oracle container will run with --network=${ORACLE_CONTAINER_NETWORK}."
      return 0
    fi
    if DOCKER_HOST="${sock}" docker info >/dev/null 2>&1; then
      CR="docker"
      export DOCKER_HOST="${sock}"
      ORACLE_CONTAINER_NETWORK="${ORACLE_CONTAINER_NETWORK:-host}"
      echo "[bootstrap] Fallback dockerd is usable; Oracle container will run with --network=${ORACLE_CONTAINER_NETWORK}."
      return 0
    fi
    sleep 1
  done

  echo "[bootstrap] Fallback dockerd did not become usable. Last log lines:" >&2
  run_privileged sh -c "tail -80 '${log_file}' 2>/dev/null" >&2 || true
  run_privileged sh -c "
    if [ -f '${pid_file}' ]; then kill \$(cat '${pid_file}') 2>/dev/null || true; fi
    pkill -f 'dockerd .*${sock}' 2>/dev/null || true
    pkill -f 'containerd .*${exec_root}' 2>/dev/null || true
  " >/dev/null 2>&1 || true
  return 1
}

start_docker_daemon() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if command -v systemctl >/dev/null 2>&1; then
    run_privileged systemctl start docker >/dev/null 2>&1 || true
  fi
  if command -v service >/dev/null 2>&1; then
    run_privileged service docker start >/dev/null 2>&1 || true
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
    make_sudo_runtime_wrapper docker
    return 0
  fi
  if command -v dockerd >/dev/null 2>&1; then
    # Some learner sandboxes have no init system. Try a standard local daemon
    # first; then fall back to a no-bridge/no-iptables daemon if needed.
    run_privileged sh -c 'nohup dockerd >/tmp/dlai-dockerd.log 2>&1 &' >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if docker info >/dev/null 2>&1; then
        return 0
      fi
      if command -v sudo >/dev/null 2>&1 && sudo -n docker info >/dev/null 2>&1; then
        make_sudo_runtime_wrapper docker
        return 0
      fi
      sleep 1
    done
    start_relaxed_dockerd && return 0
  fi
  return 1
}

try_switch_to_docker_runtime() {
  # Used after a Podman/rootless pull failure such as
  # "failed to register layer: unshare: operation not permitted". Force Docker
  # selection even if CR currently points at Podman.
  local previous_cr="${CR:-}"
  CR=""
  if command -v docker >/dev/null 2>&1 && start_docker_daemon; then
    : "${CR:=docker}"
    echo "[bootstrap] Switched container runtime to Docker: ${CR}"
    return 0
  fi
  CR="${previous_cr}"
  return 1
}

select_existing_runtime() {
  # Prefer Docker/sudo-Docker for Oracle's large images. Rootless Podman can
  # report healthy yet fail layer registration in hosted notebook sandboxes with
  # `unshare: operation not permitted`.
  if command -v docker >/dev/null 2>&1 && start_docker_daemon; then
    : "${CR:=docker}"
    return 0
  fi

  # Sudo Podman is usually rootful and avoids the rootless unshare failure.
  if command -v podman >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && sudo -n podman info >/dev/null 2>&1; then
    make_sudo_runtime_wrapper podman
    return 0
  fi

  # Rootful Podman is OK. Rootless Podman is only OK if `podman unshare` works;
  # otherwise image pulls can fail after downloading every layer.
  if command -v podman >/dev/null 2>&1; then
    if [ "$(id -u)" -eq 0 ] && podman info >/dev/null 2>&1; then
      CR="podman"
      return 0
    fi
    if podman info >/dev/null 2>&1 && podman unshare true >/dev/null 2>&1; then
      CR="podman"
      return 0
    fi
  fi
  return 1
}

install_container_runtime() {
  if [ "${DLAI_AUTO_INSTALL_CONTAINER_RUNTIME:-1}" = "0" ]; then
    return 1
  fi

  echo "[bootstrap] No usable container runtime found; attempting to install Docker/Podman..."
  if ! run_privileged true; then
    echo "[bootstrap] ERROR: cannot install a container runtime: this user is not root and has no passwordless sudo." >&2
    return 1
  fi

  if command -v apt-get >/dev/null 2>&1; then
    run_privileged apt-get update
    run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates curl gnupg lsb-release uidmap iptables fuse-overlayfs slirp4netns dbus-user-session
    if run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io; then
      if start_docker_daemon; then
        : "${CR:=docker}"
        return 0
      fi
      echo "[bootstrap] Docker installed but the daemon is not usable; trying Podman fallback..." >&2
    fi
    if run_privileged env DEBIAN_FRONTEND=noninteractive apt-get install -y podman; then
      if select_existing_runtime; then
        return 0
      fi
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if run_privileged dnf install -y docker podman || run_privileged dnf install -y moby-engine podman; then
      start_docker_daemon || true
      select_existing_runtime && return 0
    fi
  elif command -v yum >/dev/null 2>&1; then
    if run_privileged yum install -y docker podman || run_privileged yum install -y moby-engine podman; then
      start_docker_daemon || true
      select_existing_runtime && return 0
    fi
  elif command -v apk >/dev/null 2>&1; then
    if run_privileged apk add --no-cache docker docker-cli podman; then
      start_docker_daemon || true
      select_existing_runtime && return 0
    fi
  else
    echo "[bootstrap] ERROR: unsupported OS/package manager; install Docker or Podman manually." >&2
    return 1
  fi

  echo "[bootstrap] ERROR: Docker/Podman installation did not yield a usable runtime." >&2
  echo "[bootstrap] If this is an unprivileged hosted notebook, it may not permit containers." >&2
  return 1
}

# ---------------------------------------------------------------------------
# 0. Fast path: if Oracle is ALREADY reachable (sandbox-provided / external DB),
#    verify it and provision the course schema over the network — never touch a
#    container runtime. This is what makes the script work in hosted notebook
#    sandboxes that forbid containers but already serve Oracle.
#    Set DLAI_SKIP_EXISTING_ORACLE=1 to force the container path regardless.
# ---------------------------------------------------------------------------
if [ "${DLAI_SKIP_EXISTING_ORACLE:-0}" != "1" ]; then
  if try_use_existing_oracle; then
    echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) — done (used existing Oracle, no container)"
    exit 0
  fi
  echo "[bootstrap] No reachable Oracle found — proceeding to container bootstrap."
fi

# Container runtime: prefer Podman, fall back to Docker. Override with CR=docker.
CR="${CR:-}"
if [ -z "${CR}" ]; then
  if ! select_existing_runtime; then
    install_container_runtime || {
      echo "[bootstrap] ERROR: neither podman nor docker is usable." >&2
      exit 1
    }
  fi
else
  case "${CR}" in
    docker)
      if ! command -v docker >/dev/null 2>&1 || ! start_docker_daemon; then
        echo "[bootstrap] Requested CR=docker, but Docker is not usable; attempting install/repair..."
        install_container_runtime || exit 1
      fi
      ;;
    podman)
      if ! command -v podman >/dev/null 2>&1; then
        echo "[bootstrap] Requested CR=podman, but Podman is not installed; attempting install..."
        install_container_runtime || exit 1
      fi
      ;;
  esac
fi

# Prefer the project's venv interpreter for the DDL step (it has oracledb +
# course_lab); fall back to whatever python is on PATH.
PYBIN="${REPO_ROOT}/.venv/bin/python"
[ -x "${PYBIN}" ] || PYBIN="python"

mkdir -p "${SCRIPT_DIR}/sql"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) — starting"
echo "[bootstrap] Using container runtime: ${CR}"
echo "[bootstrap] Oracle Free image: ${PRIMARY_IMAGE}"
echo "[bootstrap] Oracle Free fallback image: ${FALLBACK_IMAGE}"
echo "[bootstrap] Require Oracle AI Database 26ai: ${REQUIRE_ORACLE_26AI}"

# ---------------------------------------------------------------------------
# 1. Start container if not already running
# ---------------------------------------------------------------------------
if "${CR}" inspect --format='{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | grep -q true; then
  echo "[bootstrap] Container '${CONTAINER_NAME}' is already running — skipping ${CR} run."
else
  # Remove stopped container of the same name so ${CR} run does not fail.
  "${CR}" rm -f "${CONTAINER_NAME}" 2>/dev/null || true

  USING_IMAGE="${PRIMARY_IMAGE}"
  echo "[bootstrap] Attempting to pull Oracle AI Database 26ai Free image: ${PRIMARY_IMAGE}"
  if ! "${CR}" pull "${PRIMARY_IMAGE}"; then
    echo "[bootstrap] WARNING: Pull failed with runtime '${CR}'."
    if try_switch_to_docker_runtime; then
      echo "[bootstrap] Retrying primary image with Docker: ${PRIMARY_IMAGE}"
      if "${CR}" pull "${PRIMARY_IMAGE}"; then
        USING_IMAGE="${PRIMARY_IMAGE}"
      else
        echo "[bootstrap] WARNING: Primary image unavailable under Docker. Falling back to official Oracle image ${FALLBACK_IMAGE}."
        USING_IMAGE="${FALLBACK_IMAGE}"
        "${CR}" pull "${FALLBACK_IMAGE}"
      fi
    else
      echo "[bootstrap] WARNING: Could not switch to Docker. Falling back to official Oracle image ${FALLBACK_IMAGE}."
      USING_IMAGE="${FALLBACK_IMAGE}"
      "${CR}" pull "${FALLBACK_IMAGE}"
    fi
  fi

  RUN_NETWORK_ARGS=()
  if [ "${ORACLE_CONTAINER_NETWORK:-bridge}" = "host" ]; then
    RUN_NETWORK_ARGS+=(--network host)
    echo "[bootstrap] Starting container '${CONTAINER_NAME}' from ${USING_IMAGE} with host networking ..."
  else
    RUN_NETWORK_ARGS+=(-p "${ORACLE_PORT}:1521")
    echo "[bootstrap] Starting container '${CONTAINER_NAME}' from ${USING_IMAGE} with port ${ORACLE_PORT}:1521 ..."
  fi
  "${CR}" run -d \
    --name "${CONTAINER_NAME}" \
    "${RUN_NETWORK_ARGS[@]}" \
    -e ORACLE_PWD="${ORACLE_PWD}" \
    "${USING_IMAGE}"
fi

# ---------------------------------------------------------------------------
# 2. Wait for the database to become ready (poll up to WAIT_SECONDS)
#    Probe with a real SQL connection rather than checkDBStatus.sh: the latter's
#    output changed on recent images ("Oracle base remains unchanged ...") and no
#    longer prints "DATABASE IS READY". A successful SELECT against FREEPDB1 is
#    the ground truth and is robust across image versions.
# ---------------------------------------------------------------------------
verify_oracle_26ai() {
  VERSION_INFO=$("${CR}" exec -i "${CONTAINER_NAME}" \
    sqlplus -s -L "system/${ORACLE_PWD}@FREEPDB1" 2>/dev/null <<'SQL' || true
SET HEADING OFF FEEDBACK OFF PAGESIZE 0 LINESIZE 32767
SELECT banner_full FROM v$version WHERE banner_full LIKE 'Oracle Database%';
SELECT product || ' ' || version_full FROM product_component_version WHERE product LIKE 'Oracle Database%';
EXIT
SQL
)
  echo "[bootstrap] Oracle version info:"
  echo "${VERSION_INFO}" | sed 's/^/[bootstrap]   /'

  if [ "${REQUIRE_ORACLE_26AI}" = "0" ]; then
    return 0
  fi

  # 26ai Free images can expose both the marketing name (26ai) and an internal
  # 23.26.x version string. Accept either; reject older 23ai/21c/19c images.
  if echo "${VERSION_INFO}" | grep -Eiq '(26ai|23[.]26[.]|(^|[^0-9])26[.])'; then
    echo "[bootstrap] Oracle AI Database 26ai verification passed."
    return 0
  fi

  echo "[bootstrap] ERROR: Oracle container is not reporting Oracle AI Database 26ai." >&2
  echo "[bootstrap]        Refusing to continue because DLAI_REQUIRE_ORACLE_26AI=${REQUIRE_ORACLE_26AI}." >&2
  echo "[bootstrap]        Remove/recreate '${CONTAINER_NAME}' or set ORACLE_FREE_IMAGE to a 26ai image." >&2
  return 1
}

echo "[bootstrap] Waiting for database to be ready (up to ${WAIT_SECONDS}s) ..."
ELAPSED=0
POLL=10
while true; do
  STATUS=$("${CR}" exec -i "${CONTAINER_NAME}" \
    sqlplus -s -L "system/${ORACLE_PWD}@FREEPDB1" 2>/dev/null <<'SQL' || true
SET HEADING OFF FEEDBACK OFF PAGESIZE 0
SELECT 'DATABASE IS READY' FROM dual;
EXIT
SQL
)
  if echo "${STATUS}" | grep -qi "DATABASE IS READY"; then
    echo "[bootstrap] Database is ready."
    verify_oracle_26ai
    break
  fi
  if [ "${ELAPSED}" -ge "${WAIT_SECONDS}" ]; then
    echo "[bootstrap] ERROR: Database did not become ready within ${WAIT_SECONDS}s." >&2
    echo "[bootstrap] Last status: ${STATUS}" >&2
    exit 1
  fi
  echo "[bootstrap] Not ready yet (${ELAPSED}s elapsed). Retrying in ${POLL}s ..."
  sleep "${POLL}"
  ELAPSED=$((ELAPSED + POLL))
done

# ---------------------------------------------------------------------------
# 3. Create dmuser (run 000_dmuser.sql as SYSTEM)
# ---------------------------------------------------------------------------
echo "[bootstrap] Creating dmuser via 000_dmuser.sql ..."
"${CR}" exec -i "${CONTAINER_NAME}" \
  sqlplus -s "system/${ORACLE_PWD}@FREEPDB1" \
  < "${SCRIPT_DIR}/sql/000_dmuser.sql"

# ---------------------------------------------------------------------------
# 4. Apply course-extra DDL files as dmuser
#    Order matters: 002_query_trace_intent.sql ALTERs the QUERY_TRACES table
#    created by 002_query_traces.sql, so it must run immediately after it.
#    Without the intent column, Module 2's trace insert fails with
#    ORA-00904: "INTENT": invalid identifier. (The migration is idempotent.)
# ---------------------------------------------------------------------------
# The apply step connects as dmuser through course_lab.oracle_db, which reads
# ORACLE_MEMORY_DB_*; make the step self-sufficient by defaulting them here.
export ORACLE_MEMORY_DB_USER="${ORACLE_MEMORY_DB_USER:-dmuser}"
export ORACLE_MEMORY_DB_PASSWORD="${ORACLE_MEMORY_DB_PASSWORD:-${ORACLE_PWD}}"
export ORACLE_MEMORY_DB_CONNECT_STRING="${ORACLE_MEMORY_DB_CONNECT_STRING:-localhost:${ORACLE_PORT}/FREEPDB1}"

for sql_file in "${SCRIPT_DIR}/sql/001_skillbox.sql" \
                "${SCRIPT_DIR}/sql/002_query_traces.sql" \
                "${SCRIPT_DIR}/sql/002_query_trace_intent.sql" \
                "${SCRIPT_DIR}/sql/003_retrieval_traces.sql" \
                "${SCRIPT_DIR}/sql/004_voyage_manifest.sql" \
                "${SCRIPT_DIR}/sql/005_memory_graph.sql" \
                "${SCRIPT_DIR}/sql/006_skillbox_governance.sql" \
                "${SCRIPT_DIR}/sql/007_enhanced_skillbox.sql"; do
  if [ -f "${sql_file}" ]; then
    echo "[bootstrap] Applying ${sql_file} as dmuser ..."
    cd "${REPO_ROOT}" && "${PYBIN}" -m course_lab.oracle_db apply "${sql_file}"
  else
    echo "[bootstrap] WARNING: ${sql_file} not found — skipping." >&2
  fi
done

# ---------------------------------------------------------------------------
# 5. Print env vars for copy-paste
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Oracle Free container is ready.  Copy-paste these exports:"
echo "================================================================"
echo "export ORACLE_MEMORY_DB_USER=dmuser"
echo "export ORACLE_MEMORY_DB_PASSWORD=${ORACLE_PWD}"
echo "export ORACLE_MEMORY_DB_CONNECT_STRING=localhost:${ORACLE_PORT}/FREEPDB1"
echo "================================================================"
echo "[bootstrap] $(date -u +%Y-%m-%dT%H:%M:%SZ) — done"
