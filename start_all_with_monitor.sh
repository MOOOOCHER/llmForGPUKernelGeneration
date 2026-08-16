#!/bin/bash

# Start KernelGym API server, worker monitor, and GPU workers.
# Assumes this script is run from any location; it will resolve the repo root.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUTO_CONFIGURE="${ROOT_DIR}/scripts/auto_configure.sh"
ENV_FILE="${KERNELGYM_ENV_FILE:-${ROOT_DIR}/.env}"
PID_FILE=""

LOG_DIR=""
LOG_DIR_OVERRIDE=""
AUTO_CONFIGURE_ARGS=()

cd "${ROOT_DIR}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            LOG_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --log-dir=*)
            LOG_DIR_OVERRIDE="${1#*=}"
            shift 1
            ;;
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --env-file=*)
            ENV_FILE="${1#*=}"
            shift 1
            ;;
        --pid-file)
            PID_FILE="$2"
            shift 2
            ;;
        --pid-file=*)
            PID_FILE="${1#*=}"
            shift 1
            ;;
        --use-indexed-ports)
            AUTO_CONFIGURE_ARGS+=("--use-indexed-ports")
            shift 1
            ;;
        --force-config)
            AUTO_CONFIGURE_ARGS+=("--force")
            shift 1
            ;;
        *)
            echo "Unknown argument: $1"
            shift 1
            ;;
    esac
done

WORKER_PIDS=()
write_pid_file() {
    if [ -z "${PID_FILE}" ]; then
        return
    fi
    mkdir -p "$(dirname "${PID_FILE}")"
    {
        [ -n "${API_PID:-}" ] && echo "${API_PID}"
        [ -n "${MONITOR_PID:-}" ] && echo "${MONITOR_PID}"
        for pid in "${WORKER_PIDS[@]}"; do
            echo "${pid}"
        done
    } > "${PID_FILE}"
}

if [ ! -f "${ENV_FILE}" ]; then
    echo "No .env found. Running auto configuration..."
    if [ ! -f "${AUTO_CONFIGURE}" ]; then
        echo "Auto configuration script not found at ${AUTO_CONFIGURE}"
        exit 1
    fi
    chmod +x "${AUTO_CONFIGURE}"
    "${AUTO_CONFIGURE}" "${AUTO_CONFIGURE_ARGS[@]}"
elif [ ${#AUTO_CONFIGURE_ARGS[@]} -gt 0 ]; then
    echo "Re-running auto configuration with explicit flags..."
    chmod +x "${AUTO_CONFIGURE}"
    "${AUTO_CONFIGURE}" "${AUTO_CONFIGURE_ARGS[@]}"
fi

set -o allexport
source "${ENV_FILE}"
set +o allexport

# The workflow reference cache and its in-flight deduplication live in the API
# process. RL job env files do not set API_WORKERS, so default them to one API
# process and keep all configured GPU evaluation workers below. Explicit
# deployments can still opt into multiple API processes.
API_WORKERS="${API_WORKERS:-1}"
export API_WORKERS

if [ -z "${LOG_DIR}" ]; then
    LOG_DIR="${LOG_DIR:-logs}"
fi
if [ -n "${LOG_DIR_OVERRIDE}" ]; then
    LOG_DIR="${LOG_DIR_OVERRIDE}"
fi

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
REDIS_KEY_PREFIX="${REDIS_KEY_PREFIX:-kernelgym}"

PYTHONPATH="${ROOT_DIR}"
export PYTHONPATH
export KERNELGYM_REPO_ROOT="${ROOT_DIR}"
KERNELGYM_WORKER_CONTAINER="${KERNELGYM_WORKER_CONTAINER:-}"
KERNELGYM_ENROOT_ROOT="${KERNELGYM_ENROOT_ROOT:-/sc/scratch/zongxiong.chen/daniel/.enroot}"
if [[ -n "${KERNELGYM_WORKER_CONTAINER}" && ! -f "${KERNELGYM_WORKER_CONTAINER}" ]]; then
    echo "KernelGym worker container not found: ${KERNELGYM_WORKER_CONTAINER}"; exit 1
fi

port_is_open() {
    local host="$1"
    local port="$2"
    python - "$host" "$port" <<PY
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=1):
        pass
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

echo "Checking Redis..."
if ! port_is_open "${REDIS_HOST}" "${REDIS_PORT}"; then
    if [ "${REDIS_HOST}" != "localhost" ] && [ "${REDIS_HOST}" != "127.0.0.1" ]; then
        echo "Redis is not reachable at ${REDIS_HOST}:${REDIS_PORT}. Please start it first."
        exit 1
    fi
    echo "Starting Redis on ${REDIS_HOST}:${REDIS_PORT}..."
    if [ -n "${REDIS_PASSWORD}" ]; then
        redis-server --port "${REDIS_PORT}" --requirepass "${REDIS_PASSWORD}" --daemonize yes
    else
        redis-server --port "${REDIS_PORT}" --daemonize yes
    fi
    sleep 2
fi

echo "Starting API server..."
python -m kernelgym.server.api.server > "${ROOT_DIR}/${LOG_DIR}/api_server.log" 2>&1 &
API_PID=$!
echo "API server PID: ${API_PID}"
write_pid_file

echo "Starting worker monitor..."
python -m kernelgym.worker.worker_monitor --persistent > "${ROOT_DIR}/${LOG_DIR}/worker_monitor.log" 2>&1 &
MONITOR_PID=$!
echo "Worker monitor PID: ${MONITOR_PID}"
write_pid_file

sleep 2

GPU_LIST="$(python - <<'PY'
import os, json
raw = os.environ.get("GPU_DEVICES", "")
if not raw:
    print("0")
    raise SystemExit(0)
try:
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        print(" ".join(str(x) for x in parsed))
    else:
        print(str(parsed))
except Exception:
    print(" ".join([x.strip() for x in raw.split(",") if x.strip()]))
PY
)"

if ! command -v redis-cli >/dev/null 2>&1; then
    echo "Warning: redis-cli not found; worker monitor persistent metadata will be skipped."
fi

REDIS_AUTH_ARGS=()
if [ -n "${REDIS_PASSWORD}" ]; then
    REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}")
fi

if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
        DEL "${REDIS_KEY_PREFIX}:expected_workers" > /dev/null 2>&1 || true
fi

echo "Starting GPU workers..."
for gpu in ${GPU_LIST}; do
    WORKER_ID="worker_gpu_${gpu}"
    echo "Launching ${WORKER_ID} on cuda:${gpu}"
    if [[ -n "${KERNELGYM_WORKER_CONTAINER}" ]]; then
        WORKER_ENROOT_ROOT="${KERNELGYM_ENROOT_ROOT}/jobs/kernelgym-${SLURM_JOB_ID:-manual}/${WORKER_ID}"
        mkdir -p \
            "${KERNELGYM_ENROOT_ROOT}/cache" \
            "${WORKER_ENROOT_ROOT}/data" \
            "${WORKER_ENROOT_ROOT}/runtime" \
            "${WORKER_ENROOT_ROOT}/tmp"
        env \
            ENROOT_CACHE_PATH="${KERNELGYM_ENROOT_ROOT}/cache" \
            ENROOT_DATA_PATH="${WORKER_ENROOT_ROOT}/data" \
            ENROOT_RUNTIME_PATH="${WORKER_ENROOT_ROOT}/runtime" \
            ENROOT_TEMP_PATH="${WORKER_ENROOT_ROOT}/tmp" \
            enroot start \
                --root \
                --mount "${ROOT_DIR}:${ROOT_DIR}" \
                --env REDIS_HOST \
                --env REDIS_PORT \
                --env REDIS_PASSWORD \
                --env REDIS_KEY_PREFIX \
                --env API_HOST \
                --env API_PORT \
                --env GPU_DEVICES \
                --env NODE_ID \
                --env KERNELGYM_TASK_ARCHIVE_DIR \
                --env PYTHONPATH \
                "${KERNELGYM_WORKER_CONTAINER}" \
                python -m kernelgym.worker.single_worker \
                    --worker-id "${WORKER_ID}" \
                    --device "cuda:${gpu}" \
                    --persistent \
            > "${ROOT_DIR}/${LOG_DIR}/worker_gpu_${gpu}.log" 2>&1 &
    else
        python -m kernelgym.worker.single_worker \
            --worker-id "${WORKER_ID}" --device "cuda:${gpu}" --persistent \
            > "${ROOT_DIR}/${LOG_DIR}/worker_gpu_${gpu}.log" 2>&1 &
    fi
    WORKER_PID=$!
    WORKER_PIDS+=("${WORKER_PID}")
    write_pid_file

    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            SADD "${REDIS_KEY_PREFIX}:expected_workers" "${WORKER_ID}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:expected_worker:${WORKER_ID}" \
            device "cuda:${gpu}" hostname "$(hostname)" node_id "${NODE_ID:-}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:worker_process:${WORKER_ID}" \
            pid "${WORKER_PID}" start_time "$(date -Iseconds)" device "cuda:${gpu}" > /dev/null 2>&1 || true
    fi

    echo "Worker PID: ${WORKER_PID}"
    sleep 0.3
done

echo "KernelGym started."
echo "Logs: ${ROOT_DIR}/${LOG_DIR}"
