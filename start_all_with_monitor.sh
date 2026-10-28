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

# An open Redis port is not sufficient: a restored RDB can still be loading.
# Do not silently lose expected-worker registration while Redis is loading.
python - <<'PY'
import os, time, redis
r = redis.Redis(host=os.environ['REDIS_HOST'], port=int(os.environ['REDIS_PORT']),
                password=os.environ.get('REDIS_PASSWORD') or None,
                socket_connect_timeout=2, socket_timeout=2)
for attempt in range(60):
    try:
        if r.ping():
            break
    except redis.RedisError:
        pass
    time.sleep(1)
else:
    raise SystemExit('ERROR: Redis did not become ready for worker registration')
PY

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
        DEL "${REDIS_KEY_PREFIX}:expected_workers" > /dev/null
fi

echo "Starting GPU workers..."
for gpu in ${GPU_LIST}; do
    WORKER_ID="worker_gpu_${gpu}"
    echo "Launching ${WORKER_ID} on cuda:${gpu}"
    # Start workers serially; cold GPFS/FUSE mounts can fail transiently when
    # launched together. Preserve every failed attempt and retry only workers
    # that exited before registering. Never duplicate a live slow worker.
    WORKER_READY=no
    for worker_attempt in 1 2 3; do
    echo "Startup attempt ${worker_attempt}: ${WORKER_ID}"
    if [[ -n "${KERNELGYM_WORKER_CONTAINER}" ]]; then
        WORKER_ENROOT_ROOT="${KERNELGYM_ENROOT_ROOT}/jobs/kernelgym-${SLURM_JOB_ID:-manual}/${WORKER_ID}"
        mkdir -p \
            "${KERNELGYM_ENROOT_ROOT}/cache" \
            "${WORKER_ENROOT_ROOT}/data" \
            "${WORKER_ENROOT_ROOT}/runtime" \
            "${WORKER_ENROOT_ROOT}/tmp"
        CACHE_MOUNT_ARGS=()
        for cache_var in TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR XDG_CACHE_HOME; do
            cache_path="${!cache_var:-}"
            if [[ -n "${cache_path}" ]]; then
                mkdir -p "${cache_path}"
                CACHE_MOUNT_ARGS+=(--mount "${cache_path}:${cache_path}")
            fi
        done
        env \
            ENROOT_CACHE_PATH="${KERNELGYM_ENROOT_ROOT}/cache" \
            ENROOT_DATA_PATH="${WORKER_ENROOT_ROOT}/data" \
            ENROOT_RUNTIME_PATH="${WORKER_ENROOT_ROOT}/runtime" \
            ENROOT_TEMP_PATH="${WORKER_ENROOT_ROOT}/tmp" \
            enroot start \
                --root \
                --mount "${ROOT_DIR}:${ROOT_DIR}" \
                "${CACHE_MOUNT_ARGS[@]}" \
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
                --env TRITON_CACHE_DIR \
                --env TORCHINDUCTOR_CACHE_DIR \
                --env XDG_CACHE_HOME \
                --env CUDA_CACHE_PATH \
                "${KERNELGYM_WORKER_CONTAINER}" \
                python -m kernelgym.worker.single_worker \
                    --worker-id "${WORKER_ID}" \
                    --device "cuda:${gpu}" \
                    --persistent \
            >> "${ROOT_DIR}/${LOG_DIR}/worker_gpu_${gpu}.log" 2>&1 &
    else
        python -m kernelgym.worker.single_worker \
            --worker-id "${WORKER_ID}" --device "cuda:${gpu}" --persistent \
            >> "${ROOT_DIR}/${LOG_DIR}/worker_gpu_${gpu}.log" 2>&1 &
    fi
    WORKER_PID=$!
    WORKER_PIDS+=("${WORKER_PID}")
    write_pid_file

    echo "Worker PID: ${WORKER_PID}"
    if python - "${WORKER_ID}" "${WORKER_PID}" <<'PY'
import os, sys, time, redis
wid, pid = sys.argv[1], int(sys.argv[2])
r = redis.Redis(host=os.environ['REDIS_HOST'], port=int(os.environ['REDIS_PORT']),
                password=os.environ.get('REDIS_PASSWORD') or None,
                socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
key = os.environ['REDIS_KEY_PREFIX'] + ':worker:' + wid
deadline = time.monotonic() + 90
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        sys.exit(1)
    try:
        info = r.hgetall(key)
        if info.get('online') == 'true' and info.get('status') == 'online':
            sys.exit(0)
    except redis.RedisError:
        pass
    time.sleep(1)
raise SystemExit('ERROR: live worker did not register within 90 seconds')
PY
    then
        WORKER_READY=yes
        break
    fi
    if kill -0 "${WORKER_PID}" 2>/dev/null; then
        echo "ERROR: ${WORKER_ID} is alive but not ready; refusing a duplicate launch"
        exit 1
    fi
    wait "${WORKER_PID}" 2>/dev/null || true
    sleep "$((worker_attempt * 2))"
    done
    if [[ "${WORKER_READY}" != yes ]]; then
        echo "ERROR: ${WORKER_ID} failed all startup attempts"
        tail -n 30 "${ROOT_DIR}/${LOG_DIR}/worker_gpu_${gpu}.log"
        exit 1
    fi

    # Publish recovery metadata only after the worker is online, preventing
    # the monitor from racing the bounded startup retries above.
    python - "${WORKER_ID}" "${WORKER_PID}" "${gpu}" <<'PY'
import os, sys, time, socket, datetime, redis
wid, pid, gpu = sys.argv[1:]
r = redis.Redis(host=os.environ['REDIS_HOST'], port=int(os.environ['REDIS_PORT']),
                password=os.environ.get('REDIS_PASSWORD') or None,
                socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
prefix = os.environ['REDIS_KEY_PREFIX']
for attempt in range(30):
    try:
        with r.pipeline(transaction=True) as pipe:
            pipe.sadd(prefix + ':expected_workers', wid)
            pipe.hset(prefix + ':expected_worker:' + wid, mapping={
                'device': 'cuda:' + gpu, 'hostname': socket.gethostname(),
                'node_id': os.environ.get('NODE_ID', '')})
            pipe.hset(prefix + ':worker_process:' + wid, mapping={
                'pid': pid, 'device': 'cuda:' + gpu,
                'start_time': datetime.datetime.now().isoformat()})
            pipe.execute()
        if r.sismember(prefix + ':expected_workers', wid) and r.hget(prefix + ':worker_process:' + wid, 'pid') == pid:
            break
    except redis.RedisError:
        pass
    time.sleep(1)
else:
    raise SystemExit('ERROR: could not verify recovery metadata for ' + wid)
PY
done

python - ${GPU_LIST} <<'PY'
import os, sys, redis
r = redis.Redis(host=os.environ['REDIS_HOST'], port=int(os.environ['REDIS_PORT']),
                password=os.environ.get('REDIS_PASSWORD') or None,
                socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
prefix = os.environ['REDIS_KEY_PREFIX']
expected = {'worker_gpu_' + gpu for gpu in sys.argv[1:]}
registered = r.smembers(prefix + ':expected_workers')
missing = [wid for wid in expected if wid not in registered or
           r.hget(prefix + ':worker:' + wid, 'online') != 'true' or
           r.hget(prefix + ':worker:' + wid, 'status') != 'online']
if missing:
    raise SystemExit('ERROR: expected workers not online: ' + ', '.join(sorted(missing)))
print('Verified all ' + str(len(expected)) + ' expected GPU workers online with recovery metadata')
PY

echo "KernelGym started."
echo "Logs: ${ROOT_DIR}/${LOG_DIR}"
