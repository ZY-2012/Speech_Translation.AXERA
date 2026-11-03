#!/usr/bin/env bash
# Start the streaming speech-translation demo on the AX650 board.
#   PORT=8096 CONFIG=config/board.yaml bash scripts/run_board.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/runtime/python/bin/python3}"
export PATH="${ROOT}/runtime/python/bin:${PATH}"
export PYTHONPATH="${ROOT}/runtime/pylibs:${PYTHONPATH:-}"

PORT="${PORT:-8096}"
CONFIG="${CONFIG:-config/board.yaml}"
export LD_LIBRARY_PATH="${ROOT}/third_party/asr/lib:${ROOT}/runtime/pylibs:/soc/lib:/opt/lib:${LD_LIBRARY_PATH:-}"
export AX_SENSEVOICE_SHARE_DIR="${AX_SENSEVOICE_SHARE_DIR:-${ROOT}/models/asr_official/sensevoice_ax650}"
mkdir -p .work/logs

# Stale workers hold the NPU; kill them before a restart.  `pkill -x` matches
# the process name, so it cannot kill this shell.
pkill -9 -x qwen3-worker 2>/dev/null || true
pkill -9 -x stream-asr-worker 2>/dev/null || true
pkill -9 -x zipvoice_worker 2>/dev/null || true
pkill -9 -f "third_party/asr/bin/stream-asr-worker" 2>/dev/null || true
pkill -9 -f "zf_asr_worker.py" 2>/dev/null || true
pkill -9 -f "punc_worker.py" 2>/dev/null || true

# Stop the previous demo server.  Buildroot has no `ss`/`curl`; use the pid
# file first, then a targeted process match for a stale instance.
if [[ -f .work/server.pid ]]; then
  kill -9 "$(cat .work/server.pid)" 2>/dev/null || true
  rm -f .work/server.pid
fi
pkill -9 -f "app/server.py.*--port ${PORT}" 2>/dev/null || true
sleep 1

TLS_PORT="${TLS_PORT:-8444}"
TLS_OPTS=""
if [[ -f config/tls/server.crt && -f config/tls/server.key ]]; then
  TLS_OPTS="--tls-port ${TLS_PORT} --tls-cert ${ROOT}/config/tls/server.crt --tls-key ${ROOT}/config/tls/server.key"
fi

echo "[run] config=${CONFIG} port=${PORT}"
nohup "${PYTHON}" app/server.py --config "${CONFIG}" --port "${PORT}" ${TLS_OPTS} \
  > .work/logs/server.log 2>&1 &
echo $! > .work/server.pid

for _ in $(seq 1 900); do
  if wget -q -O /dev/null "http://127.0.0.1:${PORT}/health" 2>/dev/null; then
    echo "[run] ready: http://<board-ip>:${PORT}/"
    if [[ -n "${TLS_OPTS}" ]]; then
      echo "[run] microphone HTTPS: https://<board-ip>:${TLS_PORT}/  (accept self-signed)"
    fi
    echo "[run] or tunnel: ssh -L ${PORT}:localhost:${PORT} root@<board-ip>"
    exit 0
  fi
  sleep 1
done

echo "[run] startup timed out; last log lines:" >&2
tail -30 .work/logs/server.log >&2
exit 1
