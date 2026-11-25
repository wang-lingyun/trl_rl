#!/usr/bin/env bash
# One command for a single-GPU run: vLLM server in the background, trainer in the foreground, same GPU.
# Logs and outputs go to runs/<timestamp>/. Extra args go to train_async_grpo_1gpu.py.
#
#   scripts/run_1gpu.sh --max-steps 5 --train-size 64     # smoke test
#   scripts/run_1gpu.sh                                   # default 200-step run
#
# For long runs start it detached so an SSH drop doesn't kill it, e.g.
#   setsid nohup scripts/run_1gpu.sh > runs/last.out 2>&1 &
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PY:-$REPO/.venv/bin/python}"
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export VIRTUAL_ENV="$REPO/.venv"
PORT="${PORT:-8000}"
URL="http://127.0.0.1:$PORT"

RUN_DIR="${RUN_DIR:-runs/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR"
echo "run dir: $RUN_DIR"

if "$PY" - "$URL" <<'EOF' 2>/dev/null; then
import sys, urllib.request
urllib.request.urlopen(sys.argv[1] + "/health", timeout=3)
EOF
  echo "something already answers on $URL; stop it first (or set PORT=...)" >&2
  exit 1
fi

PORT="$PORT" "$REPO/scripts/serve_vllm.sh" > "$RUN_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM pid $VLLM_PID (log: $RUN_DIR/vllm.log)"

cleanup() {
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "stopping vLLM ($VLLM_PID)"
    kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

if ! "$PY" "$REPO/scripts/wait_for_server.py" --url "$URL" --timeout 900 --pid "$VLLM_PID"; then
  tail -n 60 "$RUN_DIR/vllm.log" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee "$RUN_DIR/gpu_mem_after_vllm.csv"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  "$PY" "$REPO/train_async_grpo_1gpu.py" \
  --vllm-url "$URL" \
  --output-dir "$RUN_DIR/output" \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
