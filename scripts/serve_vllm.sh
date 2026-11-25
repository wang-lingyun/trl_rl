#!/usr/bin/env bash
# Start the vLLM generation server for single-GPU async GRPO. It shares the GPU with the trainer, so
# its memory is capped (GPU_MEM_UTIL) and weights come in as LoRA adapters over HTTP, not over NCCL.
#
#   scripts/serve_vllm.sh                      # defaults below
#   GPU_MEM_UTIL=0.4 scripts/serve_vllm.sh     # override any setting through the environment
#   scripts/serve_vllm.sh --enforce-eager      # extra args go straight to `vllm serve`
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_BIN="${VLLM_BIN:-$REPO/.venv/bin/vllm}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HOST="${HOST:-127.0.0.1}"          # localhost only: DEV_MODE exposes pause / weight / adapter endpoints
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.35}" # fraction of the WHOLE GPU vLLM may take (weights + KV cache + graphs)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}" # >= trainer --lora-r; one of 1 8 16 32 64 128 256 320 512
MAX_LORAS="${MAX_LORAS:-6}"          # >= trainer --max-staleness + 2

# Pause/resume + /server_info need DEV_MODE; /v1/load_lora_adapter needs runtime LoRA updating.
export VLLM_SERVER_DEV_MODE=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=1

# --weight-transfer-config is kept as upstream recommends; it is only used if the trainer falls back to
# merged sync, which train_async_grpo_1gpu.py refuses to do on one GPU.
# No --api-server-count: runtime adapter loading works with a single API server only.
exec "$VLLM_BIN" serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --logprobs-mode processed_logprobs \
  --weight-transfer-config '{"backend":"nccl"}' \
  --enable-lora \
  --max-lora-rank "$MAX_LORA_RANK" \
  --max-loras "$MAX_LORAS" \
  "$@"
