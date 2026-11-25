#!/usr/bin/env bash
# Sync between the Mac checkout and the GPU box (no git needed there).
#
#   scripts/sync.sh push            # Mac -> GPU: code only (no .venv, runs, outputs)
#   scripts/sync.sh pull            # GPU -> Mac: runs/ logs and metrics (skips checkpoints and adapters)
#   GPU_HOST=gpu5090-lan scripts/sync.sh push   # use the LAN alias when Tailscale is off
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${GPU_HOST:-gpu5090}"
REMOTE_DIR="${REMOTE_DIR:-src/trl_rl}"

case "${1:-}" in
  push)
    rsync -az --delete \
      --exclude .git/ --exclude .venv/ --exclude runs/ --exclude outputs/ \
      --exclude __pycache__/ --exclude .pytest_cache/ --exclude .DS_Store \
      "$REPO/" "$HOST:$REMOTE_DIR/"
    ;;
  pull)
    mkdir -p "$REPO/runs"
    rsync -az \
      --exclude 'checkpoint-*/' --exclude '.vllm_lora/' --exclude 'final_adapter/' --exclude '*.safetensors' \
      "$HOST:$REMOTE_DIR/runs/" "$REPO/runs/"
    ;;
  *)
    echo "usage: $0 push|pull" >&2
    exit 2
    ;;
esac
