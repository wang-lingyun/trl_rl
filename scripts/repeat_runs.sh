#!/usr/bin/env bash
# Measure run-to-run spread: for each seed and sync mode, train (scripts/run_1gpu.sh) and then evaluate on GSM8K
# test, one after another. Each result is one JSON line in runs/repeat-summary.jsonl.
#
#   scripts/repeat_runs.sh                       # modes "file ipc", seeds "43 44", 200 steps
#   MODES="lora file ipc" SEEDS="43 44 45" scripts/repeat_runs.sh --max-steps 200
#
# Extra args go to the trainer. SYNC=ipc runs are skipped while another login user has processes on the machine,
# because the IPC server accepts pickles on localhost (see scripts/serve_vllm.sh).
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export VIRTUAL_ENV="$REPO/.venv"
PY="$REPO/.venv/bin/python"
MODES="${MODES:-file ipc}"
SEEDS="${SEEDS:-43 44}"
SUMMARY="runs/repeat-summary.jsonl"
mkdir -p runs/eval

others_logged_in() {  # any process owned by another regular (uid >= 1000) user?
  ps -eo uid= | awk -v me="$(id -u)" '$1 >= 1000 && $1 < 60000 && $1 != me {found=1} END {exit !found}'
}

for seed in $SEEDS; do
  for mode in $MODES; do
    name="rep-$mode-s$seed"
    if [ "$mode" = ipc ] && others_logged_in; then
      echo "$(date +%T) skip $name: another user has processes (IPC server would accept pickles on localhost)"
      continue
    fi
    echo "$(date +%T) train $name"
    if ! RUN_DIR="runs/$name" SYNC="$mode" scripts/run_1gpu.sh --max-steps 200 --seed "$seed" --attn-backend fa2 "$@" \
        > "runs/$name.out" 2>&1; then
      echo "$(date +%T) FAILED $name (see runs/$name.out)"
      continue
    fi
    if [ "$mode" = lora ]; then target=(--adapter "runs/$name/output/final_adapter"); else target=(--model "runs/$name/output/final_model"); fi
    echo "$(date +%T) eval $name"
    "$PY" eval_gsm8k.py --boxed-hint "${target[@]}" --out "runs/eval/$name.jsonl" > "runs/eval/$name.log" 2>&1 \
      || { echo "$(date +%T) EVAL FAILED $name"; continue; }
    "$PY" - "$name" "$mode" "$seed" >> "$SUMMARY" <<'EOF'
import json, sys
name, mode, seed = sys.argv[1:]
ev = next(json.loads(l) for l in open(f"runs/eval/{name}.log") if l.startswith('{"model"'))
hist = json.load(open(f"runs/{name}/output/trainer_state.json"))["log_history"]
syncs = [x["perf/weight_sync_s"] for x in hist if "perf/weight_sync_s" in x]
runtime = [x["train_runtime"] for x in hist if "train_runtime" in x][-1]
print(json.dumps({"name": name, "mode": mode, "seed": int(seed), "accuracy": ev["accuracy"], "truncated": ev["truncated"],
                  "train_runtime_s": round(runtime, 1), "weight_sync_mean_s": round(sum(syncs) / len(syncs), 3)}))
EOF
    echo "$(date +%T) done $name: $(tail -n 1 "$SUMMARY")"
  done
done
echo "$(date +%T) ALL DONE"
