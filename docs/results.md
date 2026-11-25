# Results

All runs: one RTX 5090 (32 GB), vLLM 0.33.0 at `--gpu-memory-utilization 0.35`, TRL 1.18.2 AsyncGRPOTrainer,
Qwen/Qwen3-0.6B + LoRA r=32 (all-linear), FA2 tree attention. Eval = `eval_gsm8k.py`: GSM8K test (1319), greedy,
thinking off, scored with `trl.rewards.accuracy_reward` (math-verify).

## GSM8K test accuracy

| Model | Prompt | `\boxed` in output | Accuracy | Notes |
|---|---|---|---|---|
| base | plain question | 44% | 25.5% | 24.9% on an earlier run (greedy batches aren't bit-exact) |
| base | + boxed hint | 99% | **64.1%** | fair baseline |
| smoke-2 adapter (5 steps, plain prompts) | plain question | 85% | 51.9% | the gain is format: 333/413 newly correct answers were already in the base output, just not extractable |
| smoke-2 adapter | + boxed hint | 99.5% | 63.7% | no math gain after 5 steps, as expected |
| base (rerun) | + boxed hint | 99.2% | 64.75% | second base run, to measure run-to-run noise |
| **r200-hint adapter** (200 steps, hint prompts) | + boxed hint | 97.2% | **67.4%** | +2.7 to +3.3 pts; paired McNemar vs the two base runs: 170 vs 127 (p=0.015), 165 vs 130 (p=0.048) |
| r200-hint adapter | plain question | 43.9% | 27.6% | trained with the hint, so plain-prompt format didn't change (base 25.5%) |

The r200 adapter hits the 1024-token cap more often under greedy decoding: 40 truncated vs 8 for base, and 36 long
outputs with no final `\boxed`. That's where to look next (length penalty, larger `max_completion_length`, or more steps).

Boxed hint = `Please reason step by step, and put your final answer within \boxed{}.`

Takeaway: the accuracy reward only credits answers that math-verify can extract, so without an instruction the first
RL steps mostly teach the model to box its answer. Training prompts therefore carry the same hint, and runs are
compared against the base + hint number.

## Training runs

| Run | Steps | Wall time | Weight sync | Staleness | IS ratio | Train reward | Notes |
|---|---|---|---|---|---|---|---|
| smoke-2 | 5 (64 prompts) | 13 s train (+24 s vLLM start, warm cache) | ~0.28 s | 1-4 | ~1.00 | 0.19-0.25 | plain prompts |
| r200-hint | 200 (full train set, 32 samples/step) | 418 s train (~2.1 s/step) | 0.28-0.37 s | mean 0.4-4, max 4 | 0.998-1.001 | ~0.6 at the start, 0.73 over the last 10 steps (noisy) | hint prompts, lr 5e-5, `score_centering=False`; peak VRAM 16.0 / 32.6 GB (vLLM + trainer), mean GPU util 70% |

## Full fine-tuning (no LoRA)

Same task and prompts, all weights trained, synced to vLLM on the same GPU without NCCL (`weight_sync.py`).

| Run | Sync | Steps | Weight sync | IS ratio | Peak VRAM | Outcome |
|---|---|---|---|---|---|---|
| full-file-smoke | file | 5 (64 prompts) | 0.43-0.53 s | 0.999-1.005 | 24.0 GB | works; final_model saved |
| full-ipc-smoke2 | ipc | 5 (64 prompts) | 0.07 s | 0.993-1.000 | 24.9 GB | works (first attempt failed: `pidfd_getfd` blocked by ptrace_scope=1 under expandable segments) |
| r200-full-file, lr 1e-5 | file | 200 | ~0.45 s | 0.998-1.001 throughout | ~24 GB | **diverged after ~step 120**: completion length 330 → 600, clipped 2% → 30%, entropy 0.25 → 0.52, train reward 0.62 → 0.33. GSM8K test **15.6%** (542/1319 truncated) |
| **r200-full-file-lr2e-6** | file | 200 (395 s) | 0.44-0.49 s | 1.000 throughout | 24.1 GB | stable: train reward 0.61 → 0.84, length 260-360, clipped ≤ 6%. GSM8K test **69.75%** (66 truncated) |
| r200-full-ipc-lr2e-6 | ipc | 200 (**301 s**) | 0.07-0.11 s (mean 0.089) | 1.000 throughout | 25.6 GB | stable: train reward 0.70 → 0.77, length 300-350, clipped ≤ 2%. GSM8K test 65.35% (44 truncated) |

**IPC vs file, same settings:** IPC trains 24% faster (18 s of total sync vs 94 s over 200 syncs), and vLLM serves the
trainer's weights at least as faithfully: mean log-prob difference 0.0128 vs 0.0137, masked tokens 0.00006 vs
0.0001, mean per-step max ratio 1.52 vs 1.56. The diverged lr 1e-5 run shows 0.045 / 0.007 / 5.3 on the same metrics,
so they do react. The two final models still differ on the test set (65.35% vs 69.75%; paired 117 vs 175, p=0.0008),
but that compares two model instances, not two sync methods. Async RL runs diverge even with the same seed, and the
sync metrics rule out IPC delivering wrong weights. Repeated runs per mode are needed to measure that run-to-run
spread; this IPC run isn't significantly above base (165 vs 149, p=0.4).

### Run-to-run spread (3 seeds per mode, `scripts/repeat_runs.sh`)

Full FT, lr 2e-6, 200 steps, boxed hint; seeds 43 and 44 added to the seed-42 runs above. Runs alternated between
modes.

| Mode | Seed 42 | Seed 43 | Seed 44 | Mean ± sd | Train time (mean) | Sync per step |
|---|---|---|---|---|---|---|
| file | 69.75% | 69.07% | 69.07% | **69.30 ± 0.39%** | 385 s | 0.46-0.47 s |
| ipc | 65.35% | 69.52% | 68.46% | **67.78 ± 2.17%** | 304 s | 0.088-0.089 s |

- File vs IPC: 1.5 points apart, exact permutation test over all 20 splits p = 0.30, so **no detectable difference**.
  IPC's lower mean comes from its seed-42 run; its other two runs sit next to file's.
- Weight delivery matches in all 6 runs: mean log-prob difference 0.0125-0.0141, masked tokens ≤ 1e-4, mean
  importance ratio 0.9999-1.0001.
- All 6 full-FT runs beat base (64.1 / 64.75%); mean 68.54 ± 1.62%.
- IPC saves ~80 s (~21%) of a ~6.5-minute run.

GSM8K test (1319, greedy, boxed hint), paired McNemar on the same questions:

| Model | Accuracy | vs base run 1 | vs base run 2 | vs LoRA r200 |
|---|---|---|---|---|
| base | 64.1% / 64.75% | | | |
| LoRA r200 (lr 5e-5) | 67.4% | 170 vs 127, p=0.015 | 165 vs 130, p=0.048 | |
| **full FT r200 (lr 2e-6)** | **69.75%** | 178 vs 104, p=1e-5 | 175 vs 109, p=1e-4 | 160 vs 129, p=0.08 |

So full FT at lr 2e-6 is clearly better than base. It's probably better than LoRA after the same 200 steps, but
one run each isn't enough to be sure.

The lr 1e-5 failure is an optimization problem, not a sync problem. The importance ratio stayed at ~1.00 for all
200 steps, so vLLM always served the trainer's current weights. 1e-5 is the LR TRL's full-FT GSM8K example uses,
but that example runs TRL's default score-centering loss on two GPUs; here it's the clipped loss
(`score_centering=False`). Watch `completions/clipped_ratio` and `entropy`: they moved ~20 steps before reward fell.

Sync cost is small on one GPU: the adapter goes to vLLM in ~0.3 s per step. The importance ratio stays at ~1.0, so the
trainer (FA2, LoRA, bf16 autocast) and vLLM (bf16 + LoRA) agree on log-probabilities. Peak VRAM of 16 GB leaves room to
raise `GPU_MEM_UTIL`, the batch size, or the model size (e.g. Qwen3-1.7B).
