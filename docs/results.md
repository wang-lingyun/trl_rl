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

Sync cost is small on one GPU: the adapter goes to vLLM in ~0.3 s per step. The importance ratio stays at ~1.0, so the
trainer (FA2, LoRA, bf16 autocast) and vLLM (bf16 + LoRA) agree on log-probabilities. Peak VRAM of 16 GB leaves room to
raise `GPU_MEM_UTIL`, the batch size, or the model size (e.g. Qwen3-1.7B).
