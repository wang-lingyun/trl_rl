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

Boxed hint = `Please reason step by step, and put your final answer within \boxed{}.`

Takeaway: the accuracy reward only credits answers that math-verify can extract, so without an instruction the first
RL steps mostly teach the model to box its answer. Training prompts therefore carry the same hint, and runs are
compared against the base + hint number.

## Training runs

| Run | Steps | Wall time | Weight sync | Staleness | IS ratio | Train reward | Notes |
|---|---|---|---|---|---|---|---|
| smoke-2 | 5 (64 prompts) | 13 s train (+24 s vLLM start, warm cache) | ~0.28 s | 1-4 | ~1.00 | 0.19-0.25 | plain prompts |
