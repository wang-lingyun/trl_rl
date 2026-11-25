# Research notes (2026-11-24)

Sources: the Space [trl-lib/async-grpo-stable](https://huggingface.co/spaces/trl-lib/async-grpo-stable)
(article published 2026-11-13, TRL v1.18.0), the TRL docs page `async_grpo_trainer.md` at tag `v1.18.2`,
`examples/async_grpo_math/async_grpo_math.py`, and the `trl-1.18.2` wheel source. Line numbers below are from that wheel.

## What the article says

- `AsyncGRPOTrainer` / `AsyncGRPOConfig` became stable API in TRL 1.18 (`from trl import ...`).
- Architecture: a vLLM server generates; a rollout worker (a spawned process with no GPU) sends prompts, scores
  completions and puts them in a queue; the trainer takes samples from the queue, trains, and pushes new weights
  back to vLLM. Samples can be a few steps old (about 2 on average in their run). The trainer corrects for that
  with an importance ratio against the sampler's logprobs (`--logprobs-mode processed_logprobs`).
- Groups where every completion got the same reward are dropped (no signal).
- Benchmark: Qwen3.5-2B on DeepMath, 2×H200, 45 min. Async did 215 optimizer steps vs 87 for sync GRPO, ~3× the
  tokens, and reward 0.43 vs 0.25 (one run).
- Hardware: "It needs vLLM on its own GPUs, so at least two." The docs say: "Make sure that the server is using
  different GPUs than the trainer, otherwise you may run into NCCL errors." There is no colocate mode.

## Why two GPUs, and the way around it

- Full-weight sync is `trl/generation/weight_transfer.py`. It sets up an NCCL group where the trainer is rank 0 and
  the vLLM workers are ranks 1..N, then broadcasts packed tensors. NCCL rejects two ranks on the same GPU.
- **Adapter-only LoRA sync skips NCCL entirely.** In `async_grpo_trainer.py` (~3302): if `self._lora_sync`, then
  `self.weight_transfer = None` ("The adapter reaches the server as a directory path over HTTP, so there is no NCCL
  transfer group to build"). `_sync_weight_lora` (~3779) saves the adapter under `<output_dir>/.vllm_lora/trl-policy-vN`
  and calls `POST /v1/load_lora_adapter`. Rollouts then request model `trl-policy-vN`.
- `select_adapter_sync` (~2733) turns it on only when the model is a PeftModel, the server reports a
  `lora_config` (`--enable-lora`), and `data_parallel_size == 1`. The adapter must be plain LoRA (no DoRA,
  `modules_to_save`, bias, lm_head or embeddings) and fit within `--max-lora-rank`, and the server needs
  `--max-loras >= max_staleness + 2`. Otherwise it falls back to merged sync, which uses NCCL, so our script
  checks `trainer._lora_sync` and exits.
- The server also needs `VLLM_SERVER_DEV_MODE=1` (for `/pause`, `/resume`, `/server_info`) and
  `VLLM_ALLOW_RUNTIME_LORA_UPDATING=1` (for `/v1/load_lora_adapter`). Runtime adapter loading works with a single
  API server only, so there is no `--api-server-count 8`. That makes `score_centering=True` (top-8 logprobs per
  token) costly, so the article suggests `score_centering=False` for LoRA.
- Cost: vLLM generates ~1.33-1.39× slower per token with a LoRA adapter than with merged weights (TRL PR 7017).

## Blackwell consumer GPU (sm_120) risk

- Training uses tree packing: `attn_implementation="tree_fa3"` → `tree_attention` → `_flash_attn()` (~306), which
  loads `kernels-community/flash-attn4` when compute capability ≥ 10, otherwise `flash-attn3`. The docs mention
  only "NVIDIA Blackwell GPUs" generally.
- FA4 is written for SM100 (tcgen05 / TMEM). The RTX 5090 is SM120 and lacks those instructions, so FA4 may fail.
  FA3 is Hopper-only. `attn_compat.py` patches `agt._flash_attn` with the first backend that passes a GPU probe
  (FA4 → hub FA2 → SDPA per-segment loop). The SDPA path is checked on CPU in `tests/`.
- The other kernels are Triton (`trl/kernels/*`, fused LM head), which supports sm_120.

## Config facts worth remembering (AsyncGRPOConfig, 1.18.2)

- Defaults: `num_generations=8`, `max_completion_length=2048`, `learning_rate=1e-6` (constant), `bf16=True`,
  `gradient_checkpointing=True`, `dtype="float32"` (weights load in fp32; compute in bf16), `max_staleness=4`,
  `weight_sync_steps=1`, `max_concurrent_rollouts=512`, `score_centering=True`, `num_sampler_logprobs=8`,
  `vllm_importance_sampling_clip_min/max=0.5/5.0`, `token_budget=None` (→ server `max_model_len`).
- `top_p < 1` / `top_k` needs vLLM ≥ 0.28 started with `--return-sampling-mask --enable-scale-out`. We keep the
  defaults (temperature 1, no truncation).
- The rollout worker is spawned and re-imports the script: keep the code under `if __name__ == "__main__":` and make
  reward functions module-level.
- Rollout concurrency adapts to vLLM's KV-cache usage (`/metrics`): it starts at `kv_tokens // max_model_len`.

## Versions resolved (uv.lock)

trl 1.18.2, vllm 0.33.0 (trl[vllm] pins `<0.34`), torch 2.14.0 (CUDA 13.0 → NVIDIA driver ≥ 580),
transformers 5.20.2, peft 0.22.0, kernels 0.17.3, accelerate 1.16.0, math-verify 0.9.0.

## Smallest Qwen candidates (HF API, 2026-11-24)

| Model | Params | Type | Note |
|---|---|---|---|
| Qwen/Qwen2.5-0.5B-Instruct | 0.49B | text | smallest, older generation |
| **Qwen/Qwen3-0.6B** | 0.75B (with embeddings) | text | used by TRL's async GSM8K example; default here |
| Qwen/Qwen3.5-0.8B | 0.87B | image-text-to-text | gated DeltaNet hybrid; needs transformers ≥ 5.15 tree-GDN patch |
