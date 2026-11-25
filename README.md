# trl_rl: async GRPO on a single RTX 5090

A minimal setup for asynchronous RL (TRL `AsyncGRPOTrainer`) where the **vLLM generation server and the
trainer share one GPU**. Task: Qwen3-0.6B on GSM8K with the math-verify accuracy reward, trained either as a
LoRA adapter or with **full fine-tuning**.

Based on the Hugging Face write-up [AsyncGRPOTrainer is stable](https://huggingface.co/spaces/trl-lib/async-grpo-stable)
(TRL 1.18). Upstream says it "needs vLLM on its own GPUs, so at least two". This repo works around that.

## Why one GPU needs changes

| Upstream assumption | Problem on one GPU | What this repo does |
|---|---|---|
| Trainer pushes full weights to vLLM over **NCCL** after each step | NCCL refuses two ranks on the same device ("Duplicate GPU detected") | Sync without NCCL, in one of three ways (`SYNC=lora\|file\|ipc`, see [Weight sync modes](#weight-sync-modes)). |
| vLLM owns its GPU | vLLM grabs 90% of VRAM by default | `--gpu-memory-utilization 0.35` (~11 GB of 32 GB), and the trainer uses the rest |
| Tree-packed training uses FlashAttention-4 on Blackwell | **Measured on the 5090 (sm_120): FA4's forward is right but its dK/dV are wrong (rel. error 0.3-1.3) whenever a batch packs more than one sequence**, which is always. There is no error, just bad gradients | `attn_compat.py` checks FA4, then FA2 (hub v4), then an SDPA fallback on the real GPU, forward and backward against a float32 reference, and installs the first that passes. On the 5090 that is **FA2** |

Generation and training still run at the same time (async): the rollout worker keeps sending requests
to vLLM while the trainer runs steps on queued samples. Each sync is a short pause plus the weight transfer.
Both processes time-slice the one GPU, so the speedup over sync GRPO is smaller than on two GPUs.

## Weight sync modes

There are four ways to get trained weights into a vLLM server on the same GPU. The first three are async and built in:

| `SYNC=` | Trains | How weights reach vLLM | Sync time (0.6B, measured) | Peak VRAM | Catch |
|---|---|---|---|---|---|
| `lora` (default) | LoRA adapter | TRL's adapter-only sync: adapter files + `POST /v1/load_lora_adapter` | ~0.3 s | 16 GB | vLLM generates ~1.35× slower with an adapter |
| `file` | all weights | bf16 safetensors in a private `/dev/shm` dir, then vLLM's in-place `reload_weights` via `POST /collective_rpc` | ~0.45 s | 24 GB | a host round-trip per sync (grows with model size); use `--weight-sync-steps N` to sync less often |
| `ipc` | all weights | vLLM's CUDA-IPC backend: vLLM maps the trainer's packed GPU buffer | ~0.07 s | 25 GB | handles are pickled, so the server runs with `VLLM_ALLOW_INSECURE_SERIALIZATION=1`, and **any local user who can reach the port can run code as you**. Only for single-user machines. Also needs the classic CUDA allocator (no expandable segments, because of Ubuntu's ptrace restriction) |

The fourth way is TRL's synchronous `GRPOTrainer` with `vllm_mode="colocate"`. It's officially supported for full
fine-tuning on one GPU, but generation and training take turns, so it isn't async.

What doesn't work: NCCL between two processes on one GPU (refused), and splitting the card with MIG (not supported
on GeForce). A fifth option would be a custom vLLM transfer plugin (e.g. IPC handles without pickle), which is
safe and fast but more code than a minimal setup needs.

```bash
SYNC=file scripts/run_1gpu.sh --max-steps 200     # full fine-tuning, safe default (lr 2e-6)
SYNC=ipc  scripts/run_1gpu.sh --max-steps 200     # full fine-tuning, fastest sync, single-user machines only
```

Result on GSM8K test (200 steps, ~7 min): base 64.1% → LoRA 67.4% → **full fine-tuning 69.75%** (`SYNC=file`,
lr 2e-6). At lr 1e-5, full fine-tuning diverged after ~120 steps (15.6%). Details in [docs/results.md](docs/results.md).

Full fine-tuning saves `runs/<ts>/output/final_model/`. Evaluate it with
`.venv/bin/python eval_gsm8k.py --boxed-hint --model runs/<ts>/output/final_model`.

## Will the smallest Qwen work?

Yes. **Qwen3-0.6B** is the default: text-only, plain attention, and TRL's own async GSM8K example uses it.
Rough memory budget on 32 GB:

- vLLM (0.35 × 32 GB ≈ 11 GB): bf16 weights ~1.2 GB, CUDA graphs and LoRA slots ~1-2 GB, and a KV cache of
  ~8 GB ≈ 70k tokens at ~112 KB/token, about 35 full 2048-token sequences at once.
- LoRA trainer: fp32 base weights ~2.4 GB (frozen), LoRA r=32 + Adam states < 0.5 GB, activations with gradient
  checkpointing for up to 2048 tokens per micro-batch, plus a chunked LM head. A few GB in total. Measured peak
  for vLLM + trainer: 16 GB of 32.6 GB.
- Full-fine-tuning trainer: fp32 weights + gradients + Adam states ≈ 9.6 GB plus activations. Measured peak for
  vLLM + trainer: 24-25 GB of 32.6 GB.

Other small options: Qwen2.5-0.5B-Instruct (smaller, older) works the same way. Qwen3.5-0.8B is a
vision-language model with gated-DeltaNet layers. TRL supports it but it has more moving parts, so it's
a later experiment.

## Layout

```
train_async_grpo_1gpu.py   trainer entry point (AsyncGRPOConfig, GSM8K, accuracy_reward; --weight-sync lora|file|ipc)
weight_sync.py             same-GPU full-weight sync clients (file reload, CUDA IPC) replacing TRL's NCCL client
attn_compat.py             attention-kernel probe and fallback for sm_120 (CLI: python attn_compat.py)
eval_gsm8k.py              GSM8K test accuracy (offline vLLM, greedy) for a base model, adapter, or full checkpoint
scripts/serve_vllm.sh      vLLM server with single-GPU flags (SYNC mode, memory cap, localhost only)
scripts/run_1gpu.sh        one command: server in background, trainer in foreground, logs in runs/<ts>/
scripts/check_env.py       preflight: driver, torch/CUDA build, versions, free VRAM, attention probe
scripts/wait_for_server.py waits for /health (stdlib only; the box has no curl)
tests/                     CPU tests: SDPA fallback (packed tree = dense attention per sample), file-sync shards + RPC
docs/research.md           notes from reading the article and the TRL 1.18.2 source
```

## Setup (on the GPU box)

Needs an NVIDIA driver ≥ 580 (torch 2.14 / vLLM 0.33 are CUDA 13 builds) and a C compiler for Triton
(`sudo apt install build-essential`). No CUDA toolkit is needed: the FlashInfer sampler, which would JIT with nvcc,
is turned off.

```bash
# on the Mac
scripts/sync.sh push                      # rsync code to gpu5090:~/src/trl_rl (GPU_HOST=gpu5090-lan if Tailscale is off)
# on the box
cd ~/src/trl_rl
~/.local/bin/uv sync --frozen             # .venv with the pinned stack (uv.lock), uv-managed CPython 3.12
.venv/bin/python scripts/check_env.py --attn
```

Gotchas this repo already handles. Keep them in mind if you change the stack:

- FA4 gives wrong gradients on sm_120. `attn_compat.py` picks FA2.
- vLLM `/server_info` returns 500 in a uv venv with no pip on PATH. `pip` is a dependency, and the scripts export PATH.
- vLLM's FlashInfer top-k/top-p sampler needs nvcc. `VLLM_USE_FLASHINFER_SAMPLER=0` turns it off.

## Run

```bash
scripts/run_1gpu.sh --max-steps 5 --train-size 64    # smoke test
scripts/run_1gpu.sh                                  # 200 steps
setsid nohup scripts/run_1gpu.sh > runs/last.out 2>&1 &   # detached, survives SSH drops
```

Or start the server and the trainer in separate terminals:

```bash
scripts/serve_vllm.sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python train_async_grpo_1gpu.py --max-steps 5
```

Useful knobs. Server settings are environment variables for `serve_vllm.sh`; trainer settings are flags
for `train_async_grpo_1gpu.py`:

| Knob | Default | Note |
|---|---|---|
| `GPU_MEM_UTIL` | 0.35 | vLLM's share of the whole GPU. Lower it if the trainer OOMs; raise it for more concurrent rollouts |
| `MAX_MODEL_LEN` | 2048 | prompt + completion cap, and the default per-row token budget for training |
| `--max-staleness` | 4 | how many steps behind a sample may be. Server needs `MAX_LORAS >= max_staleness + 2` |
| `--lora-r` | 32 | server needs `MAX_LORA_RANK >= r` |
| `--learning-rate` | 5e-5 | LoRA LR. TRL's full-FT GSM8K example uses 1e-5 |
| `--score-centering` | off | TRL's default loss. Off because with one API server its extra logprobs slow generation |
| `--attn-backend` | auto | `trl` (FA4), `fa2`, `sdpa`, or `auto` to probe |

Evaluate on GSM8K test (offline vLLM, greedy; run it when no training is using the GPU):

```bash
.venv/bin/python eval_gsm8k.py --boxed-hint                                        # base model
.venv/bin/python eval_gsm8k.py --boxed-hint --adapter runs/<ts>/output/final_adapter
```

Training prompts end with "put your final answer within \boxed{}" (`--no-boxed-hint` turns it off).
Without it, the first RL steps mostly teach the model to box its answer, not math. See [docs/results.md](docs/results.md).

What to watch in the logs: `reward`, `rollout/queue_size`, `perf/rollout_wait_s` (trainer starved = generation-bound),
`rollout/backpressure_s` (generation held back = trainer-bound), `sample/staleness_mean`, `perf/weight_sync_s`.

## Security note

The vLLM server runs with `VLLM_SERVER_DEV_MODE=1`, which exposes pause, weight-loading, adapter-loading and
`/collective_rpc` endpoints. `serve_vllm.sh` binds it to `127.0.0.1` only. Other local users on the box can still
reach localhost, so stop the server when you're not training (`run_1gpu.sh` stops it on exit).
`SYNC=ipc` goes further: it turns on `VLLM_ALLOW_INSECURE_SERIALIZATION`, so a local user could run code as you.
On a shared machine, use `lora` or `file`, or block other users' access to the port (for example a per-uid
firewall rule on loopback).

## License

[Apache-2.0](LICENSE), the same license as TRL.
