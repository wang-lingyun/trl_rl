"""Minimal async GRPO on ONE GPU with TRL's AsyncGRPOTrainer: Qwen3-0.6B on GSM8K, LoRA or full fine-tuning.

Upstream assumes two GPUs: vLLM on one, the trainer on the other, with weights pushed over NCCL. NCCL can't put two
ranks on the same device, so this script syncs weights to a vLLM server on the same GPU in one of three ways
(`--weight-sync`, and start the server with the matching `SYNC=` for scripts/serve_vllm.sh):

  lora  train a LoRA adapter. TRL's adapter-only sync writes it to disk and loads it with POST /v1/load_lora_adapter.
  file  full fine-tuning. Weights go through bf16 safetensors in a private /dev/shm dir, and vLLM reloads them in place
        with POST /collective_rpc reload_weights. No pickle; ~1 s per sync for 0.6B.
  ipc   full fine-tuning. vLLM's CUDA-IPC backend shares the trainer's GPU buffer, so there are no host copies, but the
        handles travel as a pickle (VLLM_ALLOW_INSECURE_SERIALIZATION=1): any local user who can reach the port can
        run code as you. Only on a box where nobody else can connect.

vLLM is capped with `--gpu-memory-utilization` so the trainer fits beside it. Use `scripts/run_1gpu.sh`, which starts
both processes.
"""

import argparse
import logging
import os


logger = logging.getLogger("train_async_grpo_1gpu")


BOXED_HINT = "Please reason step by step, and put your final answer within \\boxed{}."


def gsm8k_to_prompt(sample, boxed_hint=True):
    """GSM8K row -> conversational prompt + gold answer (the number after '####').

    With `boxed_hint`, the question ends with the usual boxed-answer instruction. Without it, base Qwen3-0.6B boxes
    only ~44% of answers, and the first RL steps mostly learn that format (see docs/results.md) rather than math.
    """
    question = f"{sample['question']}\n{BOXED_HINT}" if boxed_hint else sample["question"]
    return {
        "prompt": [{"role": "user", "content": question}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--output-dir", default="outputs/qwen3-0.6b-gsm8k")
    p.add_argument("--weight-sync", default="lora", choices=["lora", "file", "ipc"],
                   help="lora = adapter-only (default); file / ipc = full fine-tuning (see module docstring)")
    p.add_argument("--weight-sync-steps", type=int, default=1, help="optimizer steps between weight syncs")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=None,
                   help="default 5e-5 for LoRA, 1e-5 for full fine-tuning (TRL's Qwen3-0.6B GSM8K example)")
    p.add_argument("--per-device-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-staleness", type=int, default=4, help="LoRA: server needs --max-loras >= this + 2")
    p.add_argument("--lora-r", type=int, default=32, help="server needs --max-lora-rank >= this")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument(
        "--score-centering",
        action="store_true",
        help="TRL's default loss. Off here: with LoRA the server runs one API server, where the extra "
        "top-k logprobs it needs roughly halve generation throughput",
    )
    p.add_argument("--train-size", type=int, default=None, help="use only the first N GSM8K train rows")
    p.add_argument("--no-boxed-hint", action="store_true", help="plain questions, without the \\boxed{} instruction")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--report-to", default="none", help="none | trackio | wandb | tensorboard")
    p.add_argument("--attn-backend", default="auto", choices=["auto", "trl", "fa2", "sdpa"])
    p.add_argument("--log-completions", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 5e-5 if args.weight_sync == "lora" else 1e-5
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    use_lora = args.weight_sync == "lora"

    # Pick a tree-attention kernel that works on this GPU (FA4 gives wrong gradients on an RTX 5090 / sm_120).
    from attn_compat import install_attention_backend

    attn_backend = install_attention_backend(args.attn_backend)

    from datasets import load_dataset
    from peft import LoraConfig

    from trl import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.rewards import accuracy_reward

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if args.train_size:
        dataset = dataset.select(range(min(args.train_size, len(dataset))))
    dataset = dataset.map(
        gsm8k_to_prompt, fn_kwargs={"boxed_hint": not args.no_boxed_hint}, remove_columns=dataset.column_names
    )

    config = AsyncGRPOConfig(
        output_dir=args.output_dir,
        vllm_server_base_url=args.vllm_url,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_staleness=args.max_staleness,
        weight_sync_steps=args.weight_sync_steps,
        score_centering=args.score_centering,
        chat_template_kwargs={"enable_thinking": False},  # short answers; Qwen3 thinking would blow the budget
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=1,
        report_to=args.report_to,
        log_completions=args.log_completions,
        num_completions_to_print=4,
        seed=args.seed,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model,
        args=config,
        train_dataset=dataset,
        reward_funcs=accuracy_reward,
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear")
        if use_lora
        else None,
    )

    if use_lora:
        # On one GPU the merged-weight fallback would open an NCCL group between two processes on the same
        # device, which NCCL refuses ("Duplicate GPU detected"). Fail early with the real reason instead.
        if not getattr(trainer, "_lora_sync", False):
            raise SystemExit(
                "Adapter-only LoRA sync is OFF, so TRL would sync merged weights over NCCL, which cannot work with "
                "vLLM and the trainer on the same GPU. Start the server with SYNC=lora scripts/serve_vllm.sh (it "
                "sets --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1) and check the warnings above."
            )
    else:
        # Full fine-tuning: swap TRL's NCCL client for a same-GPU one (file reload or CUDA IPC).
        from weight_sync import install_full_weight_sync

        install_full_weight_sync(trainer, args.weight_sync)
    logger.info(f"single-GPU async GRPO: weight sync = {args.weight_sync}, attention backend = {attn_backend}")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter" if use_lora else "final_model")
    trainer.save_model(final_dir)
    trainer.state.save_to_json(os.path.join(args.output_dir, "trainer_state.json"))  # full per-step log_history
    logger.info(f"saved final {'LoRA adapter' if use_lora else 'model'} to {final_dir}")


if __name__ == "__main__":  # the rollout worker is a spawned process that re-imports this file
    main()
