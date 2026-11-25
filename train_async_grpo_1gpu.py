"""Minimal async GRPO on ONE GPU with TRL's AsyncGRPOTrainer: Qwen3-0.6B + LoRA on GSM8K.

Upstream assumes two GPUs: vLLM on one, the trainer on the other, with weights pushed over NCCL.
NCCL can't put two ranks on the same device, so on one GPU this script trains a LoRA adapter and
lets TRL's adapter-only sync ship it to vLLM as files over HTTP (`/v1/load_lora_adapter`), with no
NCCL. vLLM is capped with `--gpu-memory-utilization` so the trainer fits beside it.

Start the server first (`scripts/serve_vllm.sh`), or use `scripts/run_1gpu.sh` which does both.
"""

import argparse
import logging
import os


logger = logging.getLogger("train_async_grpo_1gpu")


def gsm8k_to_prompt(sample):
    """GSM8K row -> conversational prompt + gold answer (the number after '####')."""
    return {
        "prompt": [{"role": "user", "content": sample["question"]}],
        "solution": sample["answer"].split("####")[-1].strip(),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--vllm-url", default="http://127.0.0.1:8000")
    p.add_argument("--output-dir", default="outputs/qwen3-0.6b-gsm8k-lora")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=5e-5, help="LoRA usually wants ~5-10x the full-FT LR")
    p.add_argument("--per-device-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--max-completion-length", type=int, default=1024)
    p.add_argument("--max-staleness", type=int, default=4, help="server needs --max-loras >= this + 2")
    p.add_argument("--lora-r", type=int, default=32, help="server needs --max-lora-rank >= this")
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument(
        "--score-centering",
        action="store_true",
        help="TRL's default loss. Off here: with LoRA the server runs one API server, where the extra "
        "top-k logprobs it needs roughly halve generation throughput",
    )
    p.add_argument("--train-size", type=int, default=None, help="use only the first N GSM8K train rows")
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--report-to", default="none", help="none | trackio | wandb | tensorboard")
    p.add_argument("--attn-backend", default="auto", choices=["auto", "trl", "fa2", "sdpa"])
    p.add_argument("--log-completions", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Pick a tree-attention kernel that works on this GPU (FA4 may not run on an RTX 5090 / sm_120).
    from attn_compat import install_attention_backend

    attn_backend = install_attention_backend(args.attn_backend)

    from datasets import load_dataset
    from peft import LoraConfig

    from trl import AsyncGRPOConfig, AsyncGRPOTrainer
    from trl.rewards import accuracy_reward

    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if args.train_size:
        dataset = dataset.select(range(min(args.train_size, len(dataset))))
    dataset = dataset.map(gsm8k_to_prompt, remove_columns=dataset.column_names)

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
        peft_config=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, target_modules="all-linear"),
    )

    # On one GPU the merged-weight fallback would open an NCCL group between two processes on the same
    # device, which NCCL refuses ("Duplicate GPU detected"). Fail early with the real reason instead.
    if not getattr(trainer, "_lora_sync", False):
        raise SystemExit(
            "Adapter-only LoRA sync is OFF, so TRL would sync merged weights over NCCL, which cannot work with "
            "vLLM and the trainer on the same GPU. Start the server with scripts/serve_vllm.sh (it sets "
            "--enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=1) and check the warnings above."
        )
    logger.info(f"single-GPU async GRPO: adapter-only sync ON, attention backend = {attn_backend}")

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final_adapter")
    trainer.save_model(final_dir)
    logger.info(f"saved final LoRA adapter to {final_dir}")


if __name__ == "__main__":  # the rollout worker is a spawned process that re-imports this file
    main()
