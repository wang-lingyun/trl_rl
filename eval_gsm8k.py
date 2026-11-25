"""GSM8K test accuracy of the base model and (optionally) a trained LoRA adapter, with vLLM offline.

Run it when no training is using the GPU:

    .venv/bin/python eval_gsm8k.py --limit 500
    .venv/bin/python eval_gsm8k.py --adapter runs/<ts>/output/final_adapter --limit 500

Greedy decoding, same prompt format and reward (`trl.rewards.accuracy_reward`) as training.
"""

import argparse
import json
import os
import time


# No CUDA toolkit on the box: FlashInfer would JIT its top-k/top-p sampler with nvcc and crash the engine.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

BOXED_HINT = "Please reason step by step, and put your final answer within \\boxed{}."


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--adapter", default=None, help="LoRA adapter dir (PEFT format); omit for the base model")
    p.add_argument("--limit", type=int, default=None, help="first N test questions (default: all 1319)")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-lora-rank", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--out", default=None, help="write per-question results as JSONL here")
    p.add_argument(
        "--boxed-hint",
        action="store_true",
        help=f"append {BOXED_HINT!r} to each question. The reward only credits answers it can extract, so without "
        "this the base model is penalised for format (it boxes ~46%% of answers) as much as for wrong math",
    )
    args = p.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from trl.rewards import accuracy_reward

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    gold = [a.split("####")[-1].strip() for a in ds["answer"]]

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=args.adapter is not None,
        max_lora_rank=args.max_lora_rank,
        generation_config="vllm",  # ignore Qwen3's HF sampling defaults (top_k=20, top_p=0.95): we want pure greedy
    )
    tok = llm.get_tokenizer()
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": f"{q}\n{BOXED_HINT}" if args.boxed_hint else q}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for q in ds["question"]
    ]
    lora = LoRARequest("eval-adapter", 1, args.adapter) if args.adapter else None

    t0 = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=args.max_tokens), lora_request=lora)
    texts = [o.outputs[0].text for o in outputs]
    rewards = accuracy_reward([[{"role": "assistant", "content": t}] for t in texts], gold)
    scored = [r for r in rewards if r is not None]
    acc = sum(scored) / max(len(scored), 1)
    truncated = sum(o.outputs[0].finish_reason == "length" for o in outputs)

    label = args.adapter or "base"
    print(
        json.dumps(
            {
                "model": args.model,
                "adapter": label,
                "boxed_hint": args.boxed_hint,
                "n": len(scored),
                "boxed_frac": round(sum("\\boxed" in t for t in texts) / len(texts), 3),
                "accuracy": round(acc, 4),
                "truncated": truncated,
                "gen_seconds": round(time.time() - t0, 1),
            }
        )
    )
    if args.out:
        with open(args.out, "w") as f:
            for q, g, t, r in zip(ds["question"], gold, texts, rewards, strict=True):
                f.write(json.dumps({"question": q, "gold": g, "completion": t, "reward": r}) + "\n")


if __name__ == "__main__":
    main()
