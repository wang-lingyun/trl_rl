"""Preflight check for the GPU box: driver, GPU, torch/CUDA build, package versions, free memory.

    .venv/bin/python scripts/check_env.py            # versions + GPU state
    .venv/bin/python scripts/check_env.py --attn     # also probe the tree-attention kernels
"""

import argparse
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGES = [
    "trl",
    "vllm",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "kernels",
    "datasets",
    "math-verify",
    "flashinfer-python",
    "triton",
    "aiohttp",
]


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<{exc}>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", action="store_true", help="probe the tree-attention backends on the GPU")
    args = ap.parse_args()

    print(f"python   {sys.version.split()[0]}  ({sys.executable})")
    print(f"platform {platform.platform()}")
    if shutil.which("nvidia-smi"):
        print("\n# nvidia-smi")
        print(sh(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.used,memory.total", "--format=csv"]))
        apps = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        print("compute apps:", apps or "none")
    else:
        print("nvidia-smi not found")

    print("\n# packages")
    for name in PACKAGES:
        try:
            print(f"{name:18s} {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:18s} -")

    print("\n# torch")
    try:
        import torch

        print(f"torch {torch.__version__}, built for CUDA {torch.version.cuda}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            free, total = torch.cuda.mem_get_info()
            print(f"device: {torch.cuda.get_device_name()}  capability sm_{cap[0]}{cap[1]}")
            print(f"arch list: {torch.cuda.get_arch_list()}")
            print(f"memory free/total: {free / 2**30:.1f} / {total / 2**30:.1f} GiB")
            x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
            print(f"bf16 matmul ok: {bool(torch.isfinite(x @ x).all())}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch check failed: {type(exc).__name__}: {exc}")

    if args.attn:
        print("\n# tree-attention backends (each probed in its own process)")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from attn_compat import BACKENDS, probe_in_subprocess

        for name in BACKENDS:
            print(json.dumps(probe_in_subprocess(name)), flush=True)


if __name__ == "__main__":
    main()
