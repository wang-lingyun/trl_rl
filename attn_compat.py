"""Pick an attention kernel for AsyncGRPOTrainer's tree-packed forward that really runs on this GPU.

AsyncGRPOTrainer (TRL 1.18) packs samples into a prefix tree and runs attention through
``flash_attn_varlen_func`` from a Hub kernel (``trl.trainer.async_grpo_trainer._flash_attn``):
FlashAttention-3 below compute capability 10, FlashAttention-4 at 10 and above. FA4 targets datacenter
Blackwell (B200, sm_100). The RTX 5090 is consumer Blackwell (sm_120), which lacks some of the
instructions FA4 uses, so it may fail there.

This module tries candidate kernels on the real GPU, checks each one against a float32 reference on a
small packed tree (forward and backward), and installs the first correct one in place of TRL's choice:

    trl   - whatever TRL picks (FA4 on a 5090)
    fa2   - kernels-community/flash-attn2 (FlashAttention-2 supports sm_80 and newer)
    sdpa  - a per-segment torch SDPA loop: slower, but runs on any GPU (and on CPU)

Each probe runs in its own subprocess, because a kernel that crashes on the device can leave the
CUDA context unusable for anything that comes after it.

CLI:
    python attn_compat.py               # probe every backend, print a table
    python attn_compat.py --probe fa2   # probe one backend, print one JSON line
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)

BACKENDS = ("trl", "fa2", "sdpa")

# Tolerances for a bf16 kernel against the float32 reference.
FWD_TOL = 3e-2  # max abs error of the output
BWD_TOL = 5e-2  # max abs error of each gradient, relative to that gradient's max magnitude


# --------------------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------------------


def _segment_mask(lq: int, lk: int, causal: bool, window_size, device) -> torch.Tensor | None:
    """Boolean mask (True = attend) for one segment, causal aligned bottom-right like FlashAttention."""
    left, right = window_size if window_size is not None else (-1, -1)
    if not causal and left < 0 and right < 0:
        return None
    qpos = torch.arange(lk - lq, lk, device=device)[:, None]  # absolute position of each query in the path
    kpos = torch.arange(lk, device=device)[None, :]
    mask = torch.ones(lq, lk, dtype=torch.bool, device=device)
    if causal:
        mask &= kpos <= qpos
    if left >= 0:
        mask &= kpos >= qpos - left
    if right >= 0:
        mask &= kpos <= qpos + right
    return mask


def sdpa_varlen_attn(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,
    max_seqlen_k=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=None,
    **_kwargs,
):
    """Drop-in for ``flash_attn_varlen_func`` built on ``F.scaled_dot_product_attention``.

    ``q`` is ``(total_q, heads, dim)``; ``k`` and ``v`` are ``(total_k, kv_heads, dim)`` (GQA allowed).
    Segment ``i`` takes queries ``cu_seqlens_q[i]:cu_seqlens_q[i+1]`` and keys ``cu_seqlens_k[i]:...``.
    A segment may have fewer queries than keys (its ancestors' tokens come first); the causal mask is
    aligned bottom-right, as in FlashAttention >= 2.1.
    """
    if softcap:
        raise NotImplementedError("The SDPA fallback does not support attention logit softcapping.")
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    gqa = q.shape[1] != k.shape[1]
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i] : cq[i + 1]].transpose(0, 1)  # (H, Lq, D)
        ki = k[ck[i] : ck[i + 1]].transpose(0, 1)  # (Hkv, Lk, D)
        vi = v[ck[i] : ck[i + 1]].transpose(0, 1)
        lq, lk = qi.shape[1], ki.shape[1]
        if lq > lk:
            raise ValueError(f"segment {i} has more queries ({lq}) than keys ({lk})")
        mask = _segment_mask(lq, lk, causal, window_size, q.device)
        oi = F.scaled_dot_product_attention(qi, ki, vi, attn_mask=mask, scale=softmax_scale, enable_gqa=gqa)
        outs.append(oi.transpose(0, 1))
    return torch.cat(outs, dim=0)


def reference_varlen_attn(
    q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=None, causal=True, window_size=(-1, -1), **_kwargs
):
    """Plain float32 matmul + softmax. Ground truth for the probe; slow and memory hungry on purpose."""
    q, k, v = q.float(), k.float(), v.float()
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    rep = q.shape[1] // k.shape[1]
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    outs = []
    for i in range(len(cq) - 1):
        qi = q[cq[i] : cq[i + 1]].transpose(0, 1)
        ki = k[ck[i] : ck[i + 1]].transpose(0, 1).repeat_interleave(rep, dim=0)
        vi = v[ck[i] : ck[i + 1]].transpose(0, 1).repeat_interleave(rep, dim=0)
        scores = qi @ ki.transpose(-1, -2) * scale
        mask = _segment_mask(qi.shape[1], ki.shape[1], causal, window_size, q.device)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        outs.append((scores.softmax(-1) @ vi).transpose(0, 1))
    return torch.cat(outs, dim=0)


# --------------------------------------------------------------------------------------------------
# Test case: a small packed prefix tree, laid out like TreeRow.pack() in TRL
# --------------------------------------------------------------------------------------------------


def make_tree_case(
    seg_lens=(37, 20, 9, 5, 12),
    parents=(-1, 0, 0, 2, -1),
    heads=16,
    kv_heads=8,
    head_dim=128,
    device="cpu",
    dtype=torch.float32,
    seed=0,
):
    """Segments in DFS pre-order; ``parents[i]`` is the parent segment (-1 for a root).

    Returns packed ``q, k, v`` plus ``kv_index`` / ``cu_seqlens_q`` / ``cu_seqlens_k`` and each segment's
    full root-to-segment ``paths`` (packed positions), which is what attention for that segment must see.
    Defaults mirror Qwen3-0.6B's attention shape (16 query heads, 8 KV heads, head_dim 128).
    """
    gen = torch.Generator().manual_seed(seed)
    starts, paths, pos = [], [], 0
    for length, parent in zip(seg_lens, parents, strict=True):
        own = list(range(pos, pos + length))
        paths.append((paths[parent] if parent >= 0 else []) + own)
        starts.append(pos)
        pos += length
    n = pos
    q = torch.randn(n, heads, head_dim, generator=gen).to(device=device, dtype=dtype)
    k = torch.randn(n, kv_heads, head_dim, generator=gen).to(device=device, dtype=dtype)
    v = torch.randn(n, kv_heads, head_dim, generator=gen).to(device=device, dtype=dtype)
    kv_index = torch.tensor([p for path in paths for p in path], dtype=torch.long, device=device)
    cu_q = torch.tensor([0, *torch.tensor(seg_lens).cumsum(0).tolist()], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, *torch.tensor([len(p) for p in paths]).cumsum(0).tolist()], dtype=torch.int32, device=device)
    return {
        "q": q,
        "k": k,
        "v": v,
        "kv_index": kv_index,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "max_seqlen_q": max(seg_lens),
        "max_seqlen_k": max(len(p) for p in paths),
        "paths": paths,
        "starts": starts,
        "seg_lens": list(seg_lens),
    }


def _run_varlen(fn, case, q, k, v):
    """Call a varlen kernel the way trl's ``tree_attention`` does (K/V gathered along each segment's path)."""
    out = fn(
        q,
        k.index_select(0, case["kv_index"]),
        v.index_select(0, case["kv_index"]),
        cu_seqlens_q=case["cu_seqlens_q"],
        cu_seqlens_k=case["cu_seqlens_k"],
        max_seqlen_q=case["max_seqlen_q"],
        max_seqlen_k=case["max_seqlen_k"],
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        window_size=(-1, -1),
    )
    return out[0] if isinstance(out, tuple) else out


def check_kernel(module, device="cuda", dtype=torch.bfloat16) -> dict:
    """Run ``module.flash_attn_varlen_func`` forward + backward on a packed tree and compare to float32."""
    case = make_tree_case(device=device, dtype=dtype)
    leaves = [case[name].detach().clone().requires_grad_(True) for name in ("q", "k", "v")]
    refs = [case[name].detach().float().clone().requires_grad_(True) for name in ("q", "k", "v")]

    out = _run_varlen(module.flash_attn_varlen_func, case, *leaves)
    ref = _run_varlen(reference_varlen_attn, case, *refs)
    weight = torch.randn(ref.shape, generator=torch.Generator().manual_seed(1)).to(device)
    (out.float() * weight).sum().backward()
    (ref * weight).sum().backward()
    if device == "cuda":
        torch.cuda.synchronize()

    fwd_err = (out.float() - ref).abs().max().item()
    bwd_errs = []
    for leaf, r in zip(leaves, refs, strict=True):
        if leaf.grad is None:
            bwd_errs.append(float("inf"))
            continue
        scale = r.grad.abs().max().clamp_min(1e-6)
        bwd_errs.append(((leaf.grad.float() - r.grad).abs().max() / scale).item())
    finite = bool(torch.isfinite(out).all()) and all(e == e and e != float("inf") for e in bwd_errs)
    ok = finite and fwd_err < FWD_TOL and max(bwd_errs) < BWD_TOL
    return {"ok": ok, "fwd_err": fwd_err, "bwd_rel_err": max(bwd_errs)}


# --------------------------------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------------------------------


def load_backend(name: str):
    """Return an object with ``flash_attn_varlen_func`` for the named backend (may download from the Hub)."""
    if name == "trl":
        import trl.trainer.async_grpo_trainer as agt

        original = getattr(agt, "_trl_rl_original_flash_attn", agt._flash_attn)
        return original()
    if name == "fa2":
        from kernels import get_kernel

        return get_kernel("kernels-community/flash-attn2")
    if name == "sdpa":
        return types.SimpleNamespace(flash_attn_varlen_func=sdpa_varlen_attn)
    raise ValueError(f"unknown attention backend {name!r}, expected one of {BACKENDS}")


def probe_in_subprocess(name: str, timeout: float = 600.0) -> dict:
    """Probe one backend in a fresh Python process so a crashing kernel can't poison this CUDA context."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--probe", name]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"backend": name, "ok": False, "error": f"probe timed out after {timeout:.0f}s"}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return {"backend": name, "ok": False, "error": f"exit {proc.returncode}: " + " | ".join(tail)}


def install_attention_backend(choice: str = "auto") -> str:
    """Make AsyncGRPOTrainer's tree attention use a kernel that works on this GPU. Returns the backend name.

    ``choice="auto"`` probes ``trl``, then ``fa2``, then ``sdpa`` (each in a subprocess) and keeps the
    first that passes. A named choice is used as is, without probing.
    """
    import trl.trainer.async_grpo_trainer as agt

    if not hasattr(agt, "_trl_rl_original_flash_attn"):
        agt._trl_rl_original_flash_attn = agt._flash_attn

    if choice == "auto":
        for name in BACKENDS:
            result = probe_in_subprocess(name)
            logger.warning(f"[attn] probe {name}: {json.dumps(result)}")
            if result.get("ok"):
                choice = name
                break
        else:
            raise RuntimeError("No attention backend passed the probe, not even the SDPA fallback.")

    if choice == "trl":
        agt._flash_attn = agt._trl_rl_original_flash_attn
    else:
        module = load_backend(choice)
        agt._flash_attn = lambda: module
    logger.warning(f"[attn] tree attention backend: {choice}")
    return choice


def _probe_main(name: str) -> None:
    result = {"backend": name}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        result["device"] = torch.cuda.get_device_name()
        result["capability"] = ".".join(map(str, torch.cuda.get_device_capability()))
        result.update(check_kernel(load_backend(name)))
    except Exception as exc:  # noqa: BLE001 - report every failure as a failed probe
        result.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", choices=BACKENDS, help="probe one backend in this process and print JSON")
    cli = parser.parse_args()
    if cli.probe:
        _probe_main(cli.probe)
    else:
        for backend in BACKENDS:
            print(json.dumps(probe_in_subprocess(backend)), flush=True)
