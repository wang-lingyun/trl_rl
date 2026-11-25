"""CPU tests for the SDPA fallback: packed tree attention must equal plain causal attention per sample.

Runs anywhere torch runs (no GPU, no TRL needed):  python -m pytest tests -q
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attn_compat import (  # noqa: E402
    _run_varlen,
    check_kernel,
    make_tree_case,
    reference_varlen_attn,
    sdpa_varlen_attn,
)


def dense_path_attention(case, seg):
    """Ordinary causal attention over the whole root-to-segment path, as one unpacked sequence."""
    path = torch.tensor(case["paths"][seg])
    q, k, v = (case[n][path].transpose(0, 1) for n in ("q", "k", "v"))  # (H, L, D)
    rep = q.shape[0] // k.shape[0]
    out = F.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, 0), v.repeat_interleave(rep, 0), is_causal=True
    ).transpose(0, 1)
    return out[-case["seg_lens"][seg] :]  # this segment's own tokens are the end of its path


@pytest.mark.parametrize("fn", [sdpa_varlen_attn, reference_varlen_attn], ids=["sdpa", "reference"])
@pytest.mark.parametrize(
    "tree",
    [
        dict(seg_lens=(37, 20, 9, 5, 12), parents=(-1, 0, 0, 2, -1)),  # prompt, two branches, a nested fork, 2nd root
        dict(seg_lens=(8,), parents=(-1,)),  # one plain sequence
        dict(seg_lens=(3, 1, 1, 1), parents=(-1, 0, 0, 0)),  # one-token branches
    ],
)
def test_packed_tree_matches_dense(fn, tree):
    case = make_tree_case(**tree, heads=4, kv_heads=2, head_dim=16)
    packed = _run_varlen(fn, case, case["q"], case["k"], case["v"])
    for seg, start in enumerate(case["starts"]):
        own = packed[start : start + case["seg_lens"][seg]]
        torch.testing.assert_close(own, dense_path_attention(case, seg), atol=1e-5, rtol=1e-5)


def test_sliding_window_matches_reference():
    case = make_tree_case(heads=4, kv_heads=2, head_dim=16)
    kwargs = dict(cu_seqlens_q=case["cu_seqlens_q"], cu_seqlens_k=case["cu_seqlens_k"], causal=True, window_size=(7, 0))
    kq, kv = case["k"].index_select(0, case["kv_index"]), case["v"].index_select(0, case["kv_index"])
    torch.testing.assert_close(
        sdpa_varlen_attn(case["q"], kq, kv, **kwargs),
        reference_varlen_attn(case["q"], kq, kv, **kwargs),
        atol=1e-5,
        rtol=1e-5,
    )


def test_probe_accepts_sdpa_on_cpu():
    import types

    result = check_kernel(types.SimpleNamespace(flash_attn_varlen_func=sdpa_varlen_attn), device="cpu")
    assert result["ok"], result


def test_probe_rejects_wrong_kernel():
    import types

    def non_causal(q, k, v, **kwargs):  # forgets the causal mask: must fail the probe
        kwargs["causal"] = False
        return sdpa_varlen_attn(q, k, v, **kwargs)

    result = check_kernel(types.SimpleNamespace(flash_attn_varlen_func=non_causal), device="cpu")
    assert not result["ok"], result
