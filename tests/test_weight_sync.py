"""CPU tests for FileReloadWeightTransfer: shards hold every weight in bf16, stale shards go, the RPC is right."""

import os
import sys
import types
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("safetensors")
from safetensors.torch import load_file  # noqa: E402

from weight_sync import FileReloadWeightTransfer  # noqa: E402


def fake_client():
    return types.SimpleNamespace(server_url="http://127.0.0.1:8000", timeout=5, wait_for_server_ready=lambda: None)


def weights(n=6, size=64):
    gen = torch.Generator().manual_seed(0)
    return {f"model.layers.{i}.mlp.up_proj.weight": torch.randn(size, size, generator=gen) for i in range(n)}


def test_shards_roundtrip_in_bf16(tmp_path):
    w = weights()
    sync = FileReloadWeightTransfer(fake_client(), root=str(tmp_path), shard_bytes=3 * 64 * 64 * 2)  # 3 tensors/shard
    sync.init_weight_transfer()
    try:
        paths = sync.write_shards(iter(w.items()))
        assert len(paths) == 2
        assert oct(os.stat(sync.dir).st_mode & 0o777) == "0o700"
        loaded = {}
        for p in paths:
            loaded.update(load_file(p))
        assert loaded.keys() == w.keys()
        for name, tensor in w.items():
            assert loaded[name].dtype == torch.bfloat16
            torch.testing.assert_close(loaded[name], tensor.to(torch.bfloat16))
    finally:
        sync.destroy()
    assert sync.dir is None


def test_fewer_shards_remove_stale_ones(tmp_path):
    sync = FileReloadWeightTransfer(fake_client(), root=str(tmp_path), shard_bytes=64 * 64 * 2)  # 1 tensor/shard
    sync.init_weight_transfer()
    try:
        sync.write_shards(iter(weights(n=4).items()))
        sync.shard_bytes = 1 << 30
        paths = sync.write_shards(iter(weights(n=4).items()))
        assert sorted(os.listdir(sync.dir)) == [os.path.basename(p) for p in paths] == ["model-00001.safetensors"]
    finally:
        sync.destroy()


def test_send_weights_calls_reload_rpc(tmp_path, monkeypatch):
    calls = []

    def fake_post(url, json, timeout):
        calls.append((url, json))
        return types.SimpleNamespace(status_code=200, text="")

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    sync = FileReloadWeightTransfer(fake_client(), root=str(tmp_path))
    sync.init_weight_transfer()
    try:
        sync.send_weights(iter(weights(n=2).items()))
    finally:
        sync.destroy()
    assert calls == [
        ("http://127.0.0.1:8000/collective_rpc", {"method": "reload_weights", "kwargs": {"weights_path": calls[0][1]["kwargs"]["weights_path"]}})
    ]
    assert all(isinstance(v, str) for v in calls[0][1]["kwargs"].values())  # the route only passes strings
