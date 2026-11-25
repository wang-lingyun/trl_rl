"""Full-weight (non-LoRA) sync from the trainer to a vLLM server on the SAME GPU.

AsyncGRPOTrainer's built-in full-weight sync (`trl.generation.weight_transfer.WeightTransferClient`) broadcasts over
NCCL, which refuses two ranks on one device. Both classes here keep its interface (`init_weight_transfer`,
`send_weights(iterator)`, `pause`, `resume`, `destroy`), so they drop in as `trainer.weight_transfer` and the
trainer's own `_sync_weight_merged` drives them unchanged.

- `FileReloadWeightTransfer` ("file"): write the weights as bf16 safetensors into a private tmpfs dir, then have vLLM
  reload them in place via `POST /collective_rpc {"method": "reload_weights", "kwargs": {"weights_path": dir}}`. Only
  JSON strings go over HTTP; the server needs nothing beyond `VLLM_SERVER_DEV_MODE=1`.
- `IPCWeightTransfer` ("ipc"): vLLM's own CUDA-IPC backend (`--weight-transfer-config '{"backend":"ipc"}'`). The trainer
  packs weights into a GPU buffer and shares a CUDA IPC handle; the vLLM worker maps that same memory and loads from
  it. No host copies, so it's the fastest. But the handle travels as a pickle, so the server must run with
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1`. Anyone who can reach the server's port can then run code as the server's
  user. Only use it where no other local user can connect.
"""

from __future__ import annotations

import base64
import logging
import os
import pickle
import shutil
import tempfile
import time

import torch


logger = logging.getLogger(__name__)

SYNC_MODES = ("lora", "file", "ipc")


class _SameGPUWeightTransfer:
    def __init__(self, vllm_client, dtype: torch.dtype = torch.bfloat16):
        self.vllm = vllm_client  # trl.generation.async_vllm_client.AsyncVLLMClient
        self.dtype = dtype
        self.last_send_s: float | None = None

    def pause(self) -> None:
        self.vllm.pause()

    def resume(self) -> None:
        self.vllm.resume()

    def destroy(self) -> None:
        pass


class FileReloadWeightTransfer(_SameGPUWeightTransfer):
    """Sync full weights through safetensors files in a private RAM-backed dir and vLLM's in-place reload."""

    def __init__(self, vllm_client, dtype=torch.bfloat16, root: str = "/dev/shm", shard_bytes: int = 2 << 30):
        super().__init__(vllm_client, dtype)
        self.root = root if os.path.isdir(root) else tempfile.gettempdir()
        self.shard_bytes = shard_bytes
        self.dir: str | None = None

    def init_weight_transfer(self) -> None:
        self.vllm.wait_for_server_ready()
        # mkdtemp creates the dir 0700 with O_EXCL: safe in a shared /dev/shm on a multi-user box.
        self.dir = tempfile.mkdtemp(prefix="trl_rl_weights_", dir=self.root)
        logger.info(f"[weight_sync:file] staging dir {self.dir}")

    def write_shards(self, iterator) -> list[str]:
        from safetensors.torch import save_file

        written, shard, size = [], {}, 0

        def flush():
            nonlocal shard, size
            path = os.path.join(self.dir, f"model-{len(written) + 1:05d}.safetensors")
            save_file(shard, path)
            written.append(path)
            shard, size = {}, 0

        for name, tensor in iterator:
            t = tensor.detach().to(self.dtype).contiguous().cpu()
            shard[name] = t
            size += t.numel() * t.element_size()
            if size >= self.shard_bytes:
                flush()
        if shard:
            flush()
        # Drop shards left over from a previous sync that had more of them (vLLM loads every *.safetensors here).
        keep = set(written)
        for leftover in os.listdir(self.dir):
            path = os.path.join(self.dir, leftover)
            if path not in keep and leftover.endswith(".safetensors"):
                os.remove(path)
        return written

    def send_weights(self, iterator) -> None:
        import requests

        t0 = time.time()
        self.write_shards(iterator)
        t_write = time.time()
        response = requests.post(
            f"{self.vllm.server_url}/collective_rpc",
            json={"method": "reload_weights", "kwargs": {"weights_path": self.dir}},
            timeout=self.vllm.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"reload_weights failed: {response.status_code} {response.text[:500]}")
        self.last_send_s = time.time() - t0
        logger.info(f"[weight_sync:file] write {t_write - t0:.2f}s + reload {time.time() - t_write:.2f}s")

    def destroy(self) -> None:
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            self.dir = None


class IPCWeightTransfer(_SameGPUWeightTransfer):
    """Sync full weights with vLLM's CUDA-IPC backend (packed into a reusable GPU buffer)."""

    def __init__(self, vllm_client, dtype=torch.bfloat16, buffer_bytes: int = 1 << 30):
        super().__init__(vllm_client, dtype)
        self.buffer_bytes = buffer_bytes

    def init_weight_transfer(self) -> None:
        if "expandable_segments:true" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").replace(" ", "").lower():
            raise RuntimeError(
                "CUDA IPC from PyTorch's expandable segments makes vLLM fetch the buffer's file descriptor with "
                "pidfd_getfd, which kernel.yama.ptrace_scope=1 (Ubuntu's default) refuses between unrelated "
                "processes ('pidfd_getfd: Operation not permitted'). Run the trainer with "
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False (scripts/run_1gpu.sh does this for SYNC=ipc)."
            )
        self.vllm.wait_for_server_ready()
        try:
            self.vllm.init_weight_transfer_engine({"packed": True})
        except Exception as exc:
            raise RuntimeError(
                "Could not initialise vLLM's IPC weight transfer. Start the server with SYNC=ipc "
                "scripts/serve_vllm.sh (--weight-transfer-config '{\"backend\":\"ipc\"}' and "
                "VLLM_ALLOW_INSECURE_SERIALIZATION=1)."
            ) from exc

    def send_weights(self, iterator) -> None:
        from vllm.distributed.weight_transfer.packed_tensor import packed_ipc_producer

        t0 = time.time()
        gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        cast = ((name, tensor.detach().to(self.dtype)) for name, tensor in iterator)
        self.vllm.start_weight_update()
        try:
            # The producer reuses one buffer, so each chunk must be consumed before the next is packed. The HTTP
            # call returns only after the worker has loaded the chunk, which guarantees that.
            for chunk in packed_ipc_producer(cast, gpu_uuid, post_iter_func=lambda item: item[1],
                                             buffer_size_bytes=self.buffer_bytes):
                self.vllm.update_weights(
                    {
                        "names": chunk.names,
                        "dtype_names": chunk.dtype_names,
                        "shapes": chunk.shapes,
                        "tensor_sizes": chunk.tensor_sizes,
                        "ipc_handles_pickled": base64.b64encode(pickle.dumps(chunk.ipc_handle)).decode(),
                    }
                )
        except Exception:
            try:
                self.vllm.finish_weight_update()
            except Exception:  # noqa: BLE001 - don't mask the original error
                pass
            raise
        self.vllm.finish_weight_update()
        torch.cuda.ipc_collect()
        self.last_send_s = time.time() - t0
        logger.info(f"[weight_sync:ipc] sent in {self.last_send_s:.2f}s")


def install_full_weight_sync(trainer, mode: str) -> None:
    """Replace the trainer's NCCL weight-transfer client with a same-GPU one. Call after construction, before train()."""
    if mode not in ("file", "ipc"):
        raise ValueError(f"full-weight sync mode must be 'file' or 'ipc', got {mode!r}")
    if getattr(trainer, "_lora_sync", False):
        raise ValueError("The trainer is using adapter-only LoRA sync; full-weight sync does not apply.")
    if trainer.accelerator.num_processes != 1:
        raise ValueError("Same-GPU full-weight sync is written for a single trainer process.")
    cls = FileReloadWeightTransfer if mode == "file" else IPCWeightTransfer
    trainer.weight_transfer = cls(trainer.vllm_client)
    logger.warning(f"[weight_sync] full-weight sync to vLLM via {mode!r} (same GPU, no NCCL)")
